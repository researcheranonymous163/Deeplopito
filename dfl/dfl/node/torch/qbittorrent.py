import asyncio
import functools
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Callable, Awaitable, override

import torch

from dfl import Meta, Aggregator, Segmenter
from dfl.communication.qbittorrent import QbittorrentCommunicator
from dfl.extensions.frozendict import frozendict
from dfl.extensions.torch import Context as TorchContext
from . import Node


class SegmentsAsMultiFileTorrentQbittorrentCommunicatorNode(Node):
	@override
	def __init__(
			self,
			logger: logging.Logger,
			torch_context: TorchContext,
			communicator: QbittorrentCommunicator,
			select_neighbors: Callable[..., Awaitable[Iterable[Meta]]],
			param_agg_factory: Callable[..., Aggregator],
			segmenter: Segmenter,
			pull_early=True,
			models_dir_path: Path = Path('./models/'),
			cleanup_rounds_age: int | None = 7,
			extra_publish_meta: dict | None = None,
	):
		super().__init__(
			logger=logger,
			torch_context=torch_context,
			communicator=communicator,
			select_neighbors=select_neighbors,
			param_agg_factory=param_agg_factory,
			segmenter=segmenter,
			pull_early=pull_early,
			cleanup_rounds_age=cleanup_rounds_age,
			extra_publish_meta=extra_publish_meta
		)
		
		self._models_dir_path = models_dir_path
	
	@override
	async def _select_neighbors(self, **kwargs) -> Iterable[Meta]:
		metas = await super()._select_neighbors(break_segments=False, **kwargs)
		
		normalized_metas = []
		
		# Merge those differing only in the `segment` or the `segments` entry.
		merged_metas_segments = defaultdict(lambda: {'segments': set()})
		for meta in metas:
			if 'segments' in meta:
				segments = meta.get('segments')
			elif 'segment' in meta:
				segments = (meta['segment'],)
			else:
				normalized_metas.append(meta)
				continue
			
			key = frozenset((k, v) for k, v in meta.items() if k not in {'segment', 'segments'})
			merged_metas_segments[key]['segments'].update(segments)
		
		for k, v in merged_metas_segments.items():
			normalized_metas.append(dict(k, **{'segments': tuple(v['segments'])}))
		
		return normalized_metas
	
	@override
	async def publish(self):
		self._logger.debug("Publishing...")
		
		sd = await asyncio.to_thread(self._torch_context.model.state_dict)
		segments = await self._segmenter.segment(sd)
		
		model_dir_path = self._models_dir_path / f'round-{self.round_}/'
		await asyncio.to_thread(model_dir_path.mkdir, exist_ok=False)
		if self._cleanup_rounds_age is not None:
			await self.register_round_cleanup(functools.partial(asyncio.to_thread, shutil.rmtree, model_dir_path))
		
		# TODO: this can be done concurrently per segment.
		for segment_name, segment in segments.items():
			segment_path = model_dir_path / f"{segment_name}.pt"
			await asyncio.to_thread(torch.save, segment, segment_path)
		
		meta = {'from': self._communicator.name, 'round': self.round_}
		
		frozen_meta = frozendict(meta)
		self._logger.debug(f"Publishing {frozen_meta}...")
		
		await self._communicator.publish(meta | self._extra_publish_meta, model_dir_path, meta_id=meta)
		
		self._logger.debug(f"Published {frozen_meta}.")
		
		if self._cleanup_rounds_age is not None:
			await self.register_round_cleanup(functools.partial(self._communicator.unpublish, meta))
		
		self._logger.debug("Published.")
	
	@override
	async def _pull_one(self, meta: Meta) -> None:
		frozen_meta = frozendict(meta)
		self._logger.debug(f"Pulling {frozen_meta}...")
		
		files = None
		segments = meta.get('segments')
		if segments is not None:
			files = tuple(f"{segment}.pt" for segment in segments)
			meta = meta.copy()
			del meta['segments']
		meta, data = await self._communicator.subscribe(meta, files=files)
		
		if isinstance(data, Path):
			await self.register_round_cleanup(functools.partial(asyncio.to_thread, SegmentsAsMultiFileTorrentQbittorrentCommunicatorNode._rm_tmp, data))
		
		if await asyncio.to_thread(data.is_dir):
			data_segments = dict()
			for f in data.iterdir():
				f_data = await asyncio.to_thread(torch.load, f, weights_only=True)
				data_segments[f.with_suffix('').name] = f_data
			data = data_segments
		elif await asyncio.to_thread(data.is_file):
			assert len(segments) == 1
			
			data = await asyncio.to_thread(torch.load, data, weights_only=True)
			segment = next(iter(segments))
			data = {segment: data}
		else:
			raise ValueError(f"Unexpected file type, neither regular nor directory: {data}.")
		
		await (await self._agg()).add(meta, data)
		
		self._logger.debug(f"Pulled {frozen_meta}.")
	
	@staticmethod
	def _rm_tmp(p: Path) -> None:
		if not p.exists():
			return
		
		if p.is_dir():
			shutil.rmtree(p)
		else:
			os.remove(p)


def register_qbittorrent_communicator_manual_unsubscribe_as_node_cleanup(communicator: QbittorrentCommunicator, node: Node):
	assert communicator._unsubscribe_strategy == 'manual'
	
	# FIXME: wrap by a decorator instead of monkey-patching.
	
	original_subscribe = communicator.subscribe
	
	async def patched_subscribe(meta: Meta, *args, **kwargs):
		return_meta, data = await original_subscribe(meta=meta, *args, **kwargs)
		await node.register_round_cleanup(functools.partial(communicator.unsubscribe, meta), round_=meta.get('round', None))
		return return_meta, data
	
	communicator.subscribe = patched_subscribe
