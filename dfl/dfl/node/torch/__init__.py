import asyncio
import functools
import io
import logging
from collections import defaultdict
from typing import Iterable, Callable, Awaitable

import torch
from codetiming import Timer

from dfl import Communicator
from dfl import Meta, Aggregator, Segmenter
from dfl.aggregation.torch import StateDictAggregator
from dfl.extensions.frozendict import frozendict
from dfl.extensions.torch import Context as TorchContext
from dfl.segmentation import AlwaysDictSegmenterDecorator


class Node:
	def __init__(
			self,
			torch_context: TorchContext,
			communicator: Communicator,
			select_neighbors: Callable[..., Awaitable[Iterable[Meta]]],  # TODO: also allow async generators.
			param_agg_factory: Callable[..., Aggregator],
			logger: logging.Logger | None = None,
			segmenter: Segmenter | None = None,
			pull_early=True,
			cleanup_rounds_age: int | None = 7,
			extra_publish_meta: dict | None = None,
			post_train: Callable[[], Awaitable[None]] | None = None,
	):
		self._torch_context = torch_context
		self._communicator = communicator
		self._select_neighbors_ = select_neighbors
		self._param_agg_factory = param_agg_factory
		self._post_train = post_train
		
		if logger is None:
			logger = logging.root
		self._logger = logger
		
		self._segmenter = segmenter
		if self._segmenter is not None:
			self._segmenter = AlwaysDictSegmenterDecorator(self._segmenter, always_str_key=True)
		
		self._pull_early = pull_early
		
		self._cleanup_rounds_age = cleanup_rounds_age
		
		if extra_publish_meta is None:
			extra_publish_meta = {}
		self._extra_publish_meta = extra_publish_meta
		
		self.round_ = 0
		self._agg_: StateDictAggregator | None = None
		self._agg_lock = asyncio.Lock()  # TODO: use a R/W lock.
		
		if self._cleanup_rounds_age is not None:
			self._rounds_cleanups: dict[int, list[Callable[[], Awaitable[None]]]] = defaultdict(list)
			self._rounds_cleanups_lock = asyncio.Lock()
	
	async def _select_neighbors(self, break_segments=True, **kwargs) -> Iterable[Meta]:
		normalized_metas = []
		for meta in await self._select_neighbors_(**kwargs):
			# Default the round meta to the current round in the node.
			if 'round' not in meta and 'round' in kwargs:
				meta = meta | {'round': kwargs['round']}
			
			if self._segmenter is not None:
				if 'segments' in meta and 'segment' in meta:
					raise ValueError(f"Having both `segment` and `segments` is prohibited in a neighbor selection meta while a segmenter is present: {meta}.")
				
				# Because of the `dfl.AlwaysDictSegmenterDecorator(self._segmenter, always_str_key=True)` wrapper (look in `__init__`), should make sure that segments' names are `str`.
				if 'segment' in meta:
					meta = meta | {'segment': str(meta['segment'])}
				elif 'segments' in meta:
					meta = meta | {'segments': [str(segment) for segment in meta['segments']]}
				
				if break_segments and 'segments' in meta:
					# Break into multiple metas per segment.
					
					segments = meta['segments']
					meta = meta.copy()
					del meta['segments']
					
					for segment in segments:
						segment_meta = meta | {'segment': segment}
						normalized_metas.append(segment_meta)
					meta = None
			
			if meta is not None:
				normalized_metas.append(meta)
		
		return normalized_metas
	
	def _agg_factory(self, **kwargs) -> StateDictAggregator:
		return StateDictAggregator(self._param_agg_factory, param_meta_key='segment')
	
	async def _agg(self) -> StateDictAggregator:
		agg = self._agg_
		if agg is None:
			async with self._agg_lock:
				agg = self._agg_
				if agg is None:
					agg = self._agg_factory()
					self._agg_ = agg
		return agg
	
	async def __ainit__(self):
		pass
	
	def __await__(self):
		async def _coro():
			await self.__ainit__()
			return self
		
		return _coro().__await__()
	
	async def pull(self):
		self._logger.debug("Pulling...")
		
		neighbors_metas = await self._select_neighbors(round=self.round_)
		
		async with asyncio.TaskGroup() as tg:
			for neighbor_meta in neighbors_metas:
				tg.create_task(self._pull_one(neighbor_meta))
		
		self._logger.debug("Pulled.")
	
	async def _pull_one(self, meta: Meta) -> None:
		frozen_meta = frozendict(meta)
		self._logger.debug(f"Pulling {frozen_meta}...")
		
		with Timer(logger=None) as wait_timer:
			_, _ = await self._communicator.subscribe(meta, meta_only=True)  # Ensure the data availability first to log an accurate download time omitting the availability wait time.
		
		with Timer(logger=None) as timer:
			meta, data = await self._communicator.subscribe(meta)
		
		self._logger.info(f"Comm. pull {frozen_meta} took {timer.last:.2f} seconds (the wait was {wait_timer.last:.2f} seconds).", extra={'type': 'time-pull', 'time-seconds': timer.last, 'round': self.round_, 'meta': frozen_meta, 'wait-time-seconds': wait_timer.last})
		
		data = torch.load(io.BytesIO(data), weights_only=True)
		if 'segment' in meta:
			data = {meta['segment']: data}
			meta = meta.copy()
			del meta['segment']
		if not isinstance(data, dict):
			raise ValueError(f"Unexpected pulled data type {type(data)}.")
		
		await (await self._agg()).add(meta, data)
		
		self._logger.debug(f"Pulled {frozen_meta}.")
	
	async def train(self, epochs: int | None = 1):
		self._logger.debug(f"Training for {epochs} epochs...")
		
		with Timer(logger=None) as timer:
			epochs_metrics = await self._torch_context.train(epochs)
		
		self._logger.debug(f"Trained {epochs} epochs.")
		
		self._logger.info(f"The train took {timer.last:.2f} seconds.", extra={'type': 'time-train', 'time-seconds': timer.last, 'round': self.round_})
		
		epochs_metrics = [(corrects / total, losses_sum / total) for (corrects, total, losses_sum) in epochs_metrics]
		accuracies, losses = [accuracy for accuracy, _ in epochs_metrics], [loss for _, loss in epochs_metrics]
		self._logger.info(f"Raw Train Accuracies & Losses, over Epochs = {epochs_metrics}.", extra={'type': 'raw-train-accuracy', 'round': self.round_, 'epochs': epochs, 'accuracies': accuracies, 'losses': losses})
	
	async def post_train(self) -> None:
		"""
		Post model train, pre model aggregation.
		"""
		
		if self._post_train is not None:
			await self._post_train()
	
	async def train_eval(self):
		self._logger.debug("Train evaluating...")
		
		corrects, total, mean_loss = await self._torch_context.train_eval()
		accuracy = corrects / total
		self._logger.info(f"Train Accuracy & Mean Loss = {accuracy} & {mean_loss}.", extra={'type': 'train-accuracy', 'accuracy': accuracy, 'mean-loss': mean_loss, 'round': self.round_})
	
	async def publish(self):
		self._logger.debug("Publishing...")
		
		if self._segmenter is not None:
			sd = await asyncio.to_thread(self._torch_context.model.state_dict)
			segments = await self._segmenter.segment(sd)
			
			# TODO: this can be done concurrently per segment.
			for segment_name, segment in segments.items():
				segment_buff = io.BytesIO()
				await asyncio.to_thread(torch.save, segment, segment_buff)
				
				meta, data = {'from': self._communicator.name, 'round': self.round_, 'segment': segment_name}, segment_buff.getvalue()
				
				frozen_meta = frozendict(meta)
				self._logger.debug(f"Publishing {frozen_meta}...")
				
				await self._communicator.publish(meta | self._extra_publish_meta, data, meta_id=meta)
				
				self._logger.debug(f"Published {frozen_meta}.")
				
				if self._cleanup_rounds_age is not None:
					await self.register_round_cleanup(functools.partial(self._communicator.unpublish, meta))
		else:
			sd = await asyncio.to_thread(self._torch_context.model.state_dict)
			
			buff = io.BytesIO()
			torch.save(sd, buff)
			
			meta, data = {'from': self._communicator.name, 'round': self.round_}, buff.getvalue()
			
			frozen_meta = frozendict(meta)
			self._logger.debug(f"Publishing {frozen_meta}...")
			
			await self._communicator.publish(meta | self._extra_publish_meta, data, meta_id=meta)
			
			self._logger.debug(f"Published {frozen_meta}.")
			
			if self._cleanup_rounds_age is not None:
				await self.register_round_cleanup(functools.partial(self._communicator.unpublish, meta))
		
		self._logger.debug("Published.")
	
	async def aggregate(self, pull=True):
		self._logger.debug("Aggregating...")
		
		async with asyncio.TaskGroup() as tg:
			if pull:
				tg.create_task(self.pull())
			
			async def _add_self_model():
				self_model = await asyncio.to_thread(self._torch_context.model.state_dict)
				if self._segmenter is not None:
					self_model = await self._segmenter.segment(self_model)
				await (await self._agg()).add({'from': self._communicator.name, 'round': self.round_} | self._extra_publish_meta, self_model)
			
			# Add self. If segmentation is on, per this action, all parameters will be present in the aggregation.
			tg.create_task(_add_self_model())
		
		aggregated_model = await (await self._agg()).aggregate()
		self._agg_ = None
		if self._segmenter is not None:
			aggregated_model = await self._segmenter.de_segment(aggregated_model)
		missing_keys, unexpected_keys = await asyncio.to_thread(self._torch_context.model.load_state_dict, aggregated_model)
		# There should not be any unexpected key; nor missing key, as the self model would tend to all keys.
		if len(unexpected_keys) > 0 or len(missing_keys) > 0:
			self._logger.error(f"Loading the model's state-dict reported some abnormal keys; unexpected keys: {', '.join(unexpected_keys)}; missing keys: {', '.join(missing_keys)}.")
		
		self._logger.debug("Aggregated.")
	
	async def test_eval(self):
		self._logger.debug("Test evaluating...")
		
		corrects, total, mean_loss = await self._torch_context.test_eval()
		accuracy = corrects / total
		self._logger.info(f"Test Accuracy & Mean Loss = {accuracy} & {mean_loss}.", extra={'type': 'test-accuracy', 'accuracy': accuracy, 'mean-loss': mean_loss, 'round': self.round_})
	
	async def cleanup(self, all_rounds: bool = False) -> None:
		self._logger.debug(f"Cleaning up{' (all rounds)' if all_rounds else ''}...")
		
		if self._cleanup_rounds_age is None:
			return
		
		async with self._rounds_cleanups_lock:
			cleaned_rounds = []
			for round_, cleanups in self._rounds_cleanups.items():
				# TODO: optimize this iteration by using a sorted tree structure based on the round number.
				if not all_rounds and self.round_ - round_ <= self._cleanup_rounds_age:
					continue
				
				for cleanup in reversed(cleanups):
					await cleanup()
				
				cleaned_rounds.append(round_)
			
			for round_ in cleaned_rounds:
				del self._rounds_cleanups[round_]
		
		self._logger.debug(f"Cleaned up{' (all rounds)' if all_rounds else ''}.")
	
	async def register_round_cleanup(self, cleanup: Callable[[], Awaitable[None]], round_: int | None = None) -> None:
		assert self._cleanup_rounds_age is not None
		
		if round_ is None:
			round_ = self.round_
		
		async with self._rounds_cleanups_lock:
			self._rounds_cleanups[round_].append(cleanup)
	
	async def step_round(self, epochs: int | None = 1, cleanup: bool = True):
		self._logger.debug(f"Stepping round #{self.round_}...")
		
		async with asyncio.TaskGroup() as tg:
			if self._pull_early:
				tg.create_task(self.pull())
			
			await self.train(epochs)
			
			if not self._pull_early:
				tg.create_task(self.pull())
			tg.create_task(self.publish())
			tg.create_task(self.train_eval())
			tg.create_task(self.post_train())
		
		await self.aggregate(pull=False)
		
		await self.test_eval()
		
		self.round_ += 1
		if cleanup:
			await self.cleanup()
		
		self._logger.debug(f"Stepped round #{self.round_ - 1}.")
