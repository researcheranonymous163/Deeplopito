import asyncio
import atexit
import pprint
import random
import re
from logging import Logger
from pathlib import Path
from typing import Literal, Iterable

import data
import dfl
import nats
import numpy as np
import torch
from dfl import Communicator
from dfl.aggregation import TensorToDeviceDecoratingAggregator
from dfl.aggregation import WeightedAvgTensorsAggregator
from dfl.communication import LocalCommunicator
from dfl.communication.grpc import GrpcCommunicator
from dfl.communication.nats import NatsCommunicator
from dfl.communication.qbittorrent import QbittorrentCommunicator
from dfl.extensions.torch import Context as TorchContext
from dfl.segmentation import AlwaysDictSegmenterDecorator, KEvenTorchStateDictSegmenter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from loggings import get_logger
from model import mnist_2nn


async def build(
		name: str,
		index: int | None = None,
		logger: Logger | None = None,
		data_distribution: Literal['stratified', 'one_ten'] = 'stratified',
		communicator: Communicator | Literal['local', 'qbittorrent-nats', 'grpc', 'qbittorrent-grpc'] = 'local',
		nodes_count: int = 10,
		data_dir_path: Path | None = None
) -> dfl.Node:
	if index is None:
		# Use the first number inside the name as the index.
		match = re.search(r'\d+', name)
		index = int(match.group()) if match else None
		if index is None:
			raise ValueError("No index.")
		
		if index < 0 or index > nodes_count:
			raise ValueError(f"Node index {index} is out of range [0, nodes_count={nodes_count}).")
	
	if logger is None:  # Default logger.
		logger = await asyncio.to_thread(get_logger, name=name)
	
	# Decide the device.
	device = torch.device('cuda') if await asyncio.to_thread(torch.cuda.is_available) else torch.device('cpu')
	logger.debug(f"Using device {device}.")
	
	# Prepare the train and test data partitions and their loaders.
	
	if data_distribution == 'stratified':
		train_data, test_data = data.stratified(index, n_splits=nodes_count)
	elif data_distribution == 'one_ten':
		if nodes_count != 10:
			logger.warning(f"Using one-ten data distribution between {nodes_count} (not 10) nodes.")
		train_data, test_data = data.one_ten(index)
	else:
		raise ValueError(f"Invalid data distribution value \"{data_distribution}\".")
	
	collate_fn = lambda x: tuple(x_.to(device) for x_ in default_collate(x))  # A mapper to move loaded tensors into the device automatically.
	train_data_loader = DataLoader(train_data, batch_size=100, shuffle=True, collate_fn=collate_fn)
	test_data_loader = DataLoader(test_data, batch_size=100, collate_fn=collate_fn)
	
	# Just some logs about the classes' distribution of the train and test data partitions.
	
	unique_targets, unique_targets_counts = np.unique([data.train_data_all.targets[i] for i in train_data.indices], return_counts=True)
	unique_targets, unique_targets_counts = map(str, unique_targets), map(str, unique_targets_counts)
	targets_counts = dict(zip(unique_targets, unique_targets_counts))
	logger.debug(f"My train targets' counts: {targets_counts}.")
	logger.debug(f"Here are my first 5 indices of the train dataset: {train_data.indices[:min(5, len(train_data.indices))]}.")
	
	unique_targets, unique_targets_counts = np.unique([data.test_data_all.targets[i] for i in test_data.indices], return_counts=True)
	unique_targets, unique_targets_counts = map(str, unique_targets), map(str, unique_targets_counts)
	targets_counts = dict(zip(unique_targets, unique_targets_counts))
	logger.debug(f"My test targets' counts: {targets_counts}.")
	logger.debug(f"Here are my first 5 indices of the test dataset: {test_data.indices[:min(5, len(test_data.indices))]}.")
	
	# Prepare the torch context.
	
	model = mnist_2nn()
	model = await asyncio.to_thread(model.to, device)
	
	torch_context = TorchContext(
		model=model,
		optimizer=Adam(model.parameters(), lr=1e-4),
		criterion=CrossEntropyLoss(),
		train_data_loader=train_data_loader,
		test_data_loader=test_data_loader,
	)
	
	# Prepare the communicator.
	if not isinstance(communicator, Communicator):
		if communicator == 'local':
			communicator = await LocalCommunicator(name=name)
		elif communicator == 'qbittorrent-nats':
			nats_client = await nats.connect('nats://nats:4222')
			nats_comm = await NatsCommunicator(name, client=nats_client)
			
			downloads_dir_path = data_dir_path / 'downloads/'
			await asyncio.to_thread(downloads_dir_path.mkdir)
			
			communicator = await QbittorrentCommunicator(name, meta_communicator=nats_comm, downloads_dir_path=downloads_dir_path, unsubscribe_strategy='manual')
		elif communicator == 'grpc':
			communicator = await GrpcCommunicator(name)
		elif communicator == 'qbittorrent-grpc':
			grpc_comm = await GrpcCommunicator(name)
			
			downloads_dir_path = data_dir_path / 'downloads/'
			await asyncio.to_thread(downloads_dir_path.mkdir)
			
			communicator = await QbittorrentCommunicator(name, grpc_comm, downloads_dir_path=downloads_dir_path, unsubscribe_strategy='manual')
		else:
			raise ValueError(f"Invalid communicator type value \"{communicator}\".")
		
		atexit.register(asyncio.run, communicator.close())
	
	# Prepare the model segmenter.
	segmenter = AlwaysDictSegmenterDecorator(KEvenTorchStateDictSegmenter(k=3))
	
	# Prepare the neighbors' selection function.
	
	all_neighbors = [f'node-{i}' for i in range(nodes_count)]
	all_neighbors.remove(name)
	
	model_segments = (await segmenter.segment(model.state_dict())).keys() if segmenter is not None else None
	
	async def select_neighbors(**kwargs) -> Iterable[dfl.Meta]:
		if model_segments is not None:
			segments_neighbors = dict()
			for segment in model_segments:
				neighbors = random.sample(all_neighbors, len(all_neighbors) - 1)
				segments_neighbors[segment] = neighbors
			
			# FIXME: add round # to the log.
			logger.info(
				f"Selected the following neighbors per segments:\n"
				f"{pprint.pformat(segments_neighbors, compact=True, sort_dicts=True)}.",
				extra={'type': 'segments-neighbors', 'segments-neighbors': segments_neighbors}
			)
			
			flattened = []
			for segment, neighbors in segments_neighbors.items():
				for neighbor in neighbors:
					flattened.append({'from': neighbor, 'segment': segment})
			
			return flattened
		else:
			neighbors = random.sample(all_neighbors, len(all_neighbors) - 1)
			
			# FIXME: add round # to the log.
			logger.info(f"Selected the following neighbors: {neighbors}", extra={'type': 'neighbors', 'neighbors': neighbors})
			
			return [{'from': neighbor} for neighbor in neighbors]
	
	# Prepare the model's parameter aggregator factory.
	def param_agg_factory(**kwargs):
		return TensorToDeviceDecoratingAggregator(device, WeightedAvgTensorsAggregator(weight_meta_key='data_len'))
	
	# Build the node instance.
	if isinstance(communicator, QbittorrentCommunicator):
		models_dir_path = data_dir_path / 'models/'
		await asyncio.to_thread(models_dir_path.mkdir)
		
		node = await dfl.node.SegmentsAsMultiFileTorrentQbittorrentCommunicatorNode(
			logger=logger,
			torch_context=torch_context,
			communicator=communicator,
			select_neighbors=select_neighbors,
			param_agg_factory=param_agg_factory,
			segmenter=segmenter,
			models_dir_path=models_dir_path,
			cleanup_age_rounds=3,
			extra_publish_meta={'data_len': len(train_data)}
		)
		
		dfl.node.register_qbittorrent_communicator_manual_unsubscribe_as_node_cleanup(communicator, node)
	else:
		node = await dfl.Node(
			logger=logger,
			torch_context=torch_context,
			communicator=communicator,
			select_neighbors=select_neighbors,
			param_agg_factory=param_agg_factory,
			segmenter=segmenter,
			cleanup_age_rounds=3,
			extra_publish_meta={'data_len': len(train_data)}
		)
	
	return node
