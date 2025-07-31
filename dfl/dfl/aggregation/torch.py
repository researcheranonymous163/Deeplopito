import asyncio
from typing import Callable, override, Any

import torch

from dfl import Meta
from . import Aggregator, PartitioningAggregator


class SumTensorsAggregator(Aggregator):
	def __init__(self):
		super().__init__()
		
		self._sum: torch.Tensor | None = None
		self._lock = asyncio.Lock()
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		async with self._lock:
			if self._sum is None:
				self._sum = await asyncio.to_thread(torch.zeros_like, data)
			
			await asyncio.to_thread(self._sum.add_, data)
	
	@override
	async def aggregate(self) -> torch.Tensor:
		async with self._lock:
			return self._sum


class AvgTensorsAggregator(Aggregator):
	def __init__(self):
		super().__init__()
		
		self._sum: torch.Tensor | None = None
		self._lock = asyncio.Lock()
		self._n = 0
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		async with self._lock:
			if self._sum is None:
				self._sum = await asyncio.to_thread(torch.zeros_like, data)
			
			self._n += 1
			await asyncio.to_thread(self._sum.add_, data)
	
	@override
	async def aggregate(self) -> torch.Tensor:
		async with self._lock:
			return await asyncio.to_thread(lambda: self._sum / self._n)


class WeightedAvgTensorsAggregator(Aggregator):
	def __init__(self, weight_meta_key: str = 'weight'):
		super().__init__()
		
		self._weight_meta_key = weight_meta_key
		
		self._sum: torch.Tensor | None = None
		self._lock = asyncio.Lock()
		self._ws = 0
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		async with self._lock:
			if self._sum is None:
				self._sum = await asyncio.to_thread(torch.zeros_like, data)
			
			w = meta[self._weight_meta_key]
			await asyncio.to_thread(self._sum.add_, data, alpha=w)
			self._ws += w
	
	@override
	async def aggregate(self) -> torch.Tensor:
		async with self._lock:
			return await asyncio.to_thread(lambda: self._sum / self._ws)


class StateDictAggregator(PartitioningAggregator):
	def __init__(self, param_agg_factory: Callable[..., Aggregator] | None = None, param_meta_key: str = 'name'):
		if param_agg_factory is None:
			param_agg_factory = AvgTensorsAggregator
		
		super().__init__(partition_agg_factory=param_agg_factory, partition_meta_key=param_meta_key)
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		if not isinstance(data, dict):
			raise ValueError(f"Unexpected data type `{type(data)}`.")
		
		for param_name, v in data.items():
			param_meta = meta.copy() | {self._partition_meta_key: param_name}
			await super().add(param_meta, v)


class TensorToDeviceDecoratingAggregator(Aggregator):
	def __init__(self, device, agg: Aggregator):
		super().__init__()
		self._device = device
		self._agg = agg
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		data = await asyncio.to_thread(data.to, device=self._device, non_blocking=True)
		await self._agg.add(meta, data)
	
	@override
	async def aggregate(self) -> Any:
		return await self._agg.aggregate()
