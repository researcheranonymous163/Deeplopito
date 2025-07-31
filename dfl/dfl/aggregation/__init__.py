import asyncio
from typing import Callable, override, Any

from dfl import Meta


class Aggregator:
	async def add(self, meta: Meta | None = None, data=None) -> None:
		raise NotImplementedError
	
	async def aggregate(self) -> Any:
		raise NotImplementedError


class PartitioningAggregator(Aggregator):
	def __init__(self, partition_agg_factory: Callable[..., Aggregator], partition_meta_key: str = 'name'):
		super().__init__()
		self._partition_agg_factory = partition_agg_factory
		self._partition_meta_key = partition_meta_key
		self._partitions_aggs = dict()
		self._lock = asyncio.Lock()  # TODO: use a read/write lock.
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		partition = meta.get(self._partition_meta_key)
		if partition is None:
			raise ValueError(f"Data `{(meta, data)}` has no `{self._partition_meta_key}` metadata.")
		
		async with self._lock:
			agg = self._partitions_aggs.get(partition, None)
			if agg is None:
				agg = self._partition_agg_factory(**{self._partition_meta_key: partition})
				self._partitions_aggs[partition] = agg
		
		await agg.add(meta, data)
	
	@override
	async def aggregate(self) -> dict[Any, Any]:
		async with self._lock:
			return dict([(partition, await agg.aggregate()) for (partition, agg) in self._partitions_aggs.items()])
