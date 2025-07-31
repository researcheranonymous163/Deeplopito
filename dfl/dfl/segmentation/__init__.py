from typing import Any, override, OrderedDict


class Segmenter:
	async def segment(self, model: Any) -> dict[Any, Any] | list[Any]:
		raise NotImplementedError
	
	async def de_segment(self, segmented_model: dict[Any, Any] | list[Any]) -> Any:
		raise NotImplementedError


class AlwaysDictSegmenterDecorator(Segmenter):
	def __init__(self, segmenter: Segmenter, expects_list_only=None, always_str_key=False, str_keys: dict[str, Any] | None = None):
		self._segmenter = segmenter
		self._expects_list_only = expects_list_only
		self._always_str_key = always_str_key
		self._str_keys = str_keys
	
	@override
	async def segment(self, model: Any) -> dict[Any, Any]:
		segments = await self._segmenter.segment(model)
		if isinstance(segments, list):
			if self._expects_list_only is None:
				self._expects_list_only = True
			
			segments = OrderedDict[int, Any](sorted({k: v for k, v in enumerate(segments)}.items()))
		
		if self._always_str_key:
			if self._str_keys is None:
				self._str_keys = {str(k): k for k, v in segments.items()}
			segments = {str(k): v for k, v in segments.items()}
		
		return segments
	
	@override
	async def de_segment(self, segmented_model: dict[Any, Any]) -> Any:
		if self._always_str_key:
			segmented_model = {self._str_keys[k]: v for k, v in segmented_model.items()}
		
		if self._expects_list_only:
			segmented_model = sorted(segmented_model.items())
			segmented_model = [(int(k), v)[1] for k, v in segmented_model]
		
		return await self._segmenter.de_segment(segmented_model)
