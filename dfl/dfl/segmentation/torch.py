from typing import override, OrderedDict

import torch

from . import Segmenter


class TorchStateDictSegmenter(Segmenter):
	@override
	async def segment(self, model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		return model
	
	@override
	async def de_segment(self, segmented_model: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
		return segmented_model


class KEvenTorchStateDictSegmenter(Segmenter):
	def __init__(self, k: int, model_params_specs: OrderedDict[str, torch.Size] | None = None):
		self.k = k
		self.model_params_specs = model_params_specs
	
	@override
	async def segment(self, model: OrderedDict[str, torch.Tensor]) -> list[torch.Tensor]:
		if self.model_params_specs is None:
			self.model_params_specs = OrderedDict[str, torch.Size]([(k, v.size()) for k, v in model.items()])
		
		model = torch.cat([p.flatten() for p in model.values()])
		
		total_size = model.numel()
		segment_size = total_size // self.k
		remainder_size = total_size % self.k
		
		segments = []
		start = 0
		for i in range(self.k):
			extra = 1 if i < remainder_size else 0
			end = start + segment_size + extra
			segments.append(model[start:end])
			start = end
		
		return segments
	
	@override
	async def de_segment(self, segmented_model: list[torch.Tensor]) -> dict[str, torch.Tensor]:
		segmented_model = torch.cat(segmented_model)
		
		offset = 0
		model = {}
		for param_name, param_size in self.model_params_specs.items():
			numel = param_size.numel()
			model[param_name] = segmented_model[offset:offset + numel].view(param_size)
			offset += numel
		
		return model
