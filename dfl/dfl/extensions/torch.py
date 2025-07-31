import asyncio
import contextlib

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import dfl.extensions.asyncio


@contextlib.asynccontextmanager
async def false_requires_grad(model: nn.Module):
	try:
		await asyncio.to_thread(model.requires_grad_, False)
		yield
	finally:
		# FIXME: preserve the original old state? Check how `torch.no_grad` works.
		await asyncio.to_thread(model.requires_grad_, True)


def eval_batch(model, batch, target, criterion=None, device=None) -> int | tuple[int, float]:
	if device is not None:
		batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
	
	output = model(batch)
	
	_, predicted = torch.max(output.data, 1)
	# noinspection PyUnresolvedReferences
	corrects = (predicted == target).sum().item()
	
	if criterion is not None:
		loss = criterion(output, target)
		return corrects, loss
	else:
		return corrects


async def test_eval(model: nn.Module, data_loader: DataLoader, criterion=None, device=None) -> tuple[int, int] | tuple[int, int, float]:
	model.eval()
	
	corrects, total = 0, 0
	if criterion is not None:
		losses_sum = 0
	
	async with false_requires_grad(model):
		async for batch, target in dfl.extensions.asyncio.iter_to_aiter(iter(data_loader)):
			if criterion is not None:
				batch_corrects, batch_loss = await asyncio.to_thread(eval_batch, model, batch, target, criterion=criterion, device=device)
				
				corrects += batch_corrects
				# noinspection PyUnboundLocalVariable
				losses_sum += batch_loss * len(target)
			else:
				batch_corrects = await asyncio.to_thread(eval_batch, model, batch, target, device=device)
				
				corrects += batch_corrects
			
			total += len(target)
	
	if criterion is not None:
		# noinspection PyUnboundLocalVariable
		return corrects, total, float(losses_sum / total)  # FIXME: return the raw losses sum, not the average.
	else:
		return corrects, total


def train_batch(model, optimizer, criterion, batch, target, device=None) -> tuple[int, int, float]:
	if device is not None:
		batch, target = batch.to(device, non_blocking=True), target.to(device, non_blocking=True)
	
	optimizer.zero_grad()
	output = model(batch)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	
	_, predicted = torch.max(output.data, 1)
	# noinspection PyUnresolvedReferences
	corrects = (predicted == target).sum().item()
	
	return corrects, len(target), float(loss)


async def train_epoch(model, optimizer, criterion, data_loader, device=None, epilogue_zero_grad: bool = True) -> tuple[int, int, float]:
	running_corrects, running_total, running_losses_sum = 0, 0, 0
	async for batch, target in dfl.extensions.asyncio.iter_to_aiter(iter(data_loader)):
		corrects, total, loss = await asyncio.to_thread(train_batch, model, optimizer, criterion, batch, target, device=device)
		
		running_corrects += corrects
		running_total += total
		running_losses_sum += loss * total
	
	if epilogue_zero_grad:
		optimizer.zero_grad()  # Don't let gradients of the last batch leak, memory and misleading-usage wise.
	
	return running_corrects, running_total, running_losses_sum


async def train(
		model: nn.Module,
		optimizer: Optimizer,
		criterion,
		data_loader: DataLoader,
		epochs: int | None = None,
		device=None,
		epilogue_zero_grad: bool = True,
) -> list[tuple[int, int, float]]:
	model.train()
	
	epochs_metrics = []
	done_epochs = 0
	while epochs is None or done_epochs < epochs:
		metrics = await train_epoch(model, optimizer, criterion, data_loader, device=device, epilogue_zero_grad=False)
		epochs_metrics.append(metrics)
		
		done_epochs += 1
	
	if epilogue_zero_grad:
		optimizer.zero_grad()
	
	return epochs_metrics


class Context:
	def __init__(
			self,
			model: nn.Module,
			optimizer: Optimizer,
			criterion,
			train_data_loader: DataLoader,
			test_data_loader: DataLoader,
			train_eval_data_loader: DataLoader | None = None,
			device: torch.device | None = None,
	):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.train_data_loader = train_data_loader
		self.test_data_loader = test_data_loader
		self.device = device
		
		if train_eval_data_loader is None:
			train_eval_data_loader = train_data_loader
		self.train_eval_data_loader = train_eval_data_loader
	
	async def train(self, epochs: int | None = None):
		return await train(self.model, self.optimizer, self.criterion, self.train_data_loader, epochs, device=self.device)
	
	async def test_eval(self):
		return await test_eval(self.model, self.test_data_loader, criterion=self.criterion, device=self.device)
	
	async def train_eval(self):
		return await test_eval(self.model, self.train_eval_data_loader, criterion=self.criterion, device=self.device)


def len_samples_data_loader(data_loader: DataLoader) -> int:
	total_samples = 0
	for batch in data_loader:
		total_samples += len(batch)
	return total_samples
