import asyncio
import platform
from os import environ
from pathlib import Path

from torch.optim.lr_scheduler import StepLR

from node import build as build_node

NAME = environ.get('NAME', platform.node())


async def amain():
	node = await build_node(
		name=NAME,
		data_distribution='stratified',
		communicator='qbittorrent-grpc',
		data_dir_path=Path('/var/lib/dfl/')
	)
	lr_stepper = StepLR(node._torch_context.optimizer, step_size=1, gamma=0.9)
	
	# Wait for all the gRPC servers to initialize. FIXME
	await asyncio.sleep(7)
	
	while True:
		await node.step_round()
		lr_stepper.step()


asyncio.run(amain())
