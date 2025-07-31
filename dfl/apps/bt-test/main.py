import asyncio
import os
import random
import time
import traceback
from pathlib import Path

import nats

import dfl.net

_srand = random.SystemRandom()


def generate_random_file(num_bytes: int, output_file):
	random_bytes = _srand.randbytes(num_bytes)
	with open(output_file, 'wb') as f:
		f.write(random_bytes)


self_name = os.environ['NAME']

all_neighbors = [f'n{i}' for i in range(10)]
all_neighbors.remove(self_name)


async def amain():
	nats_client = await nats.connect('nats://nats:4222')
	downloads_dir = Path('/downloads/')
	downloads_dir.mkdir(parents=True, exist_ok=True)
	
	net_node = await dfl.net.QbittorrentNatsNetworkNode(name=self_name, nats_client=nats_client, downloads_dir_path=downloads_dir)
	round_num = 0
	
	while True:
		rnd_file = downloads_dir / f"{self_name}-{round_num}.bin"
		# noinspection PyTypeChecker
		await asyncio.to_thread(rnd_file.parent.mkdir, parents=True, exist_ok=True)
		await asyncio.to_thread(generate_random_file, num_bytes=1 * 1024 ** 2, output_file=rnd_file)
		
		await net_node.push(({'round': round_num}, rnd_file))
		
		round_neighbors = random.sample(all_neighbors, k=3)
		
		ts = []
		async with asyncio.TaskGroup() as tg:
			for neighbor in round_neighbors:
				ts.append(tg.create_task(net_node.pull({'from': neighbor, 'round': round_num})))
		
		for t in ts:
			data = await t
			print(f"Received {data}.")
		
		round_num += 1
		await asyncio.sleep(7)


try:
	asyncio.run(amain())
except Exception as e:
	traceback.print_exception(e)

time.sleep(9999999)
