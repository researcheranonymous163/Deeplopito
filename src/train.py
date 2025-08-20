import dfl
from sympy.matrices.expressions.matadd import matrix_of
from torch.distributed.rpc.internal import serialize

from .dataloader import DataLoader , DataManager

from .partitions import FinalPartition
from .Communicate import get_communicator , CommunicatorType , Communicator
import logging
logging.basicConfig(level=logging.INFO)
communicator:Communicator = None
import asyncio
import torch
import random
import uuid
import os
from .logger import write_time_execution
from pathlib import Path
client_number = int(os.environ.get('CLIENT_NUMBER'))
batch_size = int(os.environ.get('BATCH_SIZE'))
import time
import gc
import sys
import pickle
from config_folder import priorities ,stages_client
from .constants import MODEL_BERT
model_name = os.environ.get("CURRENT_MODEL")

def write_log_time(runner_client,owner_client, current_stage  , type , time):
    write_time_execution(f"{runner_client}-{owner_client}-{current_stage}-{type}:{time}")
# this class handle train of partitions assigned to the client
class Train:
    def __init__(self, client_id:int, partitionsOfClient,matrix):
        self.client_id = client_id
        # key structure is runner_stage_distance
        self.partitionsOfClient = partitionsOfClient
        # last partition actually is loss calculation
        self.numPartitions = len(self.partitionsOfClient) -1
        self.dataManager = self.dataLoaderMnist()
        self.dataStoreLabels = {}
        self.dataStore = []
        self.dataStoreAttentionMasks = {}
        self.matrix = matrix
        self.events = [] # for save time stamp execute times
        # logging.info(f"partition number is {self.partitionNumber}")

        self.stages_client = stages_client[self.client_id -1 ]

    async def get_communicator(self):
        if not communicator:
            await self.initial_communicator()

    def addEvent(self, round_number, partition, client_number, start_time, end_time, communication):
        group = "communication" if communication else f"{client_number}"

        self.events.append(
            dict(RunnerClient=f"{self.client_id}_round_{round_number}",
                 Start=start_time,
                 Finish=end_time,
                 Duration=end_time - start_time,
                 Group=group,
                 key=f'{client_number}_{partition}')
        )

    def write_tasks(self):
        import csv
        from typing import TextIO

        file_name = "log_time/activities_clients.csv"

        with open(file_name, mode='w', newline='') as csvfile:  # type: TextIO
            partition = 'key'
            fieldnames = ['RunnerClient', 'Start', 'Finish','Duration', 'Group', ('%s' % partition)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for task in self.events:
                writer.writerow(task)

        print(f"✅ Tasks written successfully to {file_name}")



            

    async def initial_communicator(self):
        global communicator
        communicator = await get_communicator(CommunicatorType.GRPC)

    def dataLoaderMnist(self):
        global client_number
        data_loader = DataLoader( batch_size=batch_size, shuffle=True, index_client=client_number)
        data_loader.set_mode("train")
        data_manager = DataManager(data_loader)
        self.test_loader = data_loader.test_loader
        return data_manager

    def select_next_client(self, client, partition, message_type):
        client -= 1
        partition -=1

        if not (0 <= client < len(self.matrix)):
            raise IndexError("Row index out of range.")

        if not (0 <= partition < len(self.matrix[client])):
            raise IndexError("Column index out of range.")

        # Get the previous value or return -1 if it's the first column
        prev_value = self.matrix[client][partition - 1] if partition - 1 >= 0 else -1

        # Get the next value or return -1 if it's the last column
        next_value = self.matrix[client][partition + 1] if partition + 1 < len(self.matrix[client]) else -1

        if message_type == "forward":
            return next_value
        else:
            return prev_value

    async def request_push(self,destination,data):
        client_id = data['client_id']
        stage = data['stage']

        if destination == f"client{self.client_id}":
            key_stage = f"{client_id}_{stage}"
            communicator.myQueue[key_stage] = data
        else:
            await communicator.push(destination=destination, data=data)

    async def train_models(self,num_epochs,current_round ,num_partitions):
        loss_count= 0
        start_time_train = time.perf_counter()
        global client_number
        broad_cast_done = False
        self.dataManager.reset_dataset()
        global communicator


        current_index_stage = 0


        while True:
            current_stage = self.stages_client[current_index_stage]
            # logging.critical(f"current stage:{current_stage},myQueue:{communicator.myQueue.keys()}")

            if current_stage in communicator.myQueue:
                # keys_queue = list(communicator.myQueue.keys())
                # key_stage = keys_queue[0]
                data = communicator.myQueue[current_stage]
                del communicator.myQueue[current_stage]
                if current_index_stage == len(self.stages_client) - 1:
                    current_index_stage = 0
                else:
                    current_index_stage += 1
                x = data["output"]
                partition = data["partition"]
                batch_id = data["batch_id"]
                client_id = data["client_id"]
                message_type = data["message_type"]
                trace = data["trace"]
                epoch = data["epoch"]
                stage = data["stage"]

                if model_name == MODEL_BERT and partition >= 2:
                    attention_mask = self.dataStoreAttentionMasks[batch_id]
                else:
                    attention_mask = None
                # backward
                if message_type == "backward":
                    oldGradient = data.get("gradient")
                    key = f"{self.client_id}_{partition}_{client_id}"
                    start_time =time.time()
                    newGradient = await asyncio.to_thread(self.partitionsOfClient[key].backward, oldGradient, batch_id)
                    end_time = time.time()
                    # self.addEvent(current_round, stage,f"{client_id}",start_time,end_time,False)
                    if partition == 2:
                        # stage 1 for every client handle by same client => key is client_number_1_client_number
                        destination_client = f"client{client_id}"
                    else:
                        next_client = self.select_next_client(client_id,partition, "backward")
                        destination_client = f"client{next_client}"
                    partition -= 1
                    stage +=1

                    if partition == 0:
                        self.dataStore.remove(batch_id)
                        del self.dataStoreLabels[batch_id]
                        if model_name == MODEL_BERT and batch_id in self.dataStoreAttentionMasks:
                            del self.dataStoreAttentionMasks[batch_id]
                    else:
                        data_value = {
                            "output": x,
                            "gradient": newGradient,
                            "partition": partition,
                            "source": client_id,
                            "batch_id": batch_id,
                            "client_id": client_id,
                            "message_type": "backward",
                            "trace": trace,
                            "epoch":epoch,
                            "stage":stage
                        }
                        start_push = time.time()
                        await self.request_push(destination_client, data_value)
                        end_push = time.time()
                        # self.addEvent(current_round,f"communication_{stage}",f"{client_id}",start_push,end_push,True)
                    del oldGradient
                    del newGradient
                    # gc.collect()

                # forward proccess
                else:
                    key = f"{self.client_id}_{partition}_{client_id}"
                    start_time = time.time()
                    # logging.critical(f"forward  for stage {stage}")
                    if model_name == MODEL_BERT and partition >= 2:
                        output = await asyncio.to_thread(self.partitionsOfClient[key].forward, x, batch_id,attention_mask=attention_mask)
                    else:
                        output = await asyncio.to_thread(self.partitionsOfClient[key].forward,x , batch_id)
                    end_time =time.time()
                    # self.addEvent(current_round, stage,f"{client_id}",start_time,end_time,False)
                    stage +=1
                    # if is last partition have to calculate loss and go for backward
                    if partition == num_partitions:
                        loss, gradFinal = await asyncio.to_thread(
                            self.partitionsOfClient["finalPartition"].compute_loss_and_grad,
                            output, self.dataStoreLabels[batch_id])
                        data_value = {
                            "output": loss,
                            "gradient": gradFinal,
                            "partition": num_partitions,
                            "epoch": epoch,
                            "source": client_id,
                            "batch_id": batch_id,
                            "client_id": client_id,
                            "message_type": "backward",
                            "trace": trace,
                            "stage":stage
                        }
                        logging.critical(f"loss: for item {loss_count}: {loss}")
                        loss_count+=1
                        start_push = time.time()
                        await self.request_push(f"client{client_id}", data_value)
                        end_push = time.time()
                        # self.addEvent(current_round, stage,f"{client_id}",start_push,end_push,True)
                    else:
                        next_client = self.select_next_client(client_id, partition, "forward")
                        partition += 1
                        destination = f"client{next_client}"
                        data_value = {
                            "output": output,
                            "partition": partition,
                            "epoch":epoch,
                            "client_id": client_id,
                            "batch_id": batch_id,
                            "message_type": "forward",
                            "trace": trace,
                            "stage": stage
                        }
                        start_push = time.time()
                        await self.request_push(destination, data_value)
                        end_push = time.time()
                        # self.addEvent(current_round, stage,f"{client_id}",start_push,end_push,True)


            # read new data => (have to change and read data after complete backward)
            if self.dataManager.epoch < num_epochs:
                # until last batch complete backward => do not read new batch
                if not self.dataStore:
                    random_id = str(uuid.uuid4())

                    features, labels = self.dataManager.next_batch()

                    if model_name == MODEL_BERT :
                        input_ids, attention_mask = features
                        x = input_ids
                        self.dataStoreAttentionMasks[random_id] = attention_mask
                    else:
                        x = features

                    stage_key = f"{self.client_id}_{1}"
                    communicator.myQueue[stage_key] = {
                        "output": x,
                        "partition": 1,
                        "client_id": self.client_id,
                        "batch_id": random_id,
                        "message_type": "forward",
                        "epoch":self.dataManager.epoch,
                        "trace": f"{self.dataManager.batch_count}",
                        "stage":1
                    }
                    del x
                    self.dataStoreLabels[random_id] = labels
                    self.dataStore.append(random_id)

                    del features, labels
                    if model_name == MODEL_BERT:
                        del input_ids, attention_mask
            else:
                if not broad_cast_done and not communicator.myQueue and not self.dataStoreLabels:
                    for key in self.partitionsOfClient:
                        if key!="finalPartition":
                            model_path = Path(f"/model/model_{key}.pt")
                            sd = await self.partitionsOfClient[key].getStateDict()
                            cpu_sd = {k: v.cpu() for k, v in sd.items()}

                            meta, data = {'round': current_round, "key": key}, cpu_sd
                            await communicator.broadcast(meta, data)
                    # break
                    broad_cast_done = True
                    return
            await asyncio.sleep(0)


