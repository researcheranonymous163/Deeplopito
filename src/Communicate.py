from dfl.communication.encoders import ChainEncoder , PickleBytesEncoder , PathBytesEncoder
import asyncio
import dfl
from dfl.communication.grpc import GrpcCommunicator
import os
from enum import Enum
import logging
logging.basicConfig(level=logging.INFO)
from config_folder import priorities
import random
import json
class CommunicatorType(Enum):
    GRPC="GRPC"
    BITTORRENT= "BITTORRENT"


number_all_clients = int(os.environ.get("NUM_CLIENTS"))


class Communicator:
    def __init__(self ,comm_type :CommunicatorType  ,custom_handler=None):
        self.comm_type = comm_type
        self.custom_handler = custom_handler
        self.communicator :dfl.Communicator = None
        self.communication_name = os.environ.get("CLIENT_NAME")
        self.myQueue = {}

        self.count_client_finished = 0
        self.barrier_events = {}
        self.barrier_counters = {}


    def cleanup_barrier(self, barrier_id:str):
        if barrier_id in self.barrier_events:
            del self.barrier_events[barrier_id]
        if barrier_id in self.barrier_counters:
            del self.barrier_counters[barrier_id]

    def create_barrier(self, barrier_id:str):
        if barrier_id not in self.barrier_counters:
            self.barrier_events[barrier_id] = asyncio.Event()
            self.barrier_counters[barrier_id] = 0

    async def wait_at_barrier(self, barrier_id:str, title=None):
        if barrier_id not in self.barrier_events:
            self.create_barrier(barrier_id)
        await self.push("client1",{
            "barrier_id": barrier_id,
            "type_barrier":"barrier_ready"
        })
        await self.barrier_events[barrier_id].wait()
        self.cleanup_barrier(barrier_id)
        logging.critical(f"barrier of {title} set.")

    async def initial_communicator(self):
        if self.comm_type == CommunicatorType.GRPC:
            self.communicator = await GrpcCommunicator(self.communication_name, handler=self.simple_handler,grpc_options={
                'grpc.http2.max_ping_strikes':0,
                'grpc.service_config': json.dumps({
                    'methodConfig': [
                        {
                            'name': [{}],
                            'retryPolicy': {
                                'maxAttempts': 5,
                                'initialBackoff': '1s',
                                'maxBackoff': '7s',
                                'backoffMultiplier': 2,
                                'retryableStatusCodes': ['UNAVAILABLE'],
                            },
                        }
                    ]
                }),
                'grpc.server_max_unrequested_time_in_server': 7 * 60,  # 7 minutes.
             } ,encoder=ChainEncoder((PathBytesEncoder(),PickleBytesEncoder())))
        elif self.comm_type == CommunicatorType.BITTORRENT:
            self.communicator = await dfl.BittorrentCommunicator(self.communication_name, handler=self.simple_handler)

    async def push(self, destination:str, data):
        await self.communicator.send(destination , {"source": self.communication_name, "to": destination}, data)

    async def broadcast(self,meta, data):
        await self.communicator.publish(meta, data,pre_encode=True)
        
    async def pull(self,meta):
        pulled_model = await self.communicator.subscribe(meta=meta)
        return pulled_model

    async def close_connection(self):
        await self.communicator.close()




    async def simple_handler(self, meta , data):
        # if run code for recorde time communication you dont need store any data

        # logging.critical("in Main program run mod")

        if data.get("type_barrier") == "barrier_ready":
            barrier_id = data["barrier_id"]

            if barrier_id not in self.barrier_counters:
                self.create_barrier(barrier_id)
                self.barrier_counters[barrier_id] +=1
            else:
                self.barrier_counters[barrier_id]+=1


            global  number_all_clients

            if self.barrier_counters[barrier_id] == number_all_clients:
                for index in range(2,number_all_clients +1):
                    await self.push(f"client{index}",{"type_barrier":"barrier_release","barrier_id":barrier_id})

                logging.critical("finished to send them message done")
                await self.push("client1",{"type_barrier":"barrier_release","barrier_id":barrier_id})

            return

        if data.get("type_barrier") == "barrier_release":
            logging.critical("all client done .")
            barrier_id = data["barrier_id"]
            self.barrier_events[barrier_id].set()
            return

        if self.custom_handler:
            await self.custom_handler(meta , data)
        # new stage received
        else:
            client_id = int(data["client_id"])
            stage = data["stage"]
            key_stage = f"{client_id}_{stage}"
            self.myQueue[key_stage] = data

# using singltone pattern for create communicator
_communicator_instance = None
async def get_communicator(comm_type:CommunicatorType,custom_handler=None):
    global _communicator_instance
    if _communicator_instance is None:
        _communicator_instance = Communicator(comm_type,custom_handler)
        await asyncio.sleep(1)
        await _communicator_instance.initial_communicator()
    return _communicator_instance
