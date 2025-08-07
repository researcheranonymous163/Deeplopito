import asyncio
import logging
import os
import gc
from asyncio import Event

import torch
import time

from .Communicate import Communicator, get_communicator, CommunicatorType
from .train import Train
from .evaluate import FullModel
from dfl.aggregation import AvgTensorsAggregator, StateDictAggregator
from .partitions import FinalPartition
from .constants import MODEL_RESNET18, MODEL_CNN , MODEL_BERT , MODEL_RESNET50
from .partition import Partition, BertPartition
from .logger import write_time_execution, write_time_values, write_to_csv
from pympler import asizeof

logging.basicConfig(level=logging.INFO)
from baseModel import  resnet_18_all_layers , resnet_50_all_layers
from baseModel import cnn_partitions, all_layers_bert
from config_folder import num_rounds, num_epochs
from config_folder import stages_client, num_clients, neighbour_clients

import logger
import shutil
import numpy as np

logging.info("inside container")
global runner_clients
global num_partitions
import time
import yaml

import torch.nn as nn
from torchvision import models
import torchvision
from collections import OrderedDict
from .logger import write_time_execution, reset_logs_time_file , write_accuracy


def write_log_time(type, time):
    write_time_execution(f"{0}-{0}-{0}-{type}:{time}")


client_number = int(os.environ.get("CLIENT_NUMBER"))
server_number = int(os.environ.get("SERVER_NUMBER"))

model_name = os.environ.get("CURRENT_MODEL")
sleep_time_initial_communicator = os.environ.get("SLEEP_TIME_COMMUNICATOR", 60)

logging.critical(f"current mode is {model_name}")

resnet = torchvision.models.resnet18(weights=None)
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 10)


def clear_folder_contents(folder="/tmp"):
    if os.path.exists(folder) and os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def get_runner_clients(stages , num_partitions):
    rn_clients = np.zeros((len(stages), num_partitions), dtype=int)
    for client_index, stages_client in enumerate(stages, start=1):
        for task in stages_client:
            owner_client, stage_of_client = map(int, task.split('_'))
            if stage_of_client > num_partitions: # about backward stages and backward do on device forward done .
                continue
            rn_clients [owner_client-1][stage_of_client-1] = client_index
    logging.critical(f"runner clients: {rn_clients}")
    return rn_clients
def force_cleanup():
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Client:
    def __init__(self):
        self.communicator: Communicator = None
        self.aggregator = None
        self.trainModel: Train = None
        self.aggregated = False
        self.timeAction = []
        self.timePullPartitions = []
        self.partition_aggregators = {}
        self.accuracies =[]

    @staticmethod
    def param_agg_factor(**kwargs):
        return AvgTensorsAggregator()

    async def initial_communicator(self):
        logging.critical("initialing communicator")
        self.communicator = await get_communicator(CommunicatorType.GRPC)

    def define_barrier(self, event: str):
        logging.critical(f"barrire of {event} inititiated . ")
        self.communicator.create_barrier(event)

    async def get_barrier_point(self, barrier_id: str, title: str):
        await self.communicator.wait_at_barrier(barrier_id=barrier_id, title=title)
        logging.critical(f"barrier of {title} set.")

    def cleanup_partition_aggregators(self):
        for partition_id in list(self.partition_aggregators.keys()):
            if self.partition_aggregators[partition_id] is not None:
                del self.partition_aggregators[partition_id]
        self.partition_aggregators.clear()
        force_cleanup()

    def initialTrainModel(self):

        partitionsOfClient = {}

        global num_partitions

        num_partitions = None
        if model_name == MODEL_BERT:
            num_partitions = len(all_layers_bert)
        elif model_name==MODEL_CNN:
            num_partitions = len(cnn_partitions)
        elif model_name == MODEL_RESNET18:
            num_partitions = len(resnet_18_all_layers)
        elif model_name == MODEL_RESNET50:
            num_partitions= len(resnet_50_all_layers)
        logging.critical(f"runner client value lenght {num_partitions}")

        global runner_clients
        logging.critical(f"number partitions :{num_partitions} and \nstages of client{stages_client}")
        runner_clients = get_runner_clients(stages=stages_client , num_partitions=num_partitions)
        self.cleanup_partition_aggregators()

        for stage_index in range(num_partitions):
            partition_id = f"partition_{stage_index + 1}"
            self.partition_aggregators[partition_id] = StateDictAggregator(
                param_agg_factory=self.param_agg_factor
            )

        for owner_client_index, row in enumerate(runner_clients):
            for stage_index, runner_client_value in enumerate(row):
                key = (
                    f"{runner_client_value}_{stage_index + 1}_{owner_client_index + 1}"
                )

                if runner_client_value == client_number:
                    if model_name == MODEL_RESNET18:
                        partitionsOfClient[key] = Partition(
                            layers=resnet_18_all_layers[stage_index]
                        )
                    elif model_name == MODEL_RESNET50:
                        partitionsOfClient[key] = Partition(
                            layers=resnet_50_all_layers[stage_index]
                        )
                    elif model_name == MODEL_CNN:
                        partitionsOfClient[key] = Partition(
                            layers=cnn_partitions[stage_index]
                        )
                    elif model_name == MODEL_BERT:
                        logging.critical(f"initialized bert{len(all_layers_bert)}")
                        partitionsOfClient[key] = BertPartition(
                            layers=all_layers_bert[stage_index],
                            is_first=(stage_index == 0),
                            is_last=(stage_index == len(all_layers_bert) - 1),
                        )

        partitionsOfClient["finalPartition"] = FinalPartition()
        self.trainModel = Train(
            client_id=client_number,
            partitionsOfClient=partitionsOfClient,
            matrix=runner_clients,
        )

    async def pull_partition(self, round, key, client_runner):
        meta = {"round": round, "key": key, "from": f"client{client_runner}"}
        pulled_model = await self.communicator.pull(meta)
        return pulled_model

    async def pullOne(self, current_round, key, runner_client):
        start_time_pull = time.time()
        (key, model_state) = await self.pull_partition(
            current_round, key, runner_client
        )

        model_state_cpu = {}
        for k, v in model_state.items():
            model_state_cpu[k] = v.detach().cpu().clone()

        del model_state
        force_cleanup()

        end_time = time.time()
        self.timePullPartitions.append(
            {
                "current_client": client_number,
                "duration_pull": end_time - start_time_pull,
                "key": key,
            }
        )

        return model_state_cpu

    async def pullModels(self, current_round: int, runner_clients):
        start_time_pull = time.perf_counter()
        logging.critical("changed to sequential")


        if client_number == server_number:
            logging.critical("pull partitions of model for server")
            aggregated_models = {}

            for stage_index in range(num_partitions):
                partition_id = f"partition_{stage_index + 1}"
                logging.critical(
                    f"Aggregating {partition_id} from {num_clients} clients"
                )

                if partition_id in self.partition_aggregators:
                    del self.partition_aggregators[partition_id]

                self.partition_aggregators[partition_id] = StateDictAggregator(
                    param_agg_factory=self.param_agg_factor
                )

                for index_client, row in enumerate(runner_clients):
                    runner_client = row[stage_index]
                    key = f"{runner_client}_{stage_index + 1}_{index_client + 1}"
                    model_state = await self.pullOne(current_round, key, runner_client)

                    await self.partition_aggregators[partition_id].add({}, model_state)

                    del model_state
                    force_cleanup()

                aggregated_result = await self.partition_aggregators[
                    partition_id
                ].aggregate()
                aggregated_models[partition_id] = {}
                for k, v in aggregated_result.items():
                    aggregated_models[partition_id][k] = v.clone()

                del aggregated_result
                del self.partition_aggregators[partition_id]
                self.partition_aggregators[partition_id] = None
                force_cleanup()

                logging.critical(
                    f"Aggregated {partition_id} with {len(aggregated_models[partition_id])} parameters"
                )

            for partition_id, state_dict in aggregated_models.items():
                cpu_sd = {}
                for k, v in state_dict.items():
                    cpu_sd[k] = v.cpu().clone()

                meta = {"round": current_round, "key": partition_id}
                await self.communicator.broadcast(meta, cpu_sd)

                del cpu_sd
                force_cleanup()

            return aggregated_models

        elif server_number != client_number:
            aggregated_models = {}
            for stage_index in range(num_partitions):
                partition_id = f"partition_{stage_index + 1}"
                meta = {
                    "round": current_round,
                    "key": partition_id,
                    "from": f"client{server_number}",
                }
                try:
                    (_, model_state) = await self.communicator.pull(meta)

                    aggregated_models[partition_id] = {}
                    for k, v in model_state.items():
                        aggregated_models[partition_id][k] = v.clone()

                    del model_state
                    force_cleanup()

                    logging.critical(
                        f"Received {partition_id} with {len(aggregated_models[partition_id])} parameters"
                    )
                except Exception as e:
                    logging.error(f"Error pulling {partition_id}: {str(e)}")
                    continue

            return aggregated_models

        end_time_pull = time.perf_counter()
        logging.critical(
            f"Finished pulling models at {end_time_pull - start_time_pull}"
        )
        write_log_time("pull", end_time_pull - start_time_pull)
        return {}

    def test_evaluate(self, state_dict,round):
        logging.critical("Starting evaluation - creating lightweight copy")

        eval_dict = {}
        for partition_id, partition_state in state_dict.items():
            eval_dict[partition_id] = {}
            for k, v in partition_state.items():
                if torch.is_tensor(v):
                    eval_dict[partition_id][k] = v.detach().cpu()
                else:
                    eval_dict[partition_id][k] = v

        try:
            model = FullModel(eval_dict)
            accuracy = model.evaluate(self.trainModel.test_loader)
            self.accuracies.append({
                "Round":round,
                "Accuracy":accuracy
            })
        finally:
            for partition_id in list(eval_dict.keys()):
                for k in list(eval_dict[partition_id].keys()):
                    del eval_dict[partition_id][k]
                eval_dict[partition_id].clear()
                del eval_dict[partition_id]
            eval_dict.clear()
            del eval_dict
            del model
            force_cleanup()

        logging.critical("Evaluation completed and cleaned up")

    async def close_connection(self):
        await self.communicator.close_connection()

    async def update_model(self, aggregation_results):
        logging.critical("Starting update_model")

        if not aggregation_results:
            logging.error("No aggregation results to update")
            return

        update_count = 0
        partition_items = list(self.trainModel.partitionsOfClient.items())

        for item, partition in partition_items:
            if item == "finalPartition":
                continue

            parts = item.split("_")
            if len(parts) < 3:
                logging.error(f"Invalid partition key format: {item}")
                continue

            partition_id = f"partition_{parts[1]}"

            if partition_id in aggregation_results:
                state = aggregation_results[partition_id]

                state_copy = {}
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state_copy[k] = v.detach().clone()
                    else:
                        state_copy[k] = v

                await partition.load_partition_state_dict(state_copy)

                for k in list(state_copy.keys()):
                    del state_copy[k]
                state_copy.clear()
                del state_copy

                update_count += 1
                logging.critical(f"Updated {item} with {partition_id} parameters")

                force_cleanup()
            else:
                logging.warning(f"No parameters found for {partition_id}")

        logging.critical(
            f"Updated {update_count}/{len(self.trainModel.partitionsOfClient) - 1} partitions"
        )
        logging.critical("Completed update_model")

    async def train(self, num_epochs: int, round, num_partitions):
        await self.trainModel.train_models(num_epochs, round, num_partitions)

    async def run_client(self, num_rounds, num_epochs: int, matrix):
        time_start_training = time.perf_counter()

        for round in range(num_rounds):
            logging.critical(f"new run client Round :{round}")

            self.define_barrier(f"train_{round}")
            self.define_barrier(f"pull_{round}")

            start_train = time.perf_counter()
            await self.train(num_epochs, round, num_partitions)
            train_time = time.perf_counter() - start_train
            logging.critical(f"Finish Train of Client {train_time:.2f}s")
            await self.get_barrier_point(f"train_{round}", "train_clients")

            start_pull = time.perf_counter()
            model_states = await self.pullModels(round, runner_clients=matrix)
            logging.critical("after pull Models")
            force_cleanup()
            end_pull = time.perf_counter()
            pull_time = end_pull - start_pull

            logging.critical("wait for all client done")
            await self.get_barrier_point(f"pull_{round}", "pull_models")
            pull_all_client = time.perf_counter() - start_pull

            logging.critical(f"Finished Pull of client in {pull_time:.2f}s")
            train_all_client = start_pull - start_train
            logging.critical(f"Finish ALL client train in {train_all_client:.2f}s")
            logging.critical(f"Finish All Client Pull in {pull_all_client:.2f}s")

            self.timeAction.append(
                {
                    "Round": round,
                    "Train_client": train_time,
                    "Train_all_client": train_all_client,
                    "Pull_client": pull_time,
                    "Pull_all_client": pull_all_client,
                }
            )

            logging.critical("Before update_model")
            await self.update_model(model_states)
            logging.critical("After update_model, before evaluate")

            self.test_evaluate(model_states,round)
            logging.critical("After evaluate model")

            for partition_key in list(model_states.keys()):
                if partition_key in model_states:
                    partition_data = model_states[partition_key]
                    if isinstance(partition_data, dict):
                        for tensor_key in list(partition_data.keys()):
                            del partition_data[tensor_key]
                        partition_data.clear()
                    del model_states[partition_key]
            model_states.clear()
            del model_states

            clear_folder_contents()
            force_cleanup()

        time_end_job_client = time.perf_counter()
        total_time = time_end_job_client - time_start_training
        logging.critical(f"Total client time: {total_time:.2f}s")
        self.timeAction.append({"Total_time": total_time})


logging.info("inside container")


def reset_log_files():
    reset_logs_time_file()


async def main():
    logging.critical(f"start time of client :{time.time()}")
    logging.critical("----- Run --------")
    zero_time = time.perf_counter()
    reset_log_files()
    reset_logs_time_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.critical(f"start run on device:{device}")
    logging.info(f"number rounds is :{num_rounds}")

    client = Client()
    client.initialTrainModel()
    await client.initial_communicator()
    await client.trainModel.get_communicator()

    client.define_barrier("event_initial")
    max_retries = 20
    send_done = False

    for attempt in range(max_retries):
        try:
            await client.communicator.push(
                "client1",
                {"barrier_id": "event_initial", "type_barrier": "barrier_ready"},
            )
            send_done = True
            break
        except Exception as e:
            logging.warning(f"Initial sync attempt {attempt + 1} failed: {str(e)}")
            await asyncio.sleep(1)

    if not send_done:
        raise ValueError("Cannot send to client1")

    await client.communicator.barrier_events["event_initial"].wait()
    client.communicator.cleanup_barrier("event_initial")
    initial_time = time.perf_counter() - zero_time
    logging.critical(f"initial time now:{initial_time:.2f}s")

    await client.run_client(num_rounds, num_epochs, matrix=runner_clients)

    end_time_all_clients = time.perf_counter()
    total_time = end_time_all_clients - zero_time
    logging.critical(f"Total execution time: {total_time:.2f}s")
    client.timeAction.append({"Full_time_client": total_time})

    if client_number == 1:
        await asyncio.sleep(10)

    try:
        client.cleanup_partition_aggregators()
        await client.close_connection()
    except Exception as e:
        logging.critical(f"Error closing connection: {str(e)}")

    # write_time_values(client.timeAction, "training_times.csv")
    write_to_csv(client.accuracies,"accuracies.csv")
    write_to_csv(client.timePullPartitions, "pull_partitions_time.csv")
    client.trainModel.write_tasks()


if __name__ == "__main__":
    asyncio.run(main())