import logging
import sys
import torch
import torch.nn as nn
import asyncio
from .Communicate import get_communicator , Communicator , CommunicatorType
from baseModel import resnet_all_layers , cnn_partitions , resnet_50_all_layers
import time
from .dataloader import DataLoader , DataManager
import os
import uuid

global communicator
global client_number , data_manager
# execute information read from .env
client_number = int(os.environ.get('CLIENT_NUMBER', default=1))
num_clients = int(os.environ.get('SIMULATION_NUM_CLIENTS', default=1))
NUMBER_ROUNDS = int(os.environ.get("CAPTURE_EXECUTION_ROUNDS", default=1))
NUMBER_EPOCHS = int(os.environ.get("CAPTURE_EXECUTION_EPOCHS", default=2))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", default=1))
BATCH_COUNT_RECORD = int(os.environ.get("BATCH_COUNT_RECORD",default=1))
CURRENT_MODEL = os.environ.get("CURRENT_MODEL", default=None)
LR = 2e-5

#model initializationn
from .partition import BertPartition , Partition
from .constants import MODEL_RESNET18 , MODEL_BERT , MODEL_CNN , MODEL_RESNET50

from transformers import BertTokenizer, BertModel
import pandas as pd

#path of output result
path_forward_execution = f"log_time/execution_forward.csv"
path_backward_execution = f"log_time/execution_backward.csv"
path_sum_forward_backward = f"log_time/sum_forward_backward.csv"

path_forward_data_size = f"log_time/forward_data_size.csv"
path_backward_data_size = f"log_time/backward_data_size.csv"

path_communication_forward = f"log_time/communication_forward.csv"
path_communication_backward = f"log_time/communication_backward.csv"
import time
global time_communication_forward , time_communication_backward , execution_backward_partitions , execution_forward_partitions ,backward_data_sizes , forward_data_sizes

global batch_count
batch_count = 1
import gc
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.critical(f" device used for train is {DEVICE}")
criterion = nn.CrossEntropyLoss()
global partitions
partitions = {}
RECORD_COMMUNICATION = os.getenv("RECORD_COMMUNICATION") == "True"
logging.critical(f"RECORD COMMUNICATION is {RECORD_COMMUNICATION}")
global test_loader


def initial_partitions():
    logging.critical(f"current model is{CURRENT_MODEL}")
    global partitions
    if CURRENT_MODEL == MODEL_RESNET18:
        logging.critical("Initializing RESNET-18")
        num_layers = len(resnet_all_layers)
        for i in range(num_layers):
            partitions[f'{i + 1}'] = Partition(layers=resnet_all_layers[i])
            i += 1

    elif CURRENT_MODEL == MODEL_CNN :
        logging.critical("Initializing CNN")
        num_layers = len(cnn_partitions)
        for i in range(num_layers):
            partitions[f'{i + 1}'] = Partition(layers=cnn_partitions[i])
            i+=1

    elif CURRENT_MODEL == MODEL_RESNET50:
        logging.critical("Initializing Resnet50")
        num_layers = len(resnet_50_all_layers)
        for i in range(num_layers):
            partitions[f'{i + 1}'] = Partition(layers=resnet_50_all_layers[i])
            i += 1
    else:
        logging.critical(f"Initializing partitions FOR BERT...")
        bert = BertModel.from_pretrained(
            "/root/.cache/huggingface/transformers/bert-base-uncased",
            local_files_only=True,
            num_labels=2,
        )
        emb = bert.embeddings
        enc_layers = list(bert.encoder.layer)
        selected_layers = enc_layers[:1] + enc_layers[-3:]
        pooler = bert.pooler
        classifier = nn.Linear(bert.config.hidden_size, 2)

        all_layers = [emb] + enc_layers + [nn.Sequential(pooler, classifier)]

        for idx, layer in enumerate(all_layers):
            p = BertPartition(layer,
                              is_first=(idx == 0),
                              is_last=(idx == len(all_layers) - 1))
            p.device = DEVICE
            p.layers = p.layers.to(p.device)

            p.optimizer = torch.optim.Adam(p.layers.parameters(), lr=LR)
            partitions[f"index_{idx}"] = p
        logging.critical(f"Initial partitions: {partitions.keys()}")

def get_tensor_size_bytes(tensor):
    if torch.is_tensor(tensor):
        return tensor.numel() * tensor.element_size()
    elif isinstance(tensor, (list, tuple)):
        return sum(get_tensor_size_bytes(t) for t in tensor)
    elif hasattr(tensor, '__sizeof__'):
        return sys.getsizeof(tensor)
    else:
        return len(pickle.dumps(tensor))

def initial_record_execution_time():
    global execution_forward_partitions , execution_backward_partitions , forward_data_sizes , backward_data_sizes , backward_data_sizes

    execution_forward_partitions = [0 for i in range(len(partitions))]
    execution_backward_partitions = [0 for i in range(len(partitions))]
    forward_data_sizes = [0 for i in range(len(partitions))]
    backward_data_sizes = [0 for i in range(len(partitions))]


    global time_communication_forward , time_communication_backward
    time_communication_forward = [[0 for i in range(len(partitions))] for j in range(num_clients)]
    time_communication_backward = [[0 for i in range(len(partitions))] for j in range(num_clients)]

    rows_forward = len(time_communication_forward)
    cols_forward = len(time_communication_forward[0]) if rows_forward > 0 else 0
    rows_backward = len(time_communication_backward)
    cols_backward = len(time_communication_backward[0]) if rows_backward > 0 else 0

    logging.critical(f"Forward communication shape: ({rows_forward}, {cols_forward})")
    logging.critical(f"Backward communication shape: ({rows_backward}, {cols_backward})")


async def custom_handler(meta, data):
    item = data.get("output", data)

    # now refer to `item`, not `data`
    if item and item.get("id"):
        received_time = time.time()

        logging.critical(f"forward - partition:{item['partition']} ,source:{item['source'] } ,destination"
                         f":{item['destination']} , time:{received_time - item['send_time']}")


async def initial_communicator():
    global communicator
    logging.critical("initialing communicator")
    communicator = await get_communicator(CommunicatorType.GRPC,custom_handler=custom_handler)
    logging.critical("wait to sure communication created")


async def request_push(destination, out):
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()

    data_value = {
        "output": out,
        "stage": "1",
        "client_id": f"{client_number}"
    }
    await communicator.push(destination=destination, data=data_value)

    # Explicitly remove reference (though Python should handle this)
    del data_value


async def push_to_all_backward(data, partition_idx):
    random_id = uuid.uuid4()

    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data

    # logging.critical(f"backward push for all {partition_idx}")
    for i in range(num_clients):
        destination_client = i + 1
        time_before_push = time.perf_counter()
        sending_time = time.time()
        item = {
            "partition": partition_idx,
            "type":"forward",
            "source":client_number ,
            "destination":destination_client,
            "id":random_id ,
            "data":data_np,
            "send_time":sending_time
        }
        await request_push(destination=f"client{destination_client}", out=item)
        push_time = time.perf_counter() - time_before_push
        time_communication_backward[i][partition_idx] += push_time
        # logging.critical(f"pushed to {destination_client}")
        # logging.critical(f"forward - partition:{partition_idx} ,source:{client_number} ,destination"
        #                  f":{destination_client} , time:{push_time}")


async def push_to_all_forward(data, partition_idx):
    random_id = uuid.uuid4()
    sending_time = time.time()
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data

    for i in range(num_clients):
        destination_client = i + 1
        item = {
            "partition": partition_idx,
            "type": "forward",
            "source": client_number,
            "destination": destination_client,
            "id": random_id,
            "data": data_np,
            "send_time": sending_time
        }
        time_before_push = time.perf_counter()
        await request_push(destination=f"client{destination_client}", out=item)
        push_time = time.perf_counter() - time_before_push
        # logging.critical(f"forward - partition:{partition_idx} ,source:{client_number} ,destination"
        #                  f":{destination_client} , time:{push_time}")
        time_communication_forward[i][partition_idx] += push_time
        # logging.critical(f"pushed to {destination_client}")

def init_dataManager():
    global client_number , test_loader

    if CURRENT_MODEL == MODEL_RESNET18 or CURRENT_MODEL== MODEL_RESNET50:
        data_loader = DataLoader(batch_size=BATCH_SIZE, shuffle=True, index_client=client_number, d_name='CIFAR10')
        data_loader.set_mode("train")
        manager = DataManager(data_loader)

        test_loader = data_loader.test_loader
        return manager
    elif CURRENT_MODEL == MODEL_CNN:
        data_loader = DataLoader(batch_size=BATCH_SIZE,shuffle=True, index_client=client_number, d_name='MNIST')
        data_loader.set_mode("train")
        manager = DataManager(data_loader)
        test_loader = data_loader.test_loader
        return manager
    else:
        data_loader = DataLoader(batch_size=BATCH_SIZE, shuffle=True, index_client=client_number, d_name='IMDB')
        data_loader.set_mode("train")
        manager = DataManager(data_loader)
        return manager


async def train_bert(c_n):
    global client_number
    if c_n != client_number:
        logging.critical(f"no train and communication")
        return
    logging.critical(f"client{c_n} is sender ")
    logging.critical(f"training client {client_number}")
    logging.critical(f"start train for batch count{BATCH_COUNT_RECORD}")
    global data_manager
    logging.critical("Starting training...")


    for r in range(NUMBER_ROUNDS):
        logging.critical(f"Round {r + 1}/{NUMBER_ROUNDS}")
        global batch_count
        batch_count = 0
        data_manager.epoch = 0

        while data_manager.epoch < NUMBER_EPOCHS:
        # for i in range(BATCH_COUNT_RECORD):
            epoch_loss = 0.0
            correct_predictions = 0
            batch_count = 0
            global_batch_counter = 0

            logging.critical(f"Epoch {data_manager.epoch + 1}/{NUMBER_EPOCHS}")

            # Reset dataset for new epoch
            data_manager.reset_dataset()
            for i in range(BATCH_COUNT_RECORD):
            # while not data_manager.epoch_done:
                #empty collector

                # Get batch data
                try:
                    features, batch_labels = data_manager.next_batch()
                    input_ids, attention_mask = features
                    batch_labels = batch_labels.to(DEVICE)
                    batch_id = global_batch_counter
                    global_batch_counter += 1
                    batch_count += 1
                    logging.critical(f"batch count: {batch_count}")

                    # ——— Forward pass ———
                    time_start = time.perf_counter()
                    out = partitions["index_0"].forward(input_ids, batch_id=batch_id)
                    execution_time_forward = time.perf_counter() - time_start
                    execution_forward_partitions[0] += execution_time_forward
                    forward_data_sizes[0] += get_tensor_size_bytes(out)

                    if RECORD_COMMUNICATION:
                        await push_to_all_forward(data=out, partition_idx=0)
                    # Middle partitions
                    for i in range(1, len(partitions) - 1):
                        time_start = time.perf_counter()
                        out = partitions[f"index_{i}"].forward(
                            out,
                            batch_id=batch_id,
                            attention_mask=attention_mask
                        )
                        time_execution_forward = time.perf_counter() - time_start
                        execution_forward_partitions[i] +=time_execution_forward
                        # logging.critical(f"after forward partition {i}")
                        if RECORD_COMMUNICATION:
                            await push_to_all_forward(data=out, partition_idx=i)
                        forward_data_sizes[i] += get_tensor_size_bytes(out)


                    # Last partition
                    time_start = time.perf_counter()
                    out = partitions[f"index_{len(partitions) - 1}"].forward(out, batch_id=batch_id)
                    execution_forward = time.perf_counter() - time_start
                    execution_forward_partitions[len(partitions)-1] +=execution_forward
                    forward_data_sizes[len(partitions)-1] += get_tensor_size_bytes(out)
                    logging.critical("after forward last")
                    logits = out


                    # ——— Compute loss and accuracy ———
                    loss = criterion(logits, batch_labels)
                    epoch_loss += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct = (preds.cpu() == batch_labels.cpu()).sum().item()
                    correct_predictions += correct

                    # ——— Backward pass ———
                    grad_logits = torch.autograd.grad(loss, logits)[0].detach().cpu()
                    grad = grad_logits
                    p = 0
                    for i in reversed(range(len(partitions))):
                        start_time_backward = time.perf_counter()
                        grad = partitions[f"index_{i}"].backward(grad, batch_id=batch_id)
                        grad_detached = grad.detach().cpu()
                        backward_time_execution = time.perf_counter() - start_time_backward
                        execution_backward_partitions[p] += backward_time_execution
                        if RECORD_COMMUNICATION:
                            await push_to_all_backward(data = grad_detached, partition_idx = p)
                        del grad
                        grad = grad_detached
                        backward_data_sizes[p] += get_tensor_size_bytes(grad)
                        p +=1

                    # Log progress
                    if (batch_count % 2) == 0:
                        batch_acc = correct / BATCH_SIZE * 100
                        logging.critical(
                            f"Batch {batch_count} | "
                            f"Loss: {loss.item():.4f} | "
                            f"Acc: {batch_acc:.2f}%"
                        )

                except StopIteration:
                    # End of epoch
                    data_manager.epoch_done = True

                del features, batch_labels, input_ids, attention_mask  # Explicit cleanup
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # End of epoch statistics
            data_manager.epoch += 1
            avg_epoch_loss = epoch_loss / batch_count
            epoch_acc = correct_predictions / (batch_count * BATCH_SIZE) * 100
            logging.critical(
                f"Epoch {data_manager.epoch} Summary | "
                f"Avg Loss: {avg_epoch_loss:.4f} | "
                f"Accuracy: {epoch_acc:.2f}%"
            )
            gc.collect()

        logging.critical(f"batch count: {batch_count}")


def output_collector():
    global batch_count

    df_forward_execution = pd.DataFrame({
        f'stage{i}': [value / batch_count ]
        for i, value in enumerate(execution_forward_partitions)
    })
    df_backward_execution = pd.DataFrame({
        f'stage{i}': [value / batch_count ]
        for i, value in enumerate(execution_backward_partitions)
    })

    avg_forward_size = [size / batch_count for size in forward_data_sizes]
    avg_backward_size = [size / batch_count for size in backward_data_sizes]
    for i in range(num_clients):
        for j in range(len(partitions)):
            time_communication_forward[i][j] /= batch_count
            time_communication_backward[i][j] /= batch_count
    # Create communication DataFrames with source/destination columns and partition columns
    forward_rows = []
    backward_rows = []

    # Create rows for each destination client
    for dest_client in range(num_clients):
        # Forward communication row
        forward_row = {
            'source_client': client_number,
            'destination_client': dest_client + 1
        }
        # Add partition columns
        for partition_idx in range(len(partitions)):
            forward_row[f'part{partition_idx}'] = time_communication_forward[dest_client][partition_idx]
        forward_rows.append(forward_row)

        # Backward communication row
        backward_row = {
            'source_client': client_number,
            'destination_client': dest_client + 1
        }
        # Add partition columns
        for partition_idx in range(len(partitions)):
            backward_row[f'part{partition_idx}'] = time_communication_backward[dest_client][partition_idx]
        backward_rows.append(backward_row)

    # Create DataFrames and save for communication
    df_communication_forward = pd.DataFrame(forward_rows)
    df_communication_backward = pd.DataFrame(backward_rows)

    df_communication_forward.to_csv(path_communication_forward, index=False)
    df_communication_backward.to_csv(path_communication_backward, index=False)

    logging.critical(f"batch count :{batch_count} and number round :{NUMBER_ROUNDS}")
    df_sum_forward_backward = df_forward_execution + df_backward_execution
    df_forward_execution.to_csv(path_forward_execution, index=False)
    df_backward_execution.to_csv(path_backward_execution, index=False)
    df_sum_forward_backward.to_csv(path_sum_forward_backward, index=False)

    # Save data size information
    df_forward_data_size = pd.DataFrame({
        f"stage{i}": [size]
        for i, size in enumerate(avg_forward_size)
    })
    df_backward_data_size = pd.DataFrame({
        f'stage{i}_mb': [size]
        for i, size in enumerate(avg_backward_size)
    })

    df_forward_data_size.to_csv(path_forward_data_size, index=False)
    df_backward_data_size.to_csv(path_backward_data_size, index=False)

    logging.critical(f"\nall batch time recorded is {batch_count}")
    logging.critical(f"\ntime execution forward is: {execution_forward_partitions}")
    logging.critical(f"\ntime execution backward is: {execution_backward_partitions}")
    logging.critical(f"\ntime communication forward is: {time_communication_forward}")
    logging.critical(f"\ntime communication backward is: {time_communication_backward}")


def compute_loss_and_grad(predictions, labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss()
    predictions = predictions.to(device)
    labels = labels.to(device)
    predictions = predictions.detach().requires_grad_(True)
    loss = loss_fn(predictions, labels)
    # Compute gradient of loss w.r.t. predictions
    loss.backward()
    grad_output = predictions.grad
    return loss.item(), grad_output

async def train_Resnet(c_n):
    global client_number
    if c_n != client_number:
        logging.critical(f"no train and communication")
        return
    logging.critical(f"client{c_n} is sender ")
    logging.critical(f"training client {client_number}")
    logging.critical(f"start train for batch count{BATCH_COUNT_RECORD}")
    global data_manager
    logging.critical("Starting training...")

    global batch_count
    for r in range(NUMBER_ROUNDS):
        batch_id = 0
        while data_manager.epoch < NUMBER_EPOCHS:
            features, label = data_manager.next_batch()
            input = features
            p = 0

            # Forward pass
            for partition in partitions.values():
                time_start = time.perf_counter()
                out = await asyncio.to_thread(partition.forward, input, batch_id)
                end_time = time.perf_counter()

                data_size_bytes = get_tensor_size_bytes(out)
                forward_data_sizes[p] += data_size_bytes

                execution_forward_partitions[p] += (end_time - time_start)
                input = out
                if RECORD_COMMUNICATION:
                    await push_to_all_forward(out, p)
                p += 1

            # calculate loss function
            loss, grad_final = await asyncio.to_thread(compute_loss_and_grad, input, label)

            # Backward pass
            p = 0
            gradient = grad_final
            for partition_index in range(len(partitions), 0, -1):
                grad_size_bytes = get_tensor_size_bytes(gradient)
                backward_data_sizes[p] += grad_size_bytes
                start_time = time.perf_counter()
                new_gradient = await asyncio.to_thread(partitions[f'{partition_index}'].backward, gradient, batch_id)
                end_time = time.perf_counter()
                execution_backward_partitions[p] += (end_time - start_time)
                gradient = new_gradient
                if RECORD_COMMUNICATION:
                    await push_to_all_backward(gradient, p)
                p += 1

            batch_id += 1
            batch_count += 1
        data_manager.reset_dataset()


async def main():
    logging.critical("-----------RUN TIME RECORDER----------")
    time_start = time.time()

    # Initialize components
    event = asyncio.Event()
    await initial_communicator()
    communicator.create_barrier(barrier_id="event_initial")
    max_retries = 20
    send_done = False

    for attempt in range(max_retries):
        try:
            await communicator.push("client1", {
                "barrier_id": "event_initial",
                "type_barrier": "barrier_ready"
            })
            send_done = True
            break
        except Exception:
            await asyncio.sleep(1)

    if not send_done:
        raise ValueError("Cannot send to client1")

    await communicator.barrier_events["event_initial"].wait()
    communicator.cleanup_barrier("event_initial")

    logging.critical("after initialized all communicators")

    initial_partitions()
    initial_record_execution_time()
    global data_manager
    data_manager = init_dataManager()



    # Training
    # for first  execute on lower server
    for i in range(1,num_clients+1):
        communicator.create_barrier("event_train")
        logging.critical(f"MODEL selected{CURRENT_MODEL}")
        if CURRENT_MODEL == MODEL_RESNET18 or CURRENT_MODEL == MODEL_CNN or CURRENT_MODEL== MODEL_RESNET50:
            await train_Resnet(i)
        else:
            await train_bert(i)

        await communicator.wait_at_barrier(barrier_id="event_train")
        communicator.cleanup_barrier(barrier_id="event_train")
        await asyncio.sleep(2)
        logging.critical(f"new client select for send is {i+1}")








    output_collector()
    logging.critical(f"done")

    logging.critical("after train**")
    # Cleanup
    await communicator.close_connection()
    logging.critical(f"Total execution time: {time.time() - time_start:.2f}s")



if __name__ == "__main__":
    asyncio.run(main())












