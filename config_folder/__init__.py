import logging
import json
import os



dataset_name = os.environ.get("CURRENT_DATASET")

num_rounds = int(os.environ.get("TRAINING_NUM_ROUNDS",default=10))
num_epochs = int(os.environ.get("TRAINING_NUM_EPOCHS",default=1))
n_spit_dataset = int(os.environ.get("DATASET_N_SPLITS",default=10))
sleep_time = int(os.environ.get("TRAINING_SLEEP_TIME",default=5))
num_clients = int(os.environ.get("SIMULATION_NUM_CLIENTS",default=10))


all_priorities = json.loads(os.environ.get("SIMULATION_PRIORITIES", "[]"))
neighbour_clients = json.loads(os.environ.get("SIMULATION_NEIGHBOUR_CLIENTS", "[]"))

priorities = all_priorities[:num_clients]

stages_client= json.loads(os.environ.get("SIMULATION_STAGES_CLIENTS", "[]"))

# Expose these variables for import
__all__ = ["dataset_name", "num_rounds", "num_epochs" , "n_spit_dataset" , "num_clients","priorities", "neighbour_clients" , "sleep_time" , "stages_client"]
