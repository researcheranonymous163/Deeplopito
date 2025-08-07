# **Quick Setup**

Follow these steps to launch a 5-client split training environment quickly.

### A. Using Docker Compose
1- Clone the repository
* `git clone https://github.com/researcheranonymous163/Deeplopito`
* `chmod +x docker-entrypoint.sh`
* `docker-compose up --build`

### B. Using Docker Swarm
Build the Docker image
* `chmod +x docker-entrypoint.sh`
* `docker build -t split-training:latest .`
* `sudo docker stack deploy -c docker-compose-swarm_5_client.yml my_stack`


for change configuration of program you can change .env file.
# Split Training Execution Configuration

**This section describes how to configure and execute a split training program across multiple clients. It covers partition assignment, stage pairing, and client ownership to ensure proper forward and backward pass execution.**

#### 1. Partition Configuration

* Each model is divided into N partitions. For split training, you must create an execution_config list of length M, where M is the number of clients.
* Each sublist in SIMULATION_STAGES_CLIENTS and set in .env file corresponds to the tasks assigned to a specific client.
* Tasks are labeled as "{client_id}_{stage_id}":
  * client_id: The owner of the partition (which client runs it).
  * stage_id: The position of this partition in the overall pipeline.

****Example: CNN model split into 4 client partitions (8 total forward/backward stages)****

    SIMULATION_STAGES_CLIENTS = [
    ['1_1', '1_2', '4_3', '1_4', '1_5', '4_6', '1_7', '1_8'],  # Client 1 tasks
    ['2_1', '2_2', '1_3', '2_4', '2_5', '1_6', '2_7', '2_8'],  # Client 2 tasks
    ['3_1', '5_2', '5_3', '3_4', '3_5', '5_6', '5_7', '3_8'],  # Client 3 tasks
    ['4_1', '4_2', '3_3', '4_4', '4_5', '3_6', '4_7', '4_8'],  # Client 4 tasks
    ['5_1', '3_2', '2_3', '5_4', '5_5', '2_6', '3_7', '5_8'],  # Client 5 tasks]

#### 2. Forward/Backward Stage Pairing

Each partition has two stages: forward and backward. To ensure they run on the same device, define a forward_backward_pair_stages mapping:
    forward_backward_pair_stages = {
    1: 8,  # forward stage 1 ↔ backward stage 8
    2: 7,  # forward stage 2 ↔ backward stage 7
    3: 6,  # forward stage 3 ↔ backward stage 6
    4: 5,  # forward stage 4 ↔ backward stage 5
    5: 4,  # backward stage 5 ↔ forward stage 4
    6: 3,
    7: 2,
    8: 1 } 

#### 3. Client Ownership Rules
1. Owner Consistency: All stages of a given partition (both forward and backward) must be owned by the same client. For example, partition 3 produces stages 3 and 6—both must appear in the same client’s task list. 
2. Endpoint Co-location & Data Privacy: The first and last partitions within each client’s sub-pipeline (i.e., its entry and exit stages) must run on the same device. This guarantees boundary consistency when passing tensors. 
   * Data Privacy: Keeping the last partition on the owner device allows local accuracy computation without exposing raw outputs to other clients. 
   * Gradient Passing: Only intermediate gradients are exchanged between clients, ensuring no raw activations or labels leave the owner device. 
`   For example, if P = 4 partitions:
   First partition → stage ID 1 (forward) and stage ID 8 (backward pair)
   Last partition → stage ID 4 (forward) and stage ID 5 (backward pair)
   In execution_config, Client 1’s endpoint tasks appear as:
   Forward: '1_1' and '1_4' 
   Backward: '1_8' and '1_5'
   Client 1 endpoint co-location:
   Forward:  '1_1' (partition 1) and '1_4' (partition 4)
   Backward: '1_8' (pair of 1) and '1_5' (pair of 4)`


# Example Configurations

several sample execution_config dictionaries located in a Python file "example_configurations" and you can use to launch split training:
