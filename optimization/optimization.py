import csv
import logging
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from mistune.plugins.task_lists import task_lists_hook


class Optimization:
    def __init__(self , num_clients , num_partitons):
        self.num_clients = num_clients
        self.num_partitions = num_partitons
        self.number_stages = num_partitons *2
        self.forward_times = None
        self.backward_times = None
        self.communication_times_forward = None
        self.communication_times_backward = None
        self.solution_stages_client=None
        self.schedule = []
        self.task_mapping = {i: self.number_stages - i + 1 for i in range(1, self.number_stages + 1)}





    def input_data(self,forward_path , backward_path ,communication_forward_path, communication_backward_path):

        self.forward_times = self.read_execution_times(forward_path)
        self.backward_times = self.read_execution_times(backward_path)
        self.communication_times_forward = self.read_communication_times(communication_forward_path)
        self.communication_times_backward = self.read_communication_times(communication_backward_path)

    def set_solution_stages(self,solution_stages_client):
        self.solution_stages_client = solution_stages_client
    @staticmethod
    def parse_task(task):
        owner , partition = task.split('_')
        return int(owner), int(partition)

    @staticmethod
    def read_execution_times(filename):
        times = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                times.append([float(x) for x in row])
        return times

    def read_communication_times(self,filename):
        comm_times = {}
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = int(row['source_client'])
                dst = int(row['destination_client'])
                times = []
                for i in range(self.num_partitions):
                    times.append(float(row[f'part{i}']))
                comm_times[(src, dst)] = times
        return comm_times

    def get_exec_time(self, partition, client):
        client_idx = client - 1
        if partition <= self.num_partitions:
            return self.forward_times[client_idx][partition - 1]
        elif partition <= self.number_stages:
            return self.backward_times[client_idx][partition - (self.num_partitions+1)]
        else:
            raise ValueError(f"Invalid partition {partition}")

    def get_communication_time(self, partition, src_client, dst_client):
        print("get communication time")
        print(partition, src_client, dst_client,type(src_client), type(dst_client))

        if src_client == dst_client:
            return 0.0

        if partition <= self.num_partitions:
            comm_idx = partition - 1
            return self.communication_times_forward[(src_client, dst_client)][comm_idx]
        elif partition <= self.number_stages:
            comm_idx = partition - (self.num_partitions+1)
            return self.communication_times_backward[(src_client, dst_client)][comm_idx]
        else:
            raise ValueError(f"Invalid partition {partition}")
    def validate_solution(self):
        pass


    def simulate_2(self):
        print("simulate Time solution")
        print("DEBUG:", self.get_exec_time(1, 1))


        # Build full_tasks dictionary and client queues
        full_tasks = {}
        client_queues = {}
        for client_id, sublist in enumerate(self.solution_stages_client, start=1):
            client_queues[client_id] = sublist[:]  # Copy of task list for this client
            for task_str in sublist:
                owner, partition = self.parse_task(task_str)
                full_tasks[task_str] = {"runner_client": client_id, "owner": owner, "partition": partition}

        print(f'full tasks is {full_tasks}')
        print(f'client queue is {client_queues}')
        # Initialize tracking structures
        end_times_clients = {cid: 0.0 for cid in range(1, self.num_clients + 1)}
        task_details = {}  # Tracks completed tasks
        schedule_temp = []  # Final schedule
        next_task_index = {cid: 0 for cid in range(1, self.num_clients + 1)}  # Next task index per client
        total_tasks = sum(len(tasks) for tasks in self.solution_stages_client)
        print(f"total tasks:{total_tasks}")
        # Main simulation loop
        scheduled_count = 0
        while scheduled_count < total_tasks:
            ready_tasks = []

            # Find next ready task for each client
            for client_id in range(1, self.num_clients + 1):
                idx = next_task_index[client_id]
                # Skip if client has no more tasks
                if idx >= len(client_queues[client_id]):
                    continue
                task_str = client_queues[client_id][idx]
                owner, partition = self.parse_task(task_str)

                # Check if task is ready (partition 1 or previous partition completed)
                if partition == 1:
                    ready_tasks.append(full_tasks[task_str])
                    next_task_index[client_id] += 1
                else:
                    prev_task = f"{owner}_{partition - 1}"
                    if prev_task in task_details:
                        ready_tasks.append(full_tasks[task_str])
                        next_task_index[client_id] += 1

            # Process all ready tasks in this batch
            for task in ready_tasks:
                runner_client = task['runner_client']
                owner = task['owner']
                partition = task['partition']

                # Calculate dependency time
                dep_end = 0.0
                comm_time = 0.0
                if partition > 1:
                    prev_task = f"{owner}_{partition - 1}"
                    dep_end = task_details[prev_task]["end_time"]
                    prev_client = task_details[prev_task]["client"]
                    if prev_client != runner_client:
                        comm_time = self.get_communication_time(partition - 1, prev_client, runner_client)

                # Calculate start time (max of dependency ready time or client availability)
                start_time = max(dep_end + comm_time, end_times_clients[runner_client])
                duration = self.get_exec_time(partition, runner_client)
                end_time = start_time + duration

                # Update client availability and record task
                end_times_clients[runner_client] = end_time
                task_details[f"{owner}_{partition}"] = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "partition": partition,
                    "client": runner_client
                }
                schedule_temp.append({
                    "client": runner_client,
                    "start_time": start_time,
                    "end_time": end_time,
                    "partition": partition,
                    "owner": owner,
                    "duration":end_time - start_time,
                    "communication_time":comm_time
                })
                scheduled_count += 1

                print(f"task {owner}_{partition}: start={start_time:.2f}, "
                      f"duration={duration:.2f}, end={end_time:.2f}, "
                      f"client={runner_client}, comm_time={comm_time:.2f}")

        self.schedule = schedule_temp


    #every time do last partition -1 => add last partition of that client to ready
    def simulate_fix_last_layer(self):
        print("simulate Time solution")
        # Build full_tasks dictionary and client queues
        full_tasks = {}
        client_queues = {}
        for client_id, sublist in enumerate(self.solution_stages_client, start=1):
            client_queues[client_id] = sublist[:]  # Copy of task list for this client
            for task_str in sublist:
                owner, partition = self.parse_task(task_str)
                full_tasks[task_str] = {"runner_client": client_id, "owner": owner, "partition": partition}

        print(f'full tasks is {full_tasks}')
        print(f'client queue is {client_queues}')
        # Initialize tracking structures
        end_times_clients = {cid: 0.0 for cid in range(1, self.num_clients + 1)}
        task_details = {}  # Tracks completed tasks
        schedule_temp = []  # Final schedule
        next_task_index = {cid: 0 for cid in range(1, self.num_clients + 1)}  # Next task index per client
        total_tasks = sum(len(tasks) for tasks in self.solution_stages_client)
        print(f"total tasks:{total_tasks}")
        # Main simulation loop
        scheduled_count = 0
        while scheduled_count < total_tasks:
            ready_tasks = []

            # Find next ready task for each client
            for client_id in range(1, self.num_clients + 1):
                idx = next_task_index[client_id]
                # Skip if client has no more tasks
                if idx >= len(client_queues[client_id]):
                    continue

                task_str = client_queues[client_id][idx]
                owner, partition = self.parse_task(task_str)

                # Check if task is ready (partition 1 or previous partition completed)\
                if partition ==self.num_partitions or partition ==self.number_stages:
                    next_task_index[client_id] +=1
                    continue
                if partition == 1:
                    ready_tasks.append(full_tasks[task_str])
                    next_task_index[client_id] += 1
                else:

                    prev_task = f"{owner}_{partition - 1}"
                    if prev_task in task_details:
                        ready_tasks.append(full_tasks[task_str])
                        next_task_index[client_id] += 1

                        if (partition == self.num_partitions-1) or (partition==self.number_stages-1):
                            task_str_last_partition = f"{owner}_{partition +1}"
                            ready_tasks.append(full_tasks[task_str_last_partition])
                            print(f"list read is {ready_tasks} and task latest part :{task_str_last_partition}")


            # Process all ready tasks in this batch
            for task in ready_tasks:
                runner_client = task['runner_client']
                owner = task['owner']
                partition = task['partition']

                # Calculate dependency time
                dep_end = 0.0
                comm_time = 0.0
                if partition > 1:
                    prev_task = f"{owner}_{partition - 1}"
                    dep_end = task_details[prev_task]["end_time"]
                    prev_client = task_details[prev_task]["client"]
                    if prev_client != runner_client:
                        comm_time = self.get_communication_time(partition - 1, prev_client, runner_client)

                # Calculate start time (max of dependency ready time or client availability)
                start_time = max(dep_end + comm_time, end_times_clients[runner_client])
                duration = self.get_exec_time(partition, runner_client)
                end_time = start_time + duration

                # Update client availability and record task
                end_times_clients[runner_client] = end_time
                task_details[f"{owner}_{partition}"] = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "partition": partition,
                    "client": runner_client
                }
                schedule_temp.append({
                    "client": runner_client,
                    "start_time": start_time,
                    "end_time": end_time,
                    "partition": partition,
                    "owner": owner,
                    "duration":end_time - start_time,
                    "communication_time":comm_time
                })
                scheduled_count += 1

                print(f"task {owner}_{partition}: start={start_time:.2f}, "
                      f"duration={duration:.2f}, end={end_time:.2f}, "
                      f"client={runner_client}, comm_time={comm_time:.2f}")

        self.schedule = schedule_temp

    def visualization_time_solution(self):

        df = pd.DataFrame(self.schedule)
        base_time = datetime(2024, 1, 1)
        df['start_datetime'] = df['start_time'].apply(lambda x: base_time + timedelta(seconds=x))
        df['end_datetime'] = df['end_time'].apply(lambda x: base_time + timedelta(seconds=x))

        fig = px.timeline(df, x_start="start_datetime", x_end="end_datetime",
                          y="client", color="partition", title="Task Schedule Timeline",hover_data=[
                "start_time","end_time","client","partition","owner"
            ])
        fig.show()
        df.to_csv("timeSimulation.csv")


    @staticmethod
    def convert_to_index1_solution(config):
        new_config = []
        num_stages = 0
        for group in config:
            new_group = []
            for item in group:
                if '_' in item:
                    client, partition = item.split('_')
                    new_item = f"{int(client) + 1}_{int(partition) + 1}"
                    new_group.append(new_item)
                    num_stages += 1

                else:
                    # Handle cases where format might be different
                    new_group.append(item)
            new_config.append(new_group)

        print("number of stages: ", num_stages)
        return new_config

    def average_client_finish(self,df):
        num_clients = self.num_clients
        num_stages = self.number_stages
        total_end = 0.0

        latest_end_time = 0.0
        for client in range(1, num_clients + 1):
            # Filter rows for this client
            client_rows = df[df['client'] == client]
            best_end = 0.0
            best_part = 0

            for part in range(1, num_stages + 1):
                # Check if there's a row for this partition
                part_row = client_rows[client_rows['partition'] == part]
                if not part_row.empty and part > best_part:
                    best_part = part
                    best_end = part_row['end_time'].iloc[0]
                    if best_end > latest_end_time:
                        latest_end_time = best_end

            # print(f"best end for client {client} and partitition:{best_part} is :{best_end}")
            total_end += best_end
        average = total_end / self.num_clients
        return total_end , average , latest_end_time , (latest_end_time - average)


    def get_complete_task(self ,task:str):

        client_num, task_num = task.split('_')
        task_num = int(task_num)

        if task_num in self.task_mapping:
            complete_task_num = self.task_mapping[task_num]
            return f"{client_num}_{complete_task_num}"
        return None

    def add_backward_stages(self , matrix):

        for sublist in matrix:
            complete_tasks = []

            for task_id in sublist:
                complete_task = self.get_complete_task(task=task_id)
                if complete_task:
                    complete_tasks.append(complete_task)

            complete_tasks.reverse()
            sublist.extend(complete_tasks)
        import json
        json_matrix = json.dumps(matrix, ensure_ascii=False, separators=(',', ':'))
        return json_matrix



