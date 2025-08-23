import json
import pandas as pd
import numpy as np


class Greedy:
    def __init__(self, num_clients, num_partitons,
                 forward_path, backward_path, forward_communication_path,
                 backward_communication_path):
        self.num_clients = num_clients
        self.num_partitons = num_partitons
        self.forward_path = forward_path
        self.backward_path = backward_path
        self.forward_communication_path = forward_communication_path
        self.backward_communication_path = backward_communication_path
        self.comm_times_forward = {}
        self.comm_times_backward = {}
        self.num_stages = num_partitons * 2

        # Load execution time data
        self.f_forward = pd.read_csv(self.forward_path)
        self.f_backward = pd.read_csv(self.backward_path)

        # Extract only the rows for active clients (1 to num_clients)
        self.time_exec_forward = self.f_backward.select_dtypes(include='number').iloc[:self.num_clients].values
        self.time_exec_backward = self.f_backward.select_dtypes(include='number').iloc[:self.num_clients].values
        self.time_exec = np.hstack((self.time_exec_forward, self.time_exec_backward)).tolist()

        # Load communication data
        self.comm_forward = pd.read_csv(self.forward_communication_path)
        self.comm_backward = pd.read_csv(self.backward_communication_path)

        # Initialize forward-backward pairs
        self.forward_backward_pairs = {i: (self.num_stages) - i + 1 for i in range(1, self.num_stages + 1)}

    def get_communication_time(self, src_client, dst_client, stage):
        """Get communication time between clients for a specific stage"""
        # Validate client IDs are within active range
        if src_client > self.num_clients or dst_client > self.num_clients:
            print(f"Warning: Client ID out of range. src={src_client}, dst={dst_client}, max={self.num_clients}")
            return 0.0

        if src_client == dst_client:
            return 0.0  # No communication needed within same client

        if stage <= self.num_partitons:  # Forward stages
            part_index = stage - 1  # Convert to 0-based index
            communication_source_to_destination = self.comm_times_forward.get((src_client, dst_client),
                                                                              [0] * self.num_partitons)
            return communication_source_to_destination[part_index]
        else:  # Backward stages
            part_index = stage - (self.num_partitons + 1)  # Convert to 0-based index
            return self.comm_times_backward.get((src_client, dst_client), [0] * self.num_partitons)[part_index]

    def allocate_tasks(self):
        """Main task allocation algorithm"""

        def find_earliest_available_time(client_id, start_after, duration):
            """Find earliest time after last task completes"""
            # If no tasks scheduled, return start_after
            if not all_stages_client[client_id]["stages"]:
                return start_after

            # Find the latest end time of existing tasks
            last_end = 0
            for task in all_stages_client[client_id]["stages"]:
                task_info = task_details[task]
                task_end = task_info["start"] + task_info["duration"]
                if task_end > last_end:
                    last_end = task_end

            # Earliest start time is max of start_after and last task end time
            return max(start_after, last_end)

        # CLIENT TRACKING
        all_stages_client = {
            c: {
                "last_time_inside_client": 0,
                "stages": []
            } for c in range(1, self.num_clients + 1)
        }

        # MODEL PROGRESS TRACKING
        model_progress = {
            i: {
                "last_stage_completed": -1,
                "last_completion_time": 0,
                "last_client": min(i, self.num_clients)
            } for i in range(1, self.num_clients + 1)
        }

        task_details = {}

        # MAIN SCHEDULING LOOP
        for stage in range(1, self.num_stages + 1):
            for model in range(1, self.num_clients + 1):
                task = f"{model}_{stage}"

                # Initialize communication_cost for each task
                communication_cost = 0.0

                if stage in [1, self.num_partitons, self.num_partitons + 1, self.num_stages]:  # BOUNDARY STAGES
                    client_id = min(model, self.num_clients)
                    if stage != 1:  # Only calculate comm cost if not first stage
                        prev_client = model_progress[model]["last_client"]
                        communication_cost = self.get_communication_time(prev_client, client_id, stage)

                    start_after = max(
                        model_progress[model]["last_completion_time"] + communication_cost,
                        all_stages_client[client_id]["last_time_inside_client"]
                    )
                    duration = self.time_exec[client_id - 1][stage - 1]
                    start_time = find_earliest_available_time(client_id, start_after, duration)

                elif stage <= self.num_partitons:  # FORWARD STAGES (greedy selection)
                    best_client = None
                    best_start_time = None
                    best_duration = None
                    best_communication_cost = 0.0
                    best_total_completion_time = float('inf')

                    prev_client = model_progress[model]["last_client"]

                    for candidate_client in range(1, self.num_clients + 1):
                        candidate_communication_cost = self.get_communication_time(prev_client, candidate_client, stage)
                        duration = self.time_exec[candidate_client - 1][stage - 1]

                        initial_ready_time = max(
                            model_progress[model]["last_completion_time"] + candidate_communication_cost,
                            all_stages_client[candidate_client]["last_time_inside_client"]
                        )
                        start_time = find_earliest_available_time(candidate_client, initial_ready_time, duration)
                        completion_time = start_time + duration

                        if completion_time < best_total_completion_time:
                            best_total_completion_time = completion_time
                            best_client = candidate_client
                            best_start_time = start_time
                            best_duration = duration
                            best_communication_cost = candidate_communication_cost

                    client_id = best_client
                    start_time = best_start_time
                    duration = best_duration
                    communication_cost = best_communication_cost

                else:  # BACKWARD STAGES (paired with forward stages)
                    paired_stage = self.forward_backward_pairs[stage]
                    paired_task = f"{model}_{paired_stage}"

                    if paired_task in task_details:
                        client_id = task_details[paired_task]['client']
                    else:
                        print(f"Error: Paired forward stage {paired_task} not found for {task}")
                        exit(1)

                    prev_client = model_progress[model]["last_client"]
                    communication_cost = self.get_communication_time(prev_client, client_id, stage)

                    start_after = max(
                        model_progress[model]["last_completion_time"] + communication_cost,
                        all_stages_client[client_id]["last_time_inside_client"]
                    )
                    duration = self.time_exec[client_id - 1][stage - 1]
                    start_time = find_earliest_available_time(client_id, start_after, duration)

                # UPDATE TRACKING STRUCTURES
                all_stages_client[client_id]["stages"].append(task)
                all_stages_client[client_id]["last_time_inside_client"] = start_time + duration

                model_progress[model]["last_stage_completed"] = stage
                model_progress[model]["last_completion_time"] = start_time + duration
                model_progress[model]["last_client"] = client_id

                task_details[task] = {
                    'client': client_id,
                    'start': start_time,
                    'duration': duration,
                    'communication_cost': communication_cost
                }

        total_time = max(model["last_completion_time"] for model in model_progress.values())
        return task_details, total_time, all_stages_client, model_progress, self.time_exec

    def run_greedy(self):
        """Main execution method"""
        # Parse forward communication times (filter for active clients only)
        for _, row in self.comm_forward.iterrows():
            src, dst = int(row['source_client']), int(row['destination_client'])
            # Only load communication data for active clients
            if src <= self.num_clients and dst <= self.num_clients:
                part_times = [row[f'part{i}'] for i in range(self.num_partitons)]
                self.comm_times_forward[(src, dst)] = part_times

        # Parse backward communication times (filter for active clients only)
        for _, row in self.comm_backward.iterrows():
            src, dst = int(row['source_client']), int(row['destination_client'])
            # Only load communication data for active clients
            if src <= self.num_clients and dst <= self.num_clients:
                part_times = [row[f'part{i}'] for i in range(self.num_partitons)]
                self.comm_times_backward[(src, dst)] = part_times

        print(f"Loaded communication matrix for {self.num_clients}x{self.num_clients} client pairs")

        # Execute the allocation algorithm
        task_details, total_execution_time, all_stages_client, model_progress, time_exec = self.allocate_tasks()

        # ANALYSIS: Check task distribution and communication costs
        client_counts = {}
        total_communication_cost = 0

        for task, details in task_details.items():
            client = details['client']
            client_counts[client] = client_counts.get(client, 0) + 1
            total_communication_cost += details.get('communication_cost', 0)

        print(f"\nTasks per client: {client_counts}")
        print(f"Total communication cost: {total_communication_cost:.3f}")
        print(f"Total execution time: {total_execution_time:.3f}")

        # CREATE ALLOCATION MATRIX
        allocation_matrix = np.zeros((self.num_clients, self.num_partitons), dtype=int)

        for task, details in task_details.items():
            model, stage = map(int, task.split('_'))
            if stage <= self.num_partitons:
                allocation_matrix[model - 1][stage - 1] = details['client']

        stages_output = []
        for i in range(1, self.num_clients + 1):
            stages_output.append(all_stages_client[i]["stages"])

        return stages_output

