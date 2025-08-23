from dataloader import ExecutionDataLoader
from dataloader import ExecutionDataLoader
import numpy as np
import random
from typing import Union
from optimization import Optimization

def generate_two_random_lists(n, m):
    values = random.sample(range(1, m + 1), 2 * n)

    # Split into two lists
    list1 = values[:n]
    list2 = values[n:]

    return list1, list2

number_partitions = 14
def Read() -> None:
    loader = ExecutionDataLoader(
        forward_exec_path="../../generated_datasets/final_input_values/bert/bert_homo_execution_forward.csv",
        backward_exec_path="../../generated_datasets/final_input_values/bert/bert_homo_execution_backward.csv",
        forward_comm_path="../../generated_datasets/final_input_values/bert/bert_homo_communication_forward.csv",
        backward_comm_path="../../generated_datasets/final_input_values/bert/bert_homo_communication_backward.csv",
    )

    (
        n_stages,
        n_machines,
        forward_exec,
        backward_exec,
        forward_comm,
        backward_comm,
    ) = loader.load()

    # Quick sanity checks / demo
    # print(f"Stages:   {n_stages}")
    # print(f"Machines: {n_machines}")
    return n_stages, n_machines, forward_exec, backward_exec, forward_comm, backward_comm


# def modifyvalues(array,indices, rate):
#     for idx in indices:
#         array[idx-1]*=rate

def remaining_flops(flops_base, machine_front, n_machines, n_stages):
    remaining_Flops = []
    for i in range(n_machines):
        sum = 0
        for j in range(machine_front[i], n_stages):
            sum += flops_base[j]
        remaining_Flops.append(sum)
    return remaining_Flops


def forward_backward_visualize(forward_exec, backward_exec):
    ratio = forward_exec / backward_exec

    import matplotlib.pyplot as plt
    import numpy as np

    # Example 2D matrix
    matrix = ratio

    import numpy as np
    import matplotlib.pyplot as plt

    # Example matrix with values in [0, 1]

    # Define bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1

    num_cols = matrix.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=num_cols, figsize=(4 * num_cols, 4))

    # If there's only one column, axs is not a list—make it a list
    if num_cols == 1:
        axs = [axs]

    for i in range(num_cols):
        col_data = matrix[:, i]
        hist, _ = np.histogram(col_data, bins=bins)
        axs[i].bar(range(len(hist)), hist, width=0.9,
                   tick_label=[f'{bins[j]:.1f}-{bins[j + 1]:.1f}' for j in range(len(hist))])
        axs[i].set_title(f'Column {i}')
        axs[i].set_xlabel('Value Range')
        axs[i].set_ylabel('Frequency')
        axs[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    # -------------------Second Plot
    values = matrix.flatten()

    # Define bins (0 to 1, in steps of 0.1)
    bins = np.linspace(0, 1, 11)

    # Count frequencies
    hist, bin_edges = np.histogram(values, bins=bins)

    # Plot
    plt.bar(range(len(hist)), hist, width=0.9,
            tick_label=[f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(hist))])
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Frequency of Matrix Values in Bins (0-1)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def Time_Modify(forward_exec, backward_exec):
    full_time =   backward_exec
    # indices1,indices2=generate_two_random_lists(int(n_machines*0.3),n_machines)
    # print(full_time[0:10])
    # modifyvalues(full_time, indices1, 0.5)
    # modifyvalues(full_time, indices2, 0.2)
    # print("\n",full_time[0:10])
    return full_time


def Speed(full_time):
    flops_base = full_time[0]
    speed = []
    speed.append(1)
    for i in range(1, n_machines):
        ratio_list = flops_base / full_time[i]
        ratio_value = ratio_list.mean()
        speed.append(ratio_value)
    return speed


def Speed_ranking(full_time):
    row_sums = full_time.sum(axis=1)
    sorted_indices = np.argsort(row_sums)
    return sorted_indices


from typing import Union

# Allow ints and strings (or any mix of hashable types) in your 4-element groups:
MyElem = Union[int, str]

# 1. Create your dict mapping int → list of 4-element lists
from collections import defaultdict

my_dict: dict[int, list[list[MyElem]]] = defaultdict(list)


# 2. Helper to append a new 4-element group (preserving order & repetition)
def add_group(key: int, group: list[MyElem]) -> None:
    my_dict[key].append(group)


if __name__ == "__main__":
    n_stages, n_machines, forward_exec, backward_exec, forward_comm, backward_comm = Read()
    # Task_lists=make_Tasks(n_machines,n_stages)

    # forward_backward_visualize(forward_exec, backward_exec)

    import numpy as np
    import matplotlib.pyplot as plt

    full_time = Time_Modify(forward_exec, backward_exec)
    speed = Speed(full_time)

    machine_front = []
    for i in range(n_machines):
        machine_front.append(1)

    n_stages -= 1
    flops_base = full_time[0]
    sum_flops = flops_base.sum()

    # ---------------Steps
    Finish_flag = True
    Remaining_flops = remaining_flops(flops_base, machine_front, n_machines, n_stages)
    step = 0
    Sum_A = sum(Remaining_flops)
    step_time = Sum_A / sum(speed)
    speed_ranking = Speed_ranking(full_time)
    # while (Finish_flag and step<5):
    # while (all(x != n_stages for x in machine_front)):

    server_index = 0

    while not all(x == n_stages for x in machine_front):
        # print(f"#-----------------------Step {step}--------------------------")
        step += 1

        Sum_A = sum(Remaining_flops)
        if step == 0:
            step_time = Sum_A / sum(speed)
        mean_value = full_time.mean()
        # print(f" values of {step_time} and {mean_value}")

        B = []
        remaining_time = []
        for i in range(n_machines):
            # if Remaining_flops[i]>0:
            # B.append(Remaining_flops[i]/speed[i])
            remaining_time.append(step_time)

        # arr = np.array(B)
        # #sorted_indices = np.argsort(arr)
        sorted_indices = speed_ranking

        # Use indices to sort the array
        # sorted_values = arr[sorted_indices]
        client_index = n_machines - 1
        t = 0

        # print("start Assignment")
        # print  (f"front : {machine_front}")
        # print  (f"sorted index : {sorted_indices}")
        # print  (f"step_time : {step_time}")
        while client_index >= 0:

            server = sorted_indices[server_index % n_machines]
            client = sorted_indices[client_index]
            if machine_front[client] < n_stages:
                # print(f"start for machine {client} with index of {client_index}")
                index = machine_front[client]
                first_index = index
                initial_time = t
                while (index < n_stages and t + full_time[server][index] <= step_time):
                    t = t + full_time[server][index]
                    index = index + 1
                if index == n_stages:
                    # print (f" Normal ,tasks {first_index} to {index-1} of  {client} is complete on {server} with {t-initial_time} and t: {t}")
                    add_group(server, [client, first_index, index - 1, "N", step])
                    machine_front[client] = index
                if index < n_stages:
                    # print (f" Proior, tasks {first_index} to {index-1} of  {client} is Incomple on {server} with time {t} and t: {t}")
                    add_group(server, [client, first_index, index - 1, "P", step])

                    server_index += 1
                    t = 0
                    initial_t2 = 0
                    index2 = index
                    server = sorted_indices[server_index % n_machines]
                    while (index2 < n_stages and t + full_time[server][index2] <= step_time):
                        t = t + full_time[server][index2]
                        index2 += 1
                    # print (f" Low2nd, tasks {index} to {index2-1} of machine {client} is done on amchine {server} with {t-initial_t2} and t:{t}")
                    add_group(server, [client, index, index2 - 1, "L", step])

                    remaining_time[server] -= t
                    machine_front[client] = index2
                # else:
                #     server_index+=1

            client_index -= 1
            Remaining_flops = remaining_flops(flops_base, machine_front, n_machines, n_stages)
            if Remaining_flops == 0:
                Finish_flag = False

    full_assigned = []
    # print(my_dict)
    for i in range(n_machines):
        assigned = []
        assigned.append(str(i) + "_0")
        target = my_dict[i]
        # print(f"machi9ne {i}")
        for step_ in range(1, step+1):
            pos = 4  # third element
            filtered_set = [g for g in target if g[pos] == step_]
            for s in filtered_set:
                if s[3] == 'P':
                    client = s[0]
                    start = s[1]
                    end = s[2]
                    for k in range(start, end + 1):
                        assigned.append(str(client) + "_" + str(k))

            for s in filtered_set:
                if s[3] == 'N':
                    client = s[0]
                    start = s[1]
                    end = s[2]
                    for k in range(start, end + 1):
                        assigned.append(str(client) + "_" + str(k))

            for s in filtered_set:
                if s[3] == 'L':
                    client = s[0]
                    start = s[1]
                    end = s[2]
                    for k in range(start, end + 1):
                        assigned.append(str(client) + "_" + str(k))

        assigned.append(str(i) + "_" + str(n_stages))

        full_assigned.append(assigned)
    print(full_assigned)

    converted_Config = Optimization.convert_to_index1_solution(full_assigned)
    print("SIMULATION_STAGES_CLIENTS=",converted_Config)

    optimization = Optimization(num_clients=100 , num_partitons=number_partitions)
    final_config=optimization.add_backward_stages(converted_Config)
    output = f"SIMULATION_STAGES_CLIENTS={final_config}"
    print(f"{output}")










