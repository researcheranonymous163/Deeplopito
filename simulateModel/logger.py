import logging
import os


client_number = os.environ.get('CLIENT_NUMBER')

# create logger
def create_logger(filename, clear_log=True):
    print("create logger ")
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{filename}.log")   
    if clear_log and os.path.exists(log_file):
        with open(log_file, 'w'): 
            pass
    logger = logging.getLogger(f"{filename}")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
  
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def write_to_file(file_name, data, overwrite=False):
    # Open file in 'w' mode if overwrite is True, else 'a' for appending
    mode = 'w' if overwrite else 'a'
    
    with open(file_name, mode) as file:
          file.write(f"{data}\n")


def write_accuracy(accuracy,overwrite=False):
    global client_number
    # Open file in 'w' mode if overwrite is True, else 'a' for appending
    file_name = f"accuracy_results/{client_number}.txt"
    mode = 'w' if overwrite else 'a'

    with open(file_name, mode) as file:
        file.write(f"{accuracy}\n")


def reset_logs_time_file():
    with open("log_time/logs_time.txt", 'w') as file:
        file.write("start log data\n")  # Or write a header if needed


def write_time_execution(message , overwrite=False):
    file_name = "log_time/logs_time.txt"
    mode = 'w' if overwrite else 'a'
    with open(file_name, mode) as file:
        file.write(f"{message}\n")

def write_bussy_times(message , overwrite=False):
    file_name = "log_time/busy.txt"
    mode = 'w' if overwrite else 'a'
    with open(file_name, mode) as file:
        file.write(f"{message}\n")

def write_time_values( values , file_name):
    import csv
    from typing import TextIO

    file_name = f"log_time/{file_name}"
    with open(file_name, mode='w', newline='') as csvfile:  # type : TextIO
        fieldnames = ['Train_client', 'Train_all_client', 'Pull_client', 'Pull_all_client', 'Full_time_client',
                      'full_time_all_client']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for task in values:
            writer.writerow(task)
    print(f"✅ Tasks written successfully to {file_name}")


def write_to_csv(values, file_name):
    import csv
    from typing import TextIO

    file_name = f"log_time/{file_name}"
    if not values:
        print("⚠️ No values to write.")
        return

    with open(file_name, mode='w', newline='') as csvfile:  # type: TextIO
        # Dynamically get fieldnames from the first dictionary
        fieldnames = list(values[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header and rows
        writer.writeheader()
        writer.writerows(values)

    print(f"✅ Tasks written successfully to {file_name}")