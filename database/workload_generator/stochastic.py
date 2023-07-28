import random  
import csv  
import numpy as np  
  
# Customizable stochastic processes for time and size  
def exponential_time(rate):  
    return random.expovariate(rate)  
  
def poisson_time(rate):  
    return np.random.poisson(rate)  
  
def normal_time(mean, std_dev):  
    return random.gauss(mean, std_dev)  
  
def uniform_time(min_time, max_time):  
    return random.uniform(min_time, max_time)  
  
def uniform_size(min_size, max_size):  
    return random.uniform(min_size, max_size)  
  
def exponential_size(rate):  
    return random.expovariate(rate)  
  
def poisson_size(rate):  
    return np.random.poisson(rate)  
  
def normal_size(mean, std_dev):  
    return random.gauss(mean, std_dev)  
  
def generate_workload(num_jobs, arrival_process, size_process, output_file):  
    # Initialize current timestamp and workload list  
    current_timestamp = 0  
    workload = []  
  
    # Generate workload  
    for _ in range(num_jobs):  
        arrival_time = arrival_process()  
        job_size = size_process()  
        current_timestamp += arrival_time  
        workload.append((current_timestamp, job_size))  
  
    # Write workload to csv file  
    with open(output_file, 'w', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(['Timestamp', 'Size'])  
        for job in workload:  
            csvwriter.writerow(job)  
  
if __name__ == "__main__":  
    num_jobs = 1000  
    output_file = "workload.csv"  
  
    # Example using exponential arrival time and normal job size  
    arrival_rate = 1  
    mean_size, std_dev_size = 500, 100  
    generate_workload(num_jobs,  
                      arrival_process=lambda: exponential_time(arrival_rate),  
                      size_process=lambda: normal_size(mean_size, std_dev_size),  
                      output_file=output_file)  
