import csv  
import json  
from operator import itemgetter  
import concurrent.futures  
  
class InvalidFormatError(Exception):  
    pass  
  
def read_jobs_from_csv(file):  
    jobs = []  
    with open(file, 'r') as csvfile:  
        csvreader = csv.reader(csvfile)  
        header = next(csvreader)  
  
        if 'Timestamp' not in header or 'Size' not in header:  
            raise InvalidFormatError(f"Invalid CSV format in {file}. 'Timestamp' and 'Size' columns are required.")  
  
        timestamp_idx = header.index('Timestamp')  
        size_idx = header.index('Size')  
  
        for row in csvreader:  
            jobs.append((float(row[timestamp_idx]), float(row[size_idx])))  
    return jobs  
  
def read_jobs_from_json(file):  
    jobs = []  
    with open(file, 'r') as jsonfile:  
        data = json.load(jsonfile)  
  
        if not all(key in data for key in ['Timestamp', 'Size']):  
            raise InvalidFormatError(f"Invalid JSON format in {file}. 'Timestamp' and 'Size' keys are required.")  
  
        for timestamp, size in zip(data['Timestamp'], data['Size']):  
            jobs.append((timestamp, size))  
    return jobs  
  
def write_jobs_to_csv(jobs, output_file):  
    with open(output_file, 'w', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(['Timestamp', 'Size'])  
        for job in jobs:  
            csvwriter.writerow(job)  
  
def merge_and_sort_jobs(input_files, output_file):  
    all_jobs = []  
  
    with concurrent.futures.ThreadPoolExecutor() as executor:  
        future_jobs = {}  
  
        for file in input_files:  
            if file.endswith('.csv'):  
                future = executor.submit(read_jobs_from_csv, file)  
            elif file.endswith('.json'):  
                future = executor.submit(read_jobs_from_json, file)  
            else:  
                print(f"Unsupported file format for {file}. Skipping.")  
                continue  
  
            future_jobs[future] = file  
  
        for future in concurrent.futures.as_completed(future_jobs):  
            file = future_jobs[future]  
            try:  
                all_jobs.extend(future.result())  
            except InvalidFormatError as e:  
                print(e)  
                exit(1)  
  
    all_jobs.sort(key=itemgetter(0))  
    write_jobs_to_csv(all_jobs, output_file)  
  
if __name__ == "__main__":  
    input_files = ["1.csv", "2.csv", "3.csv"]  
    output_file = "merged_sorted_workload.csv"  
    merge_and_sort_jobs(input_files, output_file)  

