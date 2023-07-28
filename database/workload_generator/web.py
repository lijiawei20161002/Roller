import csv  
import json  
import requests  
from operator import itemgetter  
import concurrent.futures  
import os  
  
class InvalidFormatError(Exception):  
    pass  
  
def download_file_from_url(url, local_path):  
    response = requests.get(url)  
    response.raise_for_status()  
  
    with open(local_path, 'wb') as f:  
        f.write(response.content)  
  
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
  
def merge_and_sort_jobs(urls, output_file):  
    all_jobs = []  
    downloaded_files = []  
  
    with concurrent.futures.ThreadPoolExecutor() as executor:  
        future_downloads = {executor.submit(download_file_from_url, url, f"{url.split('/')[-1]}"): url for url in urls}  
  
        for future in concurrent.futures.as_completed(future_downloads):  
            url = future_downloads[future]  
            try:  
                local_file = future.result()  
                downloaded_files.append(local_file)  
            except requests.exceptions.RequestException as e:  
                print(f"Failed to download {url}: {e}")  
                exit(1)  
  
    with concurrent.futures.ThreadPoolExecutor() as executor:  
        future_jobs = {}  
  
        for file in downloaded_files:  
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
  
    # Clean up downloaded files  
    for file in downloaded_files:  
        os.remove(file)  
  
if __name__ == "__main__":  
    urls = ["https://example.com/workload1.csv", "https://example.com/workload2.json", "https://example.com/workload3.csv"]  
    output_file = "merged_sorted_workload.csv"  
    merge_and_sort_jobs(urls, output_file)  
