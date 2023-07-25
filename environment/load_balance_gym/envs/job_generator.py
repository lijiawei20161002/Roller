from load_balance_gym.envs.config import config
import csv

def generate_job(np_random, mystep):
    file = open(config['database'])
    data = list(csv.reader(file, delimiter='\t'))[0]
    data = data[0].split('\t')
    size = int(float(data[mystep]))
    t = int(np_random.exponential(config.job_interval))
    return t, size

def generate_jobs(num_stream_jobs, np_random):
    all_t = []
    all_size = []

    t = 0
    for _ in range(num_stream_jobs):
        dt, size = generate_job(np_random)
        t += dt
        all_t.append(t)
        all_size.append(size)

    return all_t, all_size