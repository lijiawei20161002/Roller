from load_balance_gym.envs.config import config
import pandas as pd

def generate_job(mystep):
    df = pd.read_csv(config.get('database', 'database'))
    size = float(df['Size'][mystep])
    t = float(df['Timestamp'][mystep])
    return t, size

def generate_jobs(num_stream_jobs, np_random):
    all_t = []
    all_size = []

    t = 0
    for _ in range(num_stream_jobs):
        t, size = generate_job(np_random)
        all_t.append(t)
        all_size.append(size)

    return all_t, all_size