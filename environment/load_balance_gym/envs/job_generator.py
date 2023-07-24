from load_balance_gym.envs.param import config
import csv
#appointments = [29, 108, 145, 131, 91, 43, 28, 97, 135, 124, 91, 41, 27, 95, 132, 121, 91, 42, 27, 91, 128, 119, 90, 43, 28, 90, 130, 120, 90, 45, 31, 87, 127, 107, 90, 52, 31, 90, 125, 107, 89, 49]

def generate_job(np_random, mystep):
    #size = int((np_random.pareto(config.job_size_pareto_shape)+1)*config.job_size_pareto_scale)
    #size = appointments[mystep%42]
    file = open("configuration/runtime.csv", 'r')
    data = list(csv.reader(file, delimiter='\t'))[0]
    data = data[0].split('\t')
    size = int(float(data[mystep%100]))
    if size > 100:
        size = 100
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