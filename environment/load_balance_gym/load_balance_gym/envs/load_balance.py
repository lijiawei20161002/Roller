from load_balance_gym.envs.config import config
from load_balance_gym.envs.job import Job
from load_balance_gym.envs.job_generator import generate_job
from load_balance_gym.envs.server import Server
from load_balance_gym.envs.timeline import Timeline
from load_balance_gym.envs.wall_time import WallTime

import gym
from gym import error, spaces, utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import numpy as np

class LoadBalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def setup_space(self):
        self.mystep = 0
        self.max_job_size = config.getfloat('parameters', 'max_job_size')
        self.queue_size = config.getfloat('parameters', 'queue_size')
        self.num_servers = config.getint('parameters', 'num_servers')
        self.queue = [self.queue_size] * (self.num_servers +1)
        self.obs_low = np.array([0] * (self.num_servers + 1))
        self.obs_high = np.array([self.queue_size] * (self.num_servers + 1))
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float64)
        self.action_space = spaces.Discrete(self.num_servers)

    def contains(self, observation_space, x):
        return x.shape == observation_space.shape and (x>=observation_space.low).all() and (x<=observation_space.high).all()

    def np_random(seed=42):
        if not (isinstance(seed, int) and seed >= 0):
            raise ValueError('Seed must be a non-negative integer.')
        rng = np.random.RandomState()
        rng.seed(seed)
        return rng

    def seed(self, seed):
        self.np_random = self.np_random(42)

    def __init__(self):
        self.setup_space()
        self.wall_time = WallTime()
        self.timeline = Timeline()
        df = pd.read_csv(config.get('database', 'database'))
        self.num_stream_jobs = len(df)
        service_rates = config.get('parameters', 'service_rates').split('[')[1].split(']')[0].split(',')
        service_rates = [float(rate) for rate in service_rates]
        self.servers = self.initialize_servers(service_rates)
        self.incoming_job = None
        self.finished_jobs = []
        self.reset()

    def generate_job(self):
        if self.num_stream_jobs_left > 0:
            t, size = generate_job(self.mystep)
            self.timeline.push(t, size)
            self.num_stream_jobs_left -= 1

    def initialize(self):
        assert self.wall_time.curr_time == 0
        self.generate_job()
        new_time, obj = self.timeline.pop()
        self.wall_time.update(new_time)
        assert isinstance(obj, float)
        size = obj
        self.incoming_job = Job(size, self.wall_time.curr_time)

    def initialize_servers(self, service_rates):
        servers = []
        for server_id in range(self.num_servers):
            server = Server(server_id, service_rates[server_id], self.wall_time)
            servers.append(server)
        return servers

    def reset(self):
        for server in self.servers:
            server.reset()
        self.wall_time.reset()
        self.timeline.reset()
        self.num_stream_jobs_left = self.num_stream_jobs
        assert self.num_stream_jobs_left > 0
        self.incoming_job = None
        self.finished_jobs = []
        self.initialize()
        return self.observe()
    
    # [queue size] 
    def observe(self):
        self.mystep += 1
        obs_arr = []
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.queue_size:
                print('Incoming job at time '+str(self.wall_time.curr_time)+' has size '+str(self.incoming_job.size)+' larger than obs_high '+str(self.obs_high[-1]))
                obs_arr.append(self.queue_size)
            else:
                obs_arr.append(self.incoming_job.size)
        for server in self.servers:
            load = 0
            if server.curr_job is not None:
                load += (server.curr_job.finish_time - self.wall_time.curr_time)
            load += sum(j.size for j in server.queue)
            obs_arr.append(int(load))
        obs_arr = np.array(obs_arr)
        assert self.observation_space.contains(obs_arr)
        return (obs_arr, {'state representation': '[queue size]'})
    
     # full observation: [tuple_to_num([curr job size, size for each job in queue] for each queue)]
    def observe_2(self):
        obs_arr = []
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.max_job_size:
                print('Incoming job at time '+str(self.wall_time.curr_time)+' has size '+str(self.incoming_job.size)+' larger than obs_high '+str(self.obs_high[-1]))
                obs_arr.append(self.max_job_size)
            else:
                obs_arr.append(self.incoming_job.size)
        for server in self.servers:
            arr = np.zeros(self.queue_size)
            if server.curr_job is not None:
                arr[0] = min(self.max_job_size, server.curr_job.finish_time - self.wall_time.curr_time)
            for i in range(1, self.queue_size):
                if i < len(server.queue):
                    arr[i] = server.queue[i-1].size
                else:
                    arr[i] = 0
            obs_arr.append(self.tuple_to_num(self.max_job_size, arr))
        obs_arr = np.array(obs_arr)
        return (obs_arr, {'state representation': 'full observation: [tuple_to_num([curr job size, size for each job in queue] for each queue)]'})
    
    # [queue size] + [active job number in each queue]
    def observe_3(self):
        obs_arr = []
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.queue_size:
                print('Incoming job at time '+str(self.wall_time.curr_time)+' has size '+str(self.incoming_job.size)+' larger than obs_high '+str(self.obs_high[-1]))
                obs_arr.append(self.queue_size)
            else:
                obs_arr.append(self.incoming_job.size)
        active_job_num = []
        for server in self.servers:
            load = 0
            cnt = 0
            if server.curr_job is not None:
                load += (server.curr_job.finish_time - self.wall_time.curr_time)
                cnt += 1
            load += sum(j.size for j in server.queue)
            cnt += len(server.queue)
            obs_arr.append(int(load))
            active_job_num.append(cnt)
        obs_arr = np.array(obs_arr+active_job_num)
        return (obs_arr, {'state representation': '[queue size] + [active job number in each queue]'})
    
    # [queue size] + [idx of max job in each queue]
    def observe_4(self):
        obs_arr = []
        if self.incoming_job is None:
            obs_arr.append(0)
        else:
            if self.incoming_job.size > self.queue_size:
                print('Incoming job at time '+str(self.wall_time.curr_time)+' has size '+str(self.incoming_job.size)+' larger than obs_high '+str(self.obs_high[-1]))
                obs_arr.append(self.queue_size)
            else:
                obs_arr.append(self.incoming_job.size)
        max_job_idx = []
        for server in self.servers:
            load = 0
            idx = 0
            max_size = 0
            if server.curr_job is not None:
                load += (server.curr_job.finish_time - self.wall_time.curr_time)
                if load > max_size:
                    max_size = load
            cnt = 1
            for j in server.queue:
                load += j.size
                if j.size > max_size:
                    max_size = j.size
                    idx = cnt
                cnt += 1
            obs_arr.append(int(load))
            max_job_idx.append(idx)
        obs_arr = np.array(obs_arr+max_job_idx)
        return (obs_arr, {'state representation': '[queue size] + [idx of max job in each queue]'})

    def step(self, action):
        self.servers[action].schedule(self.incoming_job)
        running_job = self.servers[action].process() # [ToDo] handle per step
        if running_job is not None:
            self.timeline.push(running_job.finish_time, running_job)
        self.incoming_job = None
        self.generate_job()
        reward = 0
        while len(self.timeline) > 0:
            new_time, obj = self.timeline.pop()
            num_active_jobs = sum(len(w.queue) for w in self.servers)
            for server in self.servers:
                for job in server.queue:
                    if job.finish_time is None:
                        reward -= new_time - self.wall_time.curr_time
                    else:
                        reward -= (min(new_time, job.finish_time)-self.wall_time.curr_time)
                if server.curr_job is not None:
                    assert server.curr_job.finish_time >= \
                        self.wall_time.curr_time
                    num_active_jobs += 1
                    if server.curr_job.finish_time is None:
                        reward -= new_time - self.wall_time.curr_time
                    else:
                        reward -= (min(new_time, server.curr_job.finish_time)-self.wall_time.curr_time)
            #old_time = self.wall_time.curr_time
            self.wall_time.update(new_time)
            
            if isinstance(obj, float):
                size = obj
                self.incoming_job = Job(size, self.wall_time.curr_time)
                break
            elif isinstance(obj, Job):
                job = obj
                if not np.isinf(self.num_stream_jobs_left):
                    self.finished_jobs.append(job)
                else:
                    if len(self.finished_jobs) > 0:
                        self.finished_jobs[-1] += 1
                    else:
                        self.finished_jobs = [1]
                if job.server.curr_job == job:
                    job.server.curr_job = None
                running_job = job.server.process()
                if running_job is not None:
                    self.timeline.push(running_job.finish_time, running_job)
            else:
                print('illegal event type')
                exit(1)
        done = ((len(self.timeline) == 0) and \
            self.incoming_job is None)
        obs = self.observe()
        return obs, reward, done, {'curr_time': self.wall_time.curr_time}