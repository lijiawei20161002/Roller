import gym
import random
import argparse
import seaborn as sns
import pandas as pd
from os.path import exists
from load_balance_gym.envs.param import config
from tabular_algs.MCControl import MCControl
from tabular_algs.QLearning import QLearning
import numpy as np
from stable_baselines.deepq.policies import MlpPolicy as mlp_dqn
from stable_baselines.common.policies import MlpPolicy as mlp
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines import A2C
from stable_baselines import ACER
from stable_baselines import ACKTR
from stable_baselines import GAIL
from stable_baselines import PPO1
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines.gail import generate_expert_traj
from stable_baselines.gail import ExpertDataset
# without DDPG, TD3, SAC only intended to work with continuous actions, not support Discrete action space
# without HER, HER requires the environment to inherits from gym.GoalEnv

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="parameters")
parser.add_argument('--figure_name', type=str, default="test.png")
main_config, _ = parser.parse_known_args()

env = gym.make('load_balance-v1', )

model_class = DQN
goal_selection_strategy = 'future'

n_episodes = 100

# add GAIL later
#algs = ['PPO2']
algs = ['A2C', 'ACER', 'ACKTR', 'PPO1', 'PPO2', 'TRPO'] #'DQN', 'PPO1', 'PPO2', 'TRPO']
#models = ['model_PPO2 = PPO2(mlp, env, verbose=1, tensorboard_log="./log/")']
#trains = ['model_PPO2.learn(total_timesteps=n_episodes)']
models = ['model_A2C = A2C(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_ACER = ACER(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_ACKTR = ACKTR(mlp, env, verbose=1, tensorboard_log="./log/")',
#'model_DQN = DQN(mlp_dqn, env, verbose=1, tensorboard_log="./log/")',
'model_PPO1 = PPO1(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_PPO2 = PPO2(mlp, env, verbose=1, tensorboard_log="./log/")',
'model_TRPO = TRPO(mlp, env, verbose=1, tensorboard_log="./log/")']

trains = ['model_A2C.learn(total_timesteps=n_episodes)',
'model_ACER.learn(total_timesteps=n_episodes)',
'model_ACKTR.learn(total_timesteps=n_episodes)',
#'model_DQN.learn(total_timesteps=n_episodes)',
'model_PPO1.learn(total_timesteps=n_episodes)',
'model_PPO2.learn(total_timesteps=n_episodes)',
'model_TRPO.learn(total_timesteps=10)']

print('env.action_space', env.action_space)
#plt.yscale('symlog')
sns.set_style("darkgrid", {'axes.grid': True, 'axes.edgecolor':'black'})

n_steps = 1000
n_iter = 10
df = pd.DataFrame(columns=['alg', 'step', 'cumulative reward'])

def expert(obs):
    threshold = 50
    if obs[0] < threshold:
        action = 0
    else:
        action = np.argmin(obs[2:]) + 1
    return action

for iter in range(n_iter):
    for i in range(len(algs)):
        alg = algs[i]
        rewards = []
        if iter < 1:
            print('=================='+alg+'====================')
            exec(models[i])
            exec(trains[i])
        step = 0
        if n_steps > config.num_stream_jobs:
            test_iter = n_steps//config.num_stream_jobs
        else:
            test_iter = 1
        for ep in range(test_iter):
            obs = env.reset()
            for _ in range(min(n_steps, config.num_stream_jobs)):
                exec('action, _states = model_'+alg+'.predict(obs)')
                state, reward, done, info = env.step(action)
                if step > 0:
                    rewards.append(reward+rewards[step-1])
                else:
                    rewards.append(reward)
                step += 1
        df = df.append(pd.DataFrame({'alg': alg, 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)

    # random
    #print('============== random ===================')
    rewards = []
    step = 0
    if n_steps > config.num_stream_jobs:
        iter = n_steps//config.num_stream_jobs
    else:
        iter = 1
    for ep in range(iter):
        obs = env.reset()
        for _ in range(min(n_steps, config.num_stream_jobs)):
            action = random.randint(0, config.num_servers-1)
            state, reward, done, info = env.step(action)
            if step > 0:
                rewards.append(reward+rewards[step-1])
            else:
                rewards.append(reward)
            step += 1
            #print("random:", state, action, reward, info)
    df = df.append(pd.DataFrame({'alg': 'random', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)

    '''
    # Q Learning
    #print('============== Q Learning ===================')
    obs = env.reset()
    if iter<1:
        #qlearning = QLearning(env, (config.load_balance_queue_size//10)**(config.num_servers+1)*2, config.num_servers, alpha=0.1, gamma=0.9, epsilon=0.01) # state without active job num
        qlearning = QLearning(env, (config.load_balance_queue_size//10)**(config.num_servers)*2, config.num_servers, alpha=0.1, gamma=0.9, epsilon=0.01) # state with active job num
        #qlearning = QLearning(env, ((100//10)**config.load_balance_queue_size) * (config.num_servers +1), config.num_servers, alpha=0.5, gamma=0.9, epsilon=0.01)
        policy = qlearning.run_q_learning(n_episodes)[1]
    rewards = []
    step = 0
    for ep in range(n_steps//config.num_stream_jobs):
        obs = env.reset()
        state = 0
        for _ in range(config.num_stream_jobs):
            #print(state)
            action = policy[state]
            #print('=====policy: ', policy)
            state, reward, done, info = env.step(action)
            if step > 0:
                rewards.append(reward+rewards[step-1])
            else:
                rewards.append(reward)
            step += 1
            #print(state, action, reward, qlearning.Q[qlearning.tuple_to_num(state)], policy[qlearning.tuple_to_num(state)])
            state = qlearning.tuple_to_num(state)
    df = df.append(pd.DataFrame({'alg': 'q learning', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)

    # First-visit Monte Carlo Control
    #print('============== first-visit Monte Carlo ===================')
    if iter<1:
        mc = MCControl(env, (config.load_balance_queue_size//10)**(config.num_servers)*2, config.num_servers, 0.01, 0.9)
        policy = mc.run_mc_control(n_episodes)[1]
    rewards = []
    step = 0
    for ep in range(n_steps//config.num_stream_jobs):
        obs = env.reset()
        state = obs
        for _ in range(config.num_stream_jobs):
            #print(_, "action:", action, "state:", state)
            state = mc.tuple_to_num(state)
            action = policy[state]
            #print('=====policy: ', policy)
            state, reward, done, info = env.step(action)
            if step > 0:
                rewards.append(reward+rewards[step-1])
            else:
                rewards.append(reward)
            step += 1
            print(state, action, reward, mc.Q[mc.tuple_to_num(state)], policy[mc.tuple_to_num(state)])
    df = df.append(pd.DataFrame({'alg': 'monte carlo control', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)
 
    # expert 
    print('============== expert ===================')
    obs = env.reset()
    rewards = []
    #if not exists('expert_load_balance.npz'):
    generate_expert_traj(expert, 'data/expert_load_balance', env, n_episodes=10000)
    dataset = ExpertDataset(expert_path='data/expert_load_balance.npz', traj_limitation=100, batch_size=128)
    model = TRPO(mlp, env, verbose=1, tensorboard_log="./log/")
    model.pretrain(dataset, n_epochs=100)
    #model.learn(total_timesteps=n_episodes)
    for step in range(n_steps):
        exec('action, _states = model.predict(obs)')
        state, reward, done, info = env.step(action)
        if step > 0:
            rewards.append(reward+rewards[step-1])
        else:
            rewards.append(reward)
        print(state, action, reward)
    df = df.append(pd.DataFrame({'alg': 'expert', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)'''


    # join the shortest queue
    #print('============== shortest queue ===================')
    rewards = []
    step = 0
    if n_steps > config.num_stream_jobs:
        iter = n_steps//config.num_stream_jobs
    else:
        iter = 1
    for ep in range(iter):
        obs = env.reset()
        action = 0
        for _ in range(min(n_steps, config.num_stream_jobs)):
            state, reward, done, info = env.step(action)
            #action = np.argmin(state[1:len(state)-1]) # state with active job num
            action = np.argmin(state[1:config.num_servers+1]) # state without active job num
            if step > 0:
                rewards.append(reward+rewards[step-1])
            else:
                rewards.append(reward)
            step += 1
            #print(state, action, reward)
    df = df.append(pd.DataFrame({'alg': 'shortest_queue', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)


    # save a dedicated server for short jobs
    #print('============== dedicated server ===================')
    threshold = 1
    rewards = []
    step = 0
    if n_steps > config.num_stream_jobs:
        iter = n_steps//config.num_stream_jobs
    else:
        iter = 1
    for ep in range(iter):
        obs = env.reset()
        action = 1
        for _ in range(min(n_steps, config.num_stream_jobs)):
            state, reward, done, info = env.step(action)
            if state[0] > threshold:
                #action = np.argmin(state[2:len(state)-1]) + 1 # state with active job num
                action = np.argmin(state[2:config.num_servers+1]) + 1 # state without active job num
            else:
                #action = np.argmin(state[1:len(state)-1]) # state with active job num
                action = np.argmin(state[1:config.num_servers+1]) # state without active job num
            if step > 0:
                rewards.append(reward+rewards[step-1])
            else:
                rewards.append(reward)
            step += 1
            #print(state, action, reward)
    df = df.append(pd.DataFrame({'alg': 'dedicated_server', 'step': range(n_steps), 'cumulative reward': rewards}), ignore_index=True)

#print(df[df['alg']=='random'].iloc[n_steps-1]['cumulative reward'])

#algs.append('monte carlo control')
#algs.append('q learning')
#algs.append('expert')
algs.append('random')
algs.append('shortest_queue')
algs.append('dedicated_server')
for alg in algs:
    sns.lineplot(data=df[df['alg']==alg], x='step', y='cumulative reward', label=alg)
plt.legend()
plt.savefig("output/"+main_config.figure_name)
'''
if not exists("output/workload/result.txt"):
    with open("output/workload/result.txt", 'a+') as f:
        f.write("pareto_shape")
        for alg in algs:
            f.write(alg+'\t')
        f.write('\n')
    f.close()
with open("output/workload/result.txt", 'a+') as f:
    f.write(str(config.job_size_pareto_shape)+'\t')
    for alg in algs:
        print(alg+'\t'+str(df[df['alg']==alg].iloc[n_steps-1]['cumulative reward']))
        f.write(str(df[df['alg']==alg].iloc[n_steps-1]['cumulative reward'])+'\t')
    f.write('\n')'''