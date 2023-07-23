import gym
import random
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

job_num = 10000
iter = 10
vary_length = [10, 100, 1000, 10000]
#algs = ['A2C', 'ACER', 'ACKTR', 'DQN', 'PPO1', 'PPO2', 'TRPO']
algs = ['PPO2']
algs.append('random')
algs.append('shortest_queue')
algs.append('dedicated_server')

avg_reward = pd.DataFrame()
for l in vary_length:
    df = pd.read_csv(str(l)+"_10000.csv")
    sns.set_style("whitegrid", {'axes.grid': True, 'axes.edgecolor':'black'})
    for alg in algs:
        #sns.lineplot(data=df[df['alg']==alg].iloc[::100,:], x='step', y='cumulative reward', label=alg)
        data = df[df['alg']==alg].iloc[::job_num-1, :] 
        data['cumulative reward'] = data['cumulative reward'].apply(lambda x: 0-x)
        data['Episode Length'] = l
        avg_reward = avg_reward.append(data)
avg_reward.rename(columns={'alg':'Algorithm', 'cumulative reward':'Total Waiting Time'}, inplace=True)
sns.barplot(x='Episode Length', y='Total Waiting Time', hue='Algorithm', data=avg_reward, palette='RdPu')
plt.savefig("test.png")
