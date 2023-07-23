import gym
import random
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

algs = ['A2C', 'ACER', 'ACKTR', 'DQN', 'PPO1', 'PPO2', 'TRPO']
algs.append('random')
algs.append('shortest queue')
algs.append('dedicated server')

data = pd.read_csv("healthcare.csv")
sets = []
for alg in algs:
    sets.append(set(data[data['alg']==alg]['state']))
union = set.union(*sets)
union = list(union)
print(union)

data = data[['alg', 'state', 'action']]
df = pd.DataFrame(columns=['alg', 'state', 'action'])
for state in union:
    df2 = data[data['state']==state]
    df = df.append(df2)

l1 = pd.DataFrame(columns=algs)
for alg1 in algs:
    sim = []
    for alg2 in algs:
        sm = 0
        cnt = 0
        action1 = df.loc[(df['state']==state)&(df['alg']==alg1)]['action']
        if action1.empty:
            sm=float(sm)/len(union)
            sim.append(sm)
            continue
        for state in union:
            action2 = df.loc[(df['state']==state)&(df['alg']==alg2)]['action']
            if action2.empty:
                continue
            for r1 in action1:
                for r2 in action2:
                    cnt += 1
                    sm+= abs(1-(float(r1)==float(r2)))
        if cnt > 0:
            sm=float(sm)/cnt
        sim.append(sm)
    l1.loc[-1] = sim
    l1.index = l1.index + 1
    l1 = l1.sort_index()
l1.index = algs
l1 = l1.iloc[::-1]

for i in range(len(l1)):
    #l1.loc[algs[i],algs[i]] = 0
    for j in range(i):
        l1.loc[algs[i], algs[j]] = l1.loc[algs[j], algs[i]]
plt.rcParams.update({'font.size': 22})
plt.figure(figsize=(23, 23))
sns.heatmap(l1, annot=True)
plt.savefig("l1.png")