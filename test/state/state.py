import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('queue_size_state.csv')
df2 = pd.read_csv('full_state.csv')
df3 = pd.read_csv('job_size_pos_num_state.csv')
algs = ['queue size state', 'full state', 'job size pos num state']
labels = ['queue size state', 'full state', 'max job position & size + active job num']
df = pd.concat([df1, df2, df3])
for i in range(len(algs)):
    alg = algs[i]
    label = labels[i]
    sns.lineplot(data=df[df['alg']==alg], x='step', y='cumulative reward', label=label)
plt.grid()
plt.legend()
plt.savefig("state.png")
