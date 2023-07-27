import gym
import load_balance_gym

env = gym.make('load_balance-v1')
observation = env.reset()
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation, reward, done, info)