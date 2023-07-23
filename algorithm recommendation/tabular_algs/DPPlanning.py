import numpy as np
from load_balance_gym.envs.param import config

class DPPlanning:
    def __init__(self, env, num_states, num_actions):
        self.env = env
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = np.zeros([num_states, num_actions])
        self.policy = np.random.choice(self.num_actions, self.num_states)

    def run_q_learning(self, num_episodes, verbose=True):
        for episode in range(num_episodes):
            print('Q-Learning: episode ', episode)
            state = self.env.reset()
            terminate = False
            while not terminate:
                action = self.argmax(self.Q, self.tuple_to_num(state))
                self.policy[self.tuple_to_num(state)] = action
                next_state, reward, terminate, info = self.env.step(action)
                old_value = self.Q[self.tuple_to_num(state), action]
                next_max = np.max(self.Q[self.tuple_to_num(next_state)])
                new_value = (1-self.alpha)*old_value + self.alpha*(reward+self.gamma*next_max)
                self.Q[self.tuple_to_num(state), action] = new_value
                state = next_state

        final_policy = self.policy
        if verbose:
            print(f"Finished training RL agent for {num_episodes} episodes!")

        return self.Q, final_policy

    def tuple_to_num(self, s):
        for pos in range(len(s)):
            if s[pos] >= config.load_balance_queue_size:
                return config.load_balance_queue_size-1
        return (s[0]//100)*(config.load_balance_queue_size//10)**2 + (s[1]//10)*(config.load_balance_queue_size//10) + (s[2]//10)
    
    def num_to_tuple(self, num):
        s = np.zeros(config.num_servers+1)
        s[i] = num % (config.load_balance_queue_size //10)
        s[0] = num 

    def argmax(self, Q, state):
        best_value = float('-inf')
        for action in range(self.num_actions):
            if Q[state][action] > best_value:
                best_action = action
                best_value = Q[state][action]
        return best_action
