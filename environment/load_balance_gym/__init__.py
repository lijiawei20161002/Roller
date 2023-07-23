from gym.envs.registration import register

register(id='load_balance-v1', entry_point='load_balance_gym.envs:LoadBalanceEnv')