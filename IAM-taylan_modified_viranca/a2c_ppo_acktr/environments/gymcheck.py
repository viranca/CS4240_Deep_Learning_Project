from stable_baselines3.common.env_checker import check_env
from gymware import * 


env = Warehouse()
check_env(env,warn=True)
