from stable_baselines3.common.cmd_util import make_vec_env

from stable_baselines3 import a2c,ppo

from gymware import Warehouse
import gym
# Instantiate the env
env = Warehouse()
#env = gym.make("PongNoFrameskip-v4")
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

model = a2c.A2C('MlpPolicy', env, verbose=1).learn(5000)
#model = ppo.PPO('MlpPolicy', env, verbose=1).learn(5000)

obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  #print('obs=', obs, 'reward=', reward, 'done=', done)
  #env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break
