import stable_baselines3.common.results_plotter as pu
import matplotlib.pyplot as plt 
LOG_DIRS = 'results/pong/cnn/results/1'
#results = pu.load_results(LOG_DIRS)
fig = pu.load_results(LOG_DIRS)
plt.savefig("look.pdf")
