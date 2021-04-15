from visualization import plot_util as pu
import matplotlib as mpl
# mpl.style.use('seaborn')


# LOG_DIRS = 'logs/halfcheetah'
# Uncomment below to see the effect of the timit limits flag

# FNN (warehouse)
# LOG_DIRS = 'results-taylan/ware/fnn/results/1_8' # FNN result 8 cores VIR
# LOG_DIRS = 'ware/fnn/results/2' # FNN result 1 core VIR

# CNN (pong)
# LOG_DIRS = 'results-taylan/pong/cnn/results/2' # CNN result 8 cores TAY

# FNN + RNN (warehouse, series)
# LOG_DIRS = 'ware/fnn/results/1' # FNN result 8 cores VIR
# LOG_DIRS = 'ware/fnn/results/2' # FNN result 1 core VIR

# FNN + RNN (warehouse, parallel)
# LOG_DIRS = 'results-taylan/ware/fnnrnnp/results/1' # FNN result 8 cores TAY
# LOG_DIRS = 'results-taylan/ware/fnnrnnp/results/2' # FNN result 1 core TAY

# CNN + RNN (pong, series)
# LOG_DIRS = 'results-kevin/pong/CNNRNN/results/1' # CNN + RNN Series result 8
# cores KEV

# CNN + RNN (pong, parallel)
# LOG_DIRS = 'results-taylan/pong/cnn/results/1' # CNN + RNN Parallel result 8
# cores
# TAY
selected_env = 'ware'
if selected_env == 'ware':
    color = ['red', 'firebrick', 'green', 'chartreuse', 'black', 'blue',
              'royalblue']
elif selected_env =='pong':
    color = ['red','green','blue']

# GROUPED UP
# LOG_DIRS = 'results-taylan/ware/fnnrnnp/results/1'
if selected_env=='ware':
    LOG_DIRS = 'together/ware'
else:
    LOG_DIRS = 'together/pong'
# LOG_DIRS = 'together_cores_seperated/ware'
results = pu.load_results(LOG_DIRS)

fig = pu.plot_results(results, average_group=True, split_fn=lambda _: '',
                      shaded_std=False,COLORS=color,select_env=selected_env,
                      xlabel='Steps', ylabel='Average Reward',legend_outside=True)
#%%
import matplotlib.pyplot as plt
import matplotlib

if selected_env == 'ware':
    fig = matplotlib.pyplot.gcf()
    plt.gcf().subplots_adjust(bottom=0.15,top=0.9,left=0.15,right=0.7)
    fig.set_size_inches(30, 30)
    plt.savefig('plot_ware.png', dpi=150)
elif selected_env == 'pong':
    fig = matplotlib.pyplot.gcf()
    plt.gcf().subplots_adjust(bottom=0.15,top=0.9,left=0.15, right=0.3)
    fig.set_size_inches(30, 30)
    plt.savefig('plot_pong.png', dpi=150)
else:
    pass
# LOG_DIRS = 'results-taylan/ware/rnn/results/1_1' # FNN result 8 cores VIR
# LOG_DIRS2 = 'results-taylan/ware/rnn/results/1_8' # FNN result 8 cores VIR
#
# results = pu.load_results(LOG_DIRS)
# results2 = pu.load_results(LOG_DIRS2)
#
# fig1, ax1 = pu.plot_results(results, average_group=True,
#                       split_fn=lambda _:
# '',
#                       shaded_std=True,shaded_err=True,
#                       legend_outside=True, xlabel='Steps',
#                       ylabel='Average Reward')
# fig2, ax2 = pu.plot_results(results, average_group=True,
#                       split_fn=lambda _:
# '',
#                       shaded_std=True,shaded_err=True,
#                       legend_outside=True, xlabel='Steps',
#                       ylabel='Average Reward')




# plt.show()

# # plt.plot(results)
# x = LOG_DIRS.split('/')
# filename = x[1]
# # L = fig.legend()
# # L.get_texts()[0].set_text('FNN (1 core)')

# fig.legend(labels=['FNN (8 core)', 'FNN (1 core)'], bbox_to_anchor=(0.9,0.2), loc="lower right",  bbox_transform=fig.transFigure)
#
# plt.savefig('plot_ware.png', dpi=300, bbox_inches='tight')
# plt.show()