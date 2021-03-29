#!/bin/sh
#
#PBS -N RL-rnn
#PBS -l nodes=1:ppn=8
#PBS -q guest
#PBS -M J.Smith@example.com
#PBS -o out.$PBS_JOBID
#PBS -e err.$PBS_JOBID
# Start echo_test example job
cd $PBS_O_WORKDIR
source ~/.bashrc
source ~/environments/RL/bin/activate
cd ~/RL/IAM
echo $PWD
echo "$(date)"
python main2.py --env-name "ware" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir "ware/rnn/results/1" --save-dir "ware/rnn/model/1" --recurrent-policy
echo "$(date)"
