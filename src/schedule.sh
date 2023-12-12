#!/usr/bin/bash

# @author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Khalid Eidoo
# @license: Crater Labs (C)

# Commands to run experiment for our model. Please comment the commands for the experiment you want to skip running
# Use nohup on this to avoid interruptions
# nohup sh schedule.sh > nohup.out &

# If you want to switch datasets, add the corresponding flags to the experiments you want to run.
# See the last experiment for Mujoco/Synthetic below for examples

# For multi GPU option, please reduce the batch size by dividing the original batch size by the number of GPUs, then add "--multi_gpu True" arguments

mkdir lightning_logs

# Set of Mujoco experiments
echo "Mujoco 50%: fixed dt"
python train.py --experiment mujoco &> lightning_logs/mujoco50_fixed_dt.out &&
echo "Mujoco 50%: adaptive_fixed"  &&
python train.py --experiment mujoco --mode adaptive_fixed &> lightning_logs/mujoco50_adaptive_fixed.out  &&
echo "Mujoco 50%: adaptive_geometric" &&
python train.py --experiment mujoco --mode adaptive_geometric &> lightning_logs/mujoco50_adaptive_geometric.out &&
echo "Mujoco 50%: Simple RNN (bypass evolver)" &&
python train.py --experiment mujoco --bypass_evolver True &> lightning_logs/mujoco50_simple_rnn.out &&

echo "Mujoco 10%: fixed dt" &&
python train.py --experiment mujoco --data_set "data/Mujoco_10.h5" --sequence_length 9 &> lightning_logs/mujoco10_fixed_dt.out &&
echo "Mujoco 10%: adaptive_fixed"  &&
python train.py --experiment mujoco --data_set "data/Mujoco_10.h5" --sequence_length 9 --mode adaptive_fixed &> lightning_logs/mujoco10_adaptive_fixed.out &&
echo "Mujoco 10%: adaptive_geometric" &&
python train.py --experiment mujoco --data_set "data/Mujoco_10.h5" --sequence_length 9 --mode adaptive_geometric &> lightning_logs/mujoco10_adaptive_geometric.out &&
echo "Mujoco 10%: Simple RNN (bypass evolver)" &&
python train.py --experiment mujoco --data_set "data/Mujoco_10.h5" --sequence_length 9 --bypass_evolver True &> lightning_logs/mujoco10_simple_rnn.out &&


# Set of Synthetic data experiments
echo "Synthetic w/ rounding 0.1: fixed dt" &&
python train.py --experiment syn &> lightning_logs/syn01_fixed_dt.out &&
echo "Synthetic w/ rounding 0.1: adaptive_fixed" &&
python train.py --experiment syn --mode adaptive_fixed &> lightning_logs/syn01_adaptive_fixed.out  &&
echo "Synthetic w/ rounding 0.1: adaptive_geometric" &&
python train.py --experiment syn --mode adaptive_geometric &> lightning_logs/syn01_adaptive_geometric.out &&
echo "Synthetic w/ rounding 0.1: Simple RNN (bypass evolver)" &&
python train.py --experiment syn --bypass_evolver True &> lightning_logs/syn01_simple_rnn.out &&

echo "Synthetic w/ rounding 0.001: fixed dt" &&
python train.py --experiment syn --data_set "0.001" &> lightning_logs/syn0001_fixed_dt.out &&
echo "Synthetic w/ rounding 0.001: adaptive_fixed" &&
python train.py --experiment syn --data_set "0.001" --mode adaptive_fixed &> lightning_logs/syn0001_adaptive_fixed.out  &&
echo "Synthetic w/ rounding 0.001: adaptive_geometric" &&
python train.py --experiment syn --data_set "0.001" --mode adaptive_geometric &> lightning_logs/syn0001_adaptive_geometric.out &&
echo "Synthetic w/ rounding 0.001: Simple RNN (bypass evolver)" &&
python train.py --experiment syn --data_set "0.001" --bypass_evolver True &> lightning_logs/syn0001_simple_rnn.out &&


# Set of MIMIC-IV data experiments
echo "MIMIC-IV: fixed dt" &&
python train.py --experiment mimic &> lightning_logs/mimic_fixed_dt.out &&
echo "MIMIC-IV: adaptive fixed" &&
python train.py --experiment mimic --mode adaptive_fixed &> lightning_logs/mimic_adaptive_fixed.out  &&
echo "MIMIC-IV: adaptive geometric" &&
python train.py --experiment mimic --mode adaptive_geometric &> lightning_logs/mimic_adaptive_geometric.out &&
echo "MIMIC-IV: Simple RNN (bypass evolver)" &&
python train.py --experiment mimic --bypass_evolver True &> lightning_logs/mimic_simple_rnn.out &&

echo "Finished!"
