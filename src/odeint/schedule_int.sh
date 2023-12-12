#!/usr/bin/bash

# @author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Khalid Eidoo
# @license: Crater Labs (C)

# Commands to run experiment for our model. Please comment the commands for the experiment you want to skip running
# Use nohup on this to avoid interruptions
# nohup sh schedule_int.sh > nohup.out &

mkdir lightning_logs

echo "Mujuco 50: euler solver" &&
python train_int.py --experiment mujoco &> lightning_logs/mujoco50_euler.out &&
echo "Mujuco 50: dopri5 solver" &&
python train_int.py --experiment mujoco --solver dopri5 &> lightning_logs/mujoco50_dopri5.out &&
echo "Mujoco 10: euler solver" &&
python train_int.py --experiment mujoco --data_set "../data/Mujoco_10_int.h5" --sequence_length 9 &> lightning_logs/mujoco10_euler.out &&
echo "Mujoco 10: dopri5 solver" &&
python train_int.py --experiment mujoco --data_set "../data/Mujoco_10_int.h5" --sequence_length 9 --solver dopri5 &> lightning_logs/mujoco10_dopri5.out &&

echo "Synthetic w/ rounding 0.1: euler solver" &&
python train_int.py --experiment syn &> lightning_logs/syn01_euler.out &&
echo "Synthetic w/ rounding 0.1: dopri5 solver" &&
python train_int.py --experiment syn --solver dopri5 &> lightning_logs/syn01_dopri5.out &&
echo "Synthetic w/ rounding 0.001: euler solver" &&
python train_int.py --experiment syn --data_set "0.001" &> lightning_logs/syn0001_euler.out &&
echo "Synthetic w/ rounding 0.001: dopri5 solver" &&
python train_int.py --experiment syn --data_set "0.001" --solver dopri5 &> lightning_logs/syn0001_dopri5.out &&

echo "Finished!"