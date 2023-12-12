#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)
import json

import time
from datetime import datetime

from argparse import ArgumentParser

import pytorch_lightning as pl

from model_int import ODE_RNN
from dataloader_mujoco import MujocoDataModule
from dataloader_syn import SynDataModule
from dataloader_phy import PhysionetDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(hparams):

    # Deterministicly fix random seeds
    pl.utilities.seed.seed_everything(0)

    # init module
    if hparams.experiment == "mujoco":
        dm = MujocoDataModule(data_set=hparams.data_set, batch_size=hparams.batch_size)
    elif hparams.experiment == "syn":
        dm = SynDataModule(data_set=hparams.data_set, batch_size=hparams.batch_size, diff=False)
    elif hparams.experiment == "mimic":
        dm = PhysionetDataModule(data_set=hparams.data_set, batch_size=hparams.batch_size, diff=False)
    else:
        raise Exception('Invalid experiment!')

    # Pass hyperparameters:
    model = ODE_RNN(hparams)
    print(f"Hyperparameters: {hparams}")

    # monitor tracks the lowest val_loss to save model (best_loss)
    checkpoint_callback = ModelCheckpoint(
        dirpath='lightning_logs',
        filename='ode_rnn_weights.ckpt',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=True,
        patience=10,
        min_delta=0.00
    )

    # Do NOT use multi_gpu because its much slower unless for testing purposes
    gpus = list(map(int, hparams.gpus.split(',')))  # cast string to list of ints
    if hparams.multi_gpu:
        print("#######Training on multi-gpu")
        assert len(gpus) > 1, "Number of GPUs should be greater than 1"
        trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                             min_epochs=50,
                             max_epochs=hparams.num_epochs,
                             gpus=gpus,
                             strategy='ddp')
    else:
        assert len(gpus) == 1,  "Number of GPUs should be equal to 1"
        trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
                             min_epochs=50,
                             max_epochs=hparams.num_epochs,
                             gpus=gpus)

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path='best')
    return model


if __name__ == '__main__':

    start_time = time.time()
    print(f"Run started at {datetime.now()}")

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--experiment', type=str)  # Find out what experiment to do
    hparams, unknown = parser.parse_known_args()

    if hparams.experiment == 'mujoco':
        params_file = 'params_int.json'
    elif hparams.experiment == 'syn':
        params_file = 'params_syn_int.json'
    elif hparams.experiment == 'mimic':
        params_file = 'params_mimic_int.json'
    else:
        raise Exception('Invalid/unspecified experiment! Please check again (mujoco or syn)')

    with open(os.path.join('params', params_file), 'r') as f:
        params = json.load(f)
        f.close()

    input_dim: int = params['input_dim']
    output_dim: int = params['output_dim']
    hidden_size: int = params['hidden_size']
    drop_out: float = params['drop_out']
    data_set: str = params['data_set']
    batch_size: int = params['batch_size']
    sequence_length: int = params['sequence_length']
    multi_gpu: bool = params['multi_gpu']
    solver: str = params['solver']
    step_size: float = params['step_size']
    num_epochs: int = params['num_epochs']
    learning_rate: float = params['learning_rate']
    gpus: str = params['gpus'] if 'gpus' in params else '0'  # Specifying what GPUs you want to use, otherwise use gpu 0

    parser.add_argument('--input_dim', type=int, default=input_dim,
                        help="Number of Input dimensions")
    parser.add_argument('--output_dim', type=int, default=output_dim,
                        help="Number of output dimensions")
    parser.add_argument('--hidden_size', type=int, default=hidden_size,
                        help="Number of hidden dimensions for the neural network")
    parser.add_argument('--drop_out', type=float, default=drop_out,
                        help="Dropout probability for the dropout layers")
    parser.add_argument('--data_set', type=str, default=data_set,
                        help="File path of the training dataset")
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help="Batch size for training/inference")
    parser.add_argument('--sequence_length', type=int, default=sequence_length,
                        help="Length of the sequence of the data for the ODE-RNN")
    parser.add_argument('--multi_gpu', type=bool, default=multi_gpu,
                        help="Use multi GPU or not")
    parser.add_argument('--solver', type=str, default=solver,
                        help="Which ODE solver to use for the evolver. Options: euler, dopri5")
    parser.add_argument('--step_size', type=float, default=step_size,
                        help="Step size for the ODE evolver")
    parser.add_argument('--num_epochs', type=int, default=num_epochs,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=learning_rate,
                        help="Learning rate for the optimizer")
    parser.add_argument('--gpus', type=str, default=gpus,
                        help="Specify the number for which GPUs to use")

    # parse params
    hparams, unknown = parser.parse_known_args()

    model = main(hparams)

    print(f"Run finished at {datetime.now()}")
    print(f"Training completed, total time: {time.time() - start_time} sec")
