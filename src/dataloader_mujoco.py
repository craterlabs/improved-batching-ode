#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ting Fung Lam, Yony Bresler, Ahmed Khorshid, Khalid Eidoo
@license: Crater Labs (C)
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import h5py

output_dtype = torch.float32


class ODEDataset(Dataset):

    def __init__(self, dataset='data/Mujoco.h5',
                 test_validation_split=0.2, train_set=True, test_set=False,
                 device='cpu'):
        """
        dataset: file path of the training dataset
        test_validation_split: How much data to be used for validation and testing combined
        train_set: Load training set data
        test_set: Load test set data. If !(train_set, test_set) load validation set data
        device: Where to load the data to
        """
        # Please make sure to use the right dataset for the ode_rnn or odeint model
        # Use Mujoco.h5/Mujoco_10.h5 for ode_rnn and Mujoco_int.h5/Mujoco_10_int.h5 for odeint
        super(ODEDataset, self).__init__()

        h5f = h5py.File(dataset, 'r')
        dataset = h5f['dataset'][:]
        h5f.close()

        self.store = None
        # Perform 80:10:10 split for train/val/test datasets
        train_dataset, test_val_dataset = train_test_split(dataset, test_size=test_validation_split, random_state=0)
        val_dataset, test_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=0)

        self.store = torch.tensor(train_dataset, device=device) if train_set else \
            torch.tensor(val_dataset, device=device)
        if test_set:  # Test set for evaluation. Overrides train_set
            self.store = torch.tensor(test_dataset, device=device)
        self.len = self.store.shape[0]  # num_samples out of 10000 by default
        self.time_steps = self.store.shape[1]  # 50 by default
        self.dim = self.store.shape[2]  # 15 by default
        self.device = device

    def __getitem__(self, index):
        '''
        x: data points x_i where i = 0 .. N - 1, where N is the number of time steps (self.time_steps)
        x_jumps: corresponding time differences delta t_i where i = 0 .. N - 1
        final jump: delta t_i where i = N
        y: target truth value
        '''
        x = self.store[index, 0:self.store.shape[1] - 1, :]  # Includes x_jumps as well
        x_jumps = self.store[index, 0:self.store.shape[1] - 1, self.store.shape[2] - 1]
        final_jump = self.store[index, self.store.shape[1] - 1, self.store.shape[2] - 1]
        y = self.store[index, self.store.shape[1] - 1, 0:self.store.shape[2] - 1]
        return x.to(output_dtype), x_jumps.to(output_dtype), \
            final_jump.to(output_dtype), y.to(output_dtype)

    def __len__(self):
        return self.len


class MujocoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_set):
        super().__init__()
        self.data_set = data_set
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.data_train = ODEDataset(dataset=self.data_set, train_set=True)
        self.data_val = ODEDataset(dataset=self.data_set, train_set=False)
        self.data_test = ODEDataset(dataset=self.data_set, train_set=False, test_set=True)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=5)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=5)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False, num_workers=5)


if __name__ == '__main__':
    dataset = ODEDataset(train_set=True, test_set=False)
    print(f"Sample size: {dataset.__len__()}")
    x, x_jumps, final_jump, y, _ = dataset.__getitem__(0)

    print(f"x shape: {x.shape}")  # [49, 15] by default
    print(f"jumps shape: {x_jumps.shape}")  # [49] by default
    print(f"final_jump shape: {final_jump.shape}")  # scalar by default
    print(f"y shape: {y.shape}")  # [14] by default
