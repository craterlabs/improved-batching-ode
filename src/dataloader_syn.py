#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ting Fung Lam, Yony Bresler, Ahmed Khorshid, Khalid Eidoo
@license: Crater Labs (C)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

output_dtype = torch.float32


class ODEDataset(Dataset):

    def __init__(self, data_set='', sequence_length=7,
                 test_validation_split=0.2, train_set=True, test_set=False,
                 device='cpu', diff=True):
        """
        dataset: file path of the training dataset
        test_validation_split: How much data to be used for validation and testing combined
        train_set: Load training set data
        test_set: Load test set data. If !(train_set, test_set) load validation set data
        device: Where to load the data to
        sequence_length: Length of generated sequence
        diff: Convert time points t to time differences delta t for ODE_RNN model, otherwise for ODE int model
        """
        super(ODEDataset, self).__init__()

        dataset = []
        a = 1
        step = 0.001 if data_set == '0.001' else 0.1
        print(f"Data time points step: {str(step)}")
        sequence_length = 50
        np.random.seed(0)

        # Refer to Duvenaud Latent ODE paper for toy dataset generation
        for i in range(10000):
            omega = 2 * 3.14 * np.random.uniform(low=0.5, high=1.0)
            starting_point = np.random.normal(loc=1.0, scale=0.1)
            pool = np.arange(5, step=step)
            t = np.sort(np.random.choice(pool, size=(sequence_length,), replace=False))
            epsilon = np.random.normal(loc=0.0, scale=0.1, size=(sequence_length,))
            x = a * np.sin(omega * (t + starting_point)) + epsilon
            if diff:  # convert to time differences or keep the absolute timesteps?
                t_diff = np.diff(t, prepend=t[0])
                dataset.append(np.stack([x, t_diff]).T)
            else:
                dataset.append(np.stack([x, t]).T)
        dataset = np.asarray(dataset)

        self.store = None
        # Perform 80:10:10 split for train/val/test datasets
        train_dataset, test_val_dataset = train_test_split(dataset, test_size=test_validation_split, random_state=0)
        val_dataset, test_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=0)

        self.store = torch.tensor(train_dataset, device=device) if train_set else  \
            torch.tensor(val_dataset, device=device)
        if test_set:  # Test set for evaluation. Overrides train_set
            self.store = torch.tensor(test_dataset, device=device)
        self.len = self.store.shape[0]  # num_samples out of 10000
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
        x = self.store[index, 0:self.store.shape[1] - 1, :]  # Included x_jumps as well
        x_jumps = self.store[index, 0:self.store.shape[1] - 1, self.store.shape[2] - 1]
        final_jump = self.store[index, self.store.shape[1] - 1, self.store.shape[2] - 1]
        y = self.store[index, self.store.shape[1] - 1, 0:self.store.shape[2] - 1].squeeze()  # y should be scalar if 1D
        return x.to(output_dtype), x_jumps.to(output_dtype), final_jump.to(output_dtype),  \
            y.to(output_dtype)

    def __len__(self):
        return self.len


class SynDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_set, diff):
        super().__init__()
        self.data_set = data_set
        self.batch_size = batch_size
        self.diff = diff

    def setup(self, stage=None):
        self.data_train = ODEDataset(data_set=self.data_set, train_set=True, diff=self.diff)
        self.data_val = ODEDataset(data_set=self.data_set, train_set=False, diff=self.diff)
        self.data_test = ODEDataset(data_set=self.data_set, train_set=False, test_set=True, diff=self.diff)

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
    print(f"x shape: {x.shape}")  # [49, 2] by default
    print(f"jumps shape: {x_jumps.shape}")  # [49] by default
    print(f"final_jump shape: {final_jump.shape}")  # scalar by default
    print(f"y shape: {y.shape}")  # scalar by default
