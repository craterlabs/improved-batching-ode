#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""

import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm


def check_mask(data, mask):
    # check that 'mask' argument indeed contains a mask for data
    n_zeros = np.sum(mask == 0.)
    n_ones = np.sum(mask == 1.)
    # mask should contain only zeros and ones
    assert ((n_zeros + n_ones) == np.prod(list(mask.shape)))

    # all masked out elements should be zeros
    assert (np.sum(data[mask == 0.] != 0.) == 0)


if __name__ == '__main__':
    df = pd.read_csv('data/MIMIC-IV-intercepted.csv')
    n_samples = len(df.ID.unique())
    print(f"Number of patients: {n_samples}")
    lens = [len(df[df.ID == patient]) for patient in df.ID.unique()]
    max_len = max(lens)

    D = 102  # Number of dimensions
    time_list = []
    data_list = []
    mask_list = []
    data = None
    mask = None
    for patient in tqdm(df.ID.unique()):
        # Combine all the data features and mask features of a patient, then combine the time feature
        for i in range(D):
            data = np.concatenate([data, np.array(df[df.ID == patient]['Value_label_' + str(i)])[:, None]],
                                  axis=-1) if i > 0  \
                else np.array(df[df.ID == patient]['Value_label_' + str(i)])[:, None]
        for i in range(D):
            mask = np.concatenate([mask, np.array(df[df.ID == patient]['Mask_label_' + str(i)])[:, None]],
                                  axis=-1) if i > 0  \
                else np.array(df[df.ID == patient]['Mask_label_' + str(i)])[:, None]
        time = np.array(df[df.ID == patient]['Time'])[:, None]

        # Pad zeros to the longest sequence
        data = np.concatenate([np.zeros((max_len - len(df[df.ID == patient]), D)), data], axis=0)
        mask = np.concatenate([np.zeros((max_len - len(df[df.ID == patient]), D)), mask], axis=0)
        time = np.concatenate([np.zeros((max_len - len(df[df.ID == patient]), 1)), time], axis=0)

        check_mask(data, mask)

        time_list.append(time)
        data_list.append(data)
        mask_list.append(mask)

    result_dataset = [np.stack(time_list), np.stack(data_list), np.stack(mask_list)]
    result_dataset[0] = result_dataset[0].squeeze()
    print(f"The resulted dataset shape are (time, data, mask): \
        {list(result_dataset[0].shape), list(result_dataset[1].shape), list(result_dataset[2].shape)}")
    assert list(result_dataset[0].shape) == [n_samples, max_len] and \
           list(result_dataset[1].shape) == [n_samples, max_len, D] and \
           list(result_dataset[2].shape) == [n_samples, max_len, D]

    # Save as h5 file
    home = os.getcwd()
    data_dir = home + '/data/'

    filename = os.path.join(data_dir, 'mimic-iv.h5')

    os.makedirs(data_dir, exist_ok=True)

    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('time', data=result_dataset[0])
    h5f.create_dataset('data', data=result_dataset[1])
    h5f.create_dataset('mask', data=result_dataset[2])

    h5f.close()
    print(f"Exported data to {filename}")
