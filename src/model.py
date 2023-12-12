#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Khalid Eidoo
@license: Crater Labs (C)
"""
import torch
import torch.nn as nn
from torchmetrics import functional as F

import pytorch_lightning as pl
import matplotlib.pyplot as plt


class Evolver(nn.Module):
    # Hyperparameters are num_steps_adaptive for 'adaptive_fixed' mode and growth_factor for 'adaptive_geometric' mode
    def __init__(self, input_size, step_size, device, dropout, mode='fixed_dt', num_steps_adaptive=5,
                 dynamic_step_growth_factor=1.5):
        """
        input_size: Number of dimensions of the input data
        step_size: The delta t's for each step for fixed_dt and adaptive_geometric modes
        device: Where to store the data, weights etc.
        dropout: Dropout probability for the dropout layers
        mode: Which of the three modes according to the paper. Choices: fixed_dt, adaptive_fixed, adaptive_geometric
        num_steps_adaptive: How many total ODE steps for the adaptive_fixed mode
        dynamic_step_growth_factor: The growth factor r for adaptive_geometric mode according to the paper
        """
        super().__init__()

        self.device = device
        self.step_size = step_size
        self.mode = mode
        self.dropout = dropout
        if mode == 'adaptive_fixed':
            print(f"Mode: {mode}, num_steps_adaptive: {num_steps_adaptive}")
            self.num_steps_adaptive = num_steps_adaptive
        elif mode == 'adaptive_geometric':
            print(f"Mode: {mode}, step_size: {step_size}, dynamic_step_growth_factor: {dynamic_step_growth_factor}")
            self.dynamic_step_growth_factor = dynamic_step_growth_factor
            self.dynamic_step_growth_factor_log = torch.log(torch.tensor(dynamic_step_growth_factor).to(device=device))
        else:
            print(f"Mode: {mode}, step_size: {step_size}")

        if self.dropout > 0:
            self.ff = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=input_size, out_features=2*input_size),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features=2*input_size, out_features=input_size)
            ).to(self.device)
        else:
            self.ff = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=2*input_size),
                nn.ReLU(),
                nn.Linear(in_features=2*input_size, out_features=input_size)
            ).to(self.device)

    def forward(self, input, jumps):
        # input ~ [batch_size, 1, input_size]

        hidden_size = input.shape[2]
        h = input

        # Fixed number of steps N mode. Refer to "Algorithm 4: Evolver module: adaptive fixed mode" of the paper
        if self.mode == 'adaptive_fixed':

            num_steps_adaptive = self.num_steps_adaptive
            # step size varies across mini-batch
            adaptive_time_step = (jumps / float(num_steps_adaptive)).view(-1, 1, 1)

            for i in range(0, num_steps_adaptive):
                # The mask is used to prevent updating any hidden state once its respective delta_t_i is reached
                mask = (i * adaptive_time_step < jumps.view(-1, 1, 1))
                a = self.ff(h)  # FC Learning dh/dt
                h = h + (a * mask * adaptive_time_step)

        # Geometric sequence mode. Refer to "Algorithm 5: Evolver module: adaptive geometric mode" of the paper
        elif self.mode == 'adaptive_geometric':
            growth_factor = self.dynamic_step_growth_factor

            # Since self.dynamic_step_growth_factor_log and "jumps" ended up on different cuda devices
            # Need to ensure self.dynamic_step_growth_factor_log and "jumps" are on the same device
            # Find geometric series steps needed for largest value
            num_iter = torch.log((growth_factor - 1) * jumps / self.step_size + 1) / \
                self.dynamic_step_growth_factor_log.to(device=jumps.device)

            max_iter = torch.ceil(torch.max(num_iter)).to(torch.int)

            # Keeps track of the remaining time difference, which is t - t_cur from the algorithm
            delta_t_remaining = jumps.view(-1, 1, 1)

            # Dynamic step size for all the steps, which is s from the algorithm
            cur_delta_t_counter = torch.zeros_like(h[0:1, 0, 0])
            cur_delta_t_counter[0] = self.step_size

            for i in range(max_iter):
                # Prevent overshooting target time if s is too large
                cur_delta = torch.min(cur_delta_t_counter, delta_t_remaining)

                a = self.ff(h)  # FC Learning dh/dt
                h = h + a * cur_delta

                delta_t_remaining = delta_t_remaining - cur_delta

                cur_delta_t_counter = cur_delta_t_counter * growth_factor  # Grow s by factor r

        # Default current mode, fixed delta t's 'fixed_dt'.
        # Refer to "Algorithm 3: Evolver module: Fixed dt mode" of the paper
        else:
            num_steps = (jumps/self.step_size).long()  # number of steps varies across mini-batch
            max_steps = num_steps.max().item()

            for i in range(1, max_steps+1):
                # Prevent overshooting target time if s is too large
                step_size_tensor = torch.tensor(self.step_size, device=self.device).repeat(jumps.shape)
                cur_step_size = torch.min(step_size_tensor, jumps - (i - 1) * self.step_size)

                # The mask is used to prevent updating any hidden state once its respective delta_t_i is reached
                mask = (i <= num_steps).float()
                mask = mask.reshape(-1, 1, 1).repeat(1, 1, hidden_size)
                cur_step_size = cur_step_size.reshape(-1, 1, 1).repeat(1, 1, hidden_size)
                a = self.ff(h)  # FC Learning dh/dt
                a = a * mask * cur_step_size
                h = h + a

        return h


class ODE_RNN(nn.Module):
    # Implements the for loop of "Algorithm 2: Our batch efficient ODE-RNN" of the paper
    def __init__(self, input_size, hidden_size, step_size, device, mode, bypass_evolver, dropout):
        """
        input_size: Number of dimensions of the input data
        step_size: The delta t's for each step for fixed_dt and adaptive_geometric modes
        device: Where to store the data, weights etc.
        dropout: Dropout probability for the dropout layers
        mode: Which of the three modes according to the paper. Choices: fixed_dt, adaptive_fixed, adaptive_geometric
        bypass_evolver: If True, bypasses the ODE evolver, resulting in a simple RNN
        """
        super().__init__()

        self.device = device
        self.step_size = step_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.bypass_evolver = bypass_evolver
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True).to(self.device)
        self.evolver = Evolver(input_size=hidden_size, step_size=step_size, device=self.device,
                               mode=self.mode, dropout=self.dropout).to(self.device)

    def forward(self, input, jumps, initial_hidden_state=None):
        # input = [batch_size, sequence_length, input_size]
        # jumps = [batch_size, sequence_length]

        batch_size = input.shape[0]
        sequence_length = input.shape[1]

        if initial_hidden_state is not None:
            h = initial_hidden_state
        else:
            h = torch.zeros(1, batch_size, self.hidden_size).to(self.device)

        output = torch.zeros(batch_size, sequence_length, self.hidden_size).to(self.device)
        for i in range(sequence_length):
            x = input[:, i:i+1, :]
            if not self.bypass_evolver:
                j = jumps[:, i]

                h = h.permute(1, 0, 2)
                h = self.evolver(h, j)  # Refer to Evolver Algorithm 3/4/5
                h = h.permute(1, 0, 2)
            y, h = self.rnn(x, h)
            output[:, i:i+1, :] = y
        return output, h


class Model(pl.LightningModule):
    # Implements the entire "Algorithm 2: Our batch efficient ODE-RNN" of the paper
    def __init__(self, hparams):
        super().__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Running on the GPU')
        else:
            device = torch.device('cpu')
            print('Running on the CPU')
        print(device)

        self.new_device = device
        # Unpack Hyperparameters ####
        self.save_hyperparameters(hparams)
        self.experiment = self.hparams.experiment
        self.input_size = self.hparams.input_size
        self.hidden_size = self.hparams.hidden_size
        self.output_size = self.hparams.output_size
        self.dropout = self.hparams.dropout
        self.data_set = self.hparams.data_set
        self.batch_size = self.hparams.batch_size
        self.sequence_length = self.hparams.sequence_length
        self.step_size = self.hparams.step_size
        self.mode = self.hparams.mode
        self.learning_rate = self.hparams.learning_rate
        self.bypass_evolver = self.hparams.bypass_evolver if 'bypass_evolver' in hparams else False

        # Check if `mode` is a valid mode. Otherwise revert it to 'fixed_dt'
        if self.mode != 'adaptive_geometric' and self.mode != 'adaptive_fixed' and self.mode != 'fixed_dt':
            print('Unknown mode! Reverting to default mode (fixed_dt)')
            self.mode = 'fixed_dt'

        self.embedding = nn.Linear(in_features=self.input_size, out_features=self.hidden_size).to(self.new_device)

        self.ode_rnn = ODE_RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                               device=self.new_device, step_size=self.step_size, mode=self.mode,
                               bypass_evolver=self.bypass_evolver, dropout=self.dropout).to(self.new_device)

        self.output_evolver = Evolver(input_size=self.hidden_size, step_size=self.step_size,
                                      device=self.new_device, mode=self.mode, dropout=self.dropout).to(self.new_device)

        self.ff = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=2 * self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=2 * self.hidden_size, out_features=self.output_size)
        ).to(self.new_device)

    def forward(self, input, input_jump, final_jump):
        # input = [batch_size, sequence_length, input_size]
        # jumps = [batch_size, sequence_length]
        # final_jump = [batch_size], which is {Δt_i} where i=N, and N is the number of steps
        batch_size = input.shape[0]
        assert list(input.shape) == [batch_size, self.sequence_length, self.input_size]
        assert list(input_jump.shape) == [batch_size, self.sequence_length]
        assert list(final_jump.shape) == [batch_size]

        jumps = input_jump

        embedded_input = self.embedding(input)

        # Perform main ODE-RNN evolution
        rnn_output, _ = self.ode_rnn(embedded_input, jumps)
        x = rnn_output[:, -1:, :]

        # Perform final-jump before making prediction
        if not self.bypass_evolver:
            h = self.output_evolver(x, final_jump)
            y = self.ff(h)  # OutputNN
        else:
            y = self.ff(x)  # OutputNN
        output = y.squeeze()
        assert list(output.shape) == ([batch_size, self.output_size] if self.output_size > 1 else [batch_size])
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def masked_loss(self, output, target, l1):
        # For MIMIC only. Calculate MSE or L1 losses only on observed data
        dim = self.output_size
        mask = target[:, dim:]  # separate the mask from target
        target = target[:, :dim]
        assert target.shape == output.shape
        if l1:
            loss = nn.functional.l1_loss(output, target, reduction='none')
        else:
            loss = nn.functional.mse_loss(output, target, reduction='none')
        loss = (loss * mask.float()).sum()
        loss_val = loss / mask.sum()
        return loss_val

    def training_step(self, batch, batch_idx):
        trainX, trainX_jump, final_jump, trainY = batch
        output = self.forward(trainX, trainX_jump, final_jump)  # Pass the x_jumps as separate argument

        if self.experiment == "mimic":
            loss = self.masked_loss(output, trainY, l1=False)
            l1_loss = self.masked_loss(output, trainY, l1=True)
        else:
            loss = self.MSE_loss(output, trainY)
            l1_loss = self.l1_loss(output, trainY)
        self.log('train_loss', loss, sync_dist=True, on_epoch=True, on_step=False, prog_bar=True)
        self.log('train_l1_loss', l1_loss, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        testX, testX_jump, final_jump, testY = batch
        output = self.forward(testX, testX_jump, final_jump)

        if self.experiment == "mimic":
            loss = self.masked_loss(output, testY, l1=False)
            l1_loss = self.masked_loss(output, testY, l1=True)
        else:
            loss = self.MSE_loss(output, testY)
            l1_loss = self.l1_loss(output, testY)
        return {'val_loss': loss, 'val_l1_loss': l1_loss, 'output': output, 'testY': testY}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)  # Just do the same as validation step

    def validation_epoch_end(self, outputs):
        output = torch.cat([x['output'] for x in outputs], dim=0)
        testY = torch.cat([x['testY'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_l1_loss = torch.stack([x['val_l1_loss'] for x in outputs]).mean()
        avg_r2_score = 0.0 if self.experiment == "mimic" else F.r2_score(output, testY)
        # Avg of r2 scores isnt the right r2 score
        self.log('val_loss', avg_loss, sync_dist=True)
        self.log('val_rmse_loss', avg_loss.sqrt(), sync_dist=True)
        self.log('val_l1_loss', avg_l1_loss, sync_dist=True)
        self.log('val_r2_score', avg_r2_score, sync_dist=True)

        # Add predicted vs. truth plot to Tensorboard, but only if self.output_size is 1 (data)
        if self.output_size == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(output.cpu().numpy(), testY.cpu().numpy())
            ax.plot([0, 5], [0, 5])
            self.logger.experiment.add_figure('test', fig, self.current_epoch)

    def test_epoch_end(self, outputs):
        output = torch.cat([x['output'] for x in outputs], dim=0)
        testY = torch.cat([x['testY'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_l1_loss = torch.stack([x['val_l1_loss'] for x in outputs]).mean()
        avg_r2_score = 0.0 if self.experiment == "mimic" else F.r2_score(output, testY)
        self.log('test_loss', avg_loss, sync_dist=True)
        self.log('test_rmse_loss', avg_loss.sqrt(), sync_dist=True)
        self.log('test_l1_loss', avg_l1_loss, sync_dist=True)
        self.log('test_r2_score', avg_r2_score, sync_dist=True)

    def MSE_loss(self, logits, labels):
        return F.mean_squared_error(logits, labels)

    def l1_loss(self, logits, labels):
        return F.mean_absolute_error(logits, labels)
