#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ahmed Khorshid, Ting Fung Lam, Yony Bresler, Nausheen Fatma, Khalid Eidoo
@license: Crater Labs (C)
"""
import torch
import torch.nn as nn
from torchmetrics import functional as F

import pytorch_lightning as pl
from torchdiffeq import odeint


# Network to learn the derivative function
class FeedForwardNetwork(nn.Module):  # Function for the ODE-Solver
    def __init__(self, input_size):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 2*input_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(2*input_size, input_size)  # Same size as input for Euler loop

    def forward(self, t, x):
        l1 = self.layer1(x)
        out_relu = self.relu(l1)
        output = self.layer2(out_relu)
        return output


class ODE_Evolver(nn.Module):
    def __init__(self, input_size, solver, step_size):
        super(ODE_Evolver, self).__init__()
        self.ff = FeedForwardNetwork(input_size)
        self.solver = solver
        self.step_size = step_size
        for param in self.ff.parameters():
            param.requires_grad = True

    def forward(self, y0, t):
        if self.solver == "euler":
            pred_y = odeint(self.ff, y0, t, method=self.solver, options=dict(step_size=self.step_size))
        else:
            pred_y = odeint(self.ff, y0, t, method=self.solver, rtol=1e-3, atol=1e-4)
        return pred_y


class ODE_RNN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Running on the GPU')
            print(device)
        else:
            device = torch.device('cpu')
            print('Running on the CPU')

        self.new_device = device
        # Unpack Hyperparameters ####
        self.save_hyperparameters(hparams)
        self.experiment = self.hparams.experiment
        self.input_dim = self.hparams.input_dim
        self.output_dim = self.hparams.output_dim
        self.hidden_size = self.hparams.hidden_size
        self.drop_prob = self.hparams.drop_out
        self.batch_size = self.hparams.batch_size
        self.sequence_length = self.hparams.sequence_length
        self.solver = self.hparams.solver
        self.learning_rate = self.hparams.learning_rate
        self.step_size = self.hparams.step_size  # Only applies for Euler solver

        self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_size, batch_first=True).to(self.new_device)
        self.evolver = ODE_Evolver(input_size=self.hidden_size,
                                   solver=self.solver, step_size=self.step_size).to(self.new_device)

        self.output_ff = nn.Sequential(
            nn.Dropout(p=self.drop_prob),
            nn.Linear(in_features=self.hidden_size, out_features=2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.Linear(in_features=2*self.hidden_size, out_features=self.output_dim)
        ).to(self.new_device)

    def get_unique_x(self, train_time_batch):
        combined_tt, inverse_indices = torch.unique(torch.cat([ex for ex in train_time_batch]),
                                                    sorted=True,
                                                    return_inverse=True)
        return combined_tt, inverse_indices

    # Perform forward pass by alternating ODE evolver with RNN on the combined unique times across the batch
    def forward(self, input, train_time, final_jump):
        # Append final jump to jump array
        train_time = torch.cat((train_time, final_jump.unsqueeze(-1)), dim=1)

        combined_tt, inverse_indices = self.get_unique_x(train_time)

        batch_size = input.shape[0]
        sequence_len = train_time.shape[1]
        input_dim = input.shape[2]

        # Maps [batch, sequence position] to its location in combined_tt
        inverse_indices_batch = inverse_indices.view(-1, sequence_len)
        # For each batch entry, store the last sample that was sampled
        seq_offset = torch.zeros([batch_size, 1], dtype=torch.int32).to(self.new_device)
        # Stores the latest hidden state from the RNN portion
        rnn_output = torch.zeros([batch_size, self.hidden_size]).to(self.new_device)
        # Stores the latest hidden state from the evolver
        evolver_output = torch.zeros([1, batch_size, self.hidden_size]).to(self.new_device)
        max_seq_index = sequence_len - 1

        rnn_output_shape = rnn_output.shape[1]

        for timestep in range(0, len(combined_tt)):
            if timestep > 0:
                prev_and_cur_time = combined_tt[timestep-1:timestep+1]
            else:
                prev_and_cur_time = combined_tt[timestep]  # avoid underflow for first iteration
            prev_and_cur_time = prev_and_cur_time.view(-1, 1)

            # Apply evolver on all samples
            tmp_evolver_output = self.evolver(rnn_output.clone(), prev_and_cur_time.squeeze(1))  # do the evolver always
            evolver_output = tmp_evolver_output[-1, :, :].unsqueeze(0)  # Keep the hidden state of the second time

            # If the current time is beyond the last timestep for a given sample
            # we do not want evolver to keep progressing, overwrite result with
            # previous rnn_output
            current_offset_value = torch.gather(inverse_indices_batch, 1, seq_offset.long()).to(self.new_device)

            evolver_mask_input = (current_offset_value > timestep).repeat(1, self.hidden_size)
            if evolver_mask_input.any():
                evolver_mask_output = evolver_mask_input.unsqueeze(2).permute(2, 0, 1)

                rnn_completed_samples = rnn_output[evolver_mask_input]
                evolver_output[evolver_mask_output] = rnn_completed_samples

            # Now perform RNN, only on samples that have features at the current time
            batch_mask = (current_offset_value == timestep)  # which sequence in the batch has the same timestamp

            # Exclude last entry of each sequence from the mask (final jump has no features, can't apply RNN)
            batch_mask = batch_mask & (seq_offset != max_seq_index)

            # Check if there are any RNN entries to perform
            if not batch_mask.any():
                # If not, copy output of evolver
                rnn_output = evolver_output.squeeze(0)
            else:
                # If there are entries, mask them, apply RNN, and combine output with unchanged h

                # Use repeated masking to match the dimensions with input sequence dimensions
                mask_for_train_input_seq = batch_mask.unsqueeze(1).repeat(1, input.shape[1], input_dim)
                # Extract rows from the batch:
                subset_train_input = input[mask_for_train_input_seq].view(-1, input.shape[1], input_dim)
                # Extract columns:
                current_input = torch.gather(subset_train_input,
                                             1,
                                             seq_offset[batch_mask]
                                             .unsqueeze(1).unsqueeze(1).repeat(1, 1, input_dim).long())

                mask_for_evolver = batch_mask.unsqueeze(1).repeat(1, self.hidden_size, 1)

                # 2-step mask for evolver:
                evolver_temp = evolver_output[mask_for_evolver.permute(2, 0, 1)].view(-1, self.hidden_size)
                evolver_temp = evolver_temp.unsqueeze(1).permute(1, 0, 2)

                # Apply RNN on the selected current_input:
                out_rnn_batch, evolver_temp = self.rnn(current_input, evolver_temp)

                # Update current out_rnn output value only for the masked rows in the batch:
                mask_for_rnn_output = batch_mask.repeat(1, rnn_output_shape)
                rnn_output[mask_for_rnn_output] = out_rnn_batch.view(1, -1)

                # Update the sequence offset for the masked rows on which processing has just been done.
                max_index_tensor = torch.tensor([max_seq_index]).repeat(seq_offset[batch_mask].shape[0])

                # Don't overflow the index while updating sequence offset, increase index only till the max_seq_length
                seq_offset[batch_mask] = torch.min(seq_offset[batch_mask] + 1,
                                                   max_index_tensor.to(self.new_device).int())

        # fully connected after all the timesteps for the last timestep prediction
        output_pre = self.output_ff(rnn_output)
        output = output_pre.squeeze(0)
        assert list(output.shape) == [batch_size, self.output_dim]
        return output.squeeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def masked_loss(self, output, target, l1):
        # For MIMIC only. Calculate MSE or L1 losses only on observed data
        dim = self.output_dim
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
        trainX, trainX_jump, train_jump, trainY = batch
        output = self.forward(trainX, trainX_jump, train_jump)  # Pass the x_jumps as separate argument

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
        testX, testX_jump, test_jump, testY = batch
        output = self.forward(testX, testX_jump, test_jump)

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
        avg_r2_score = 0 if self.experiment == "mimic" else F.r2_score(output, testY)
        # Avg of r2 scores isnt the right r2 score
        self.log('val_loss', avg_loss, sync_dist=True)
        self.log('val_rmse_loss', avg_loss.sqrt(), sync_dist=True)
        self.log('val_l1_loss', avg_l1_loss, sync_dist=True)
        self.log('val_r2_score', avg_r2_score, sync_dist=True)

    def test_epoch_end(self, outputs):
        output = torch.cat([x['output'] for x in outputs], dim=0)
        testY = torch.cat([x['testY'] for x in outputs], dim=0)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_l1_loss = torch.stack([x['val_l1_loss'] for x in outputs]).mean()
        avg_r2_score = 0 if self.experiment == "mimic" else F.r2_score(output, testY)
        self.log('test_loss', avg_loss, sync_dist=True)
        self.log('test_rmse_loss', avg_loss.sqrt(), sync_dist=True)
        self.log('test_l1_loss', avg_l1_loss, sync_dist=True)
        self.log('test_r2_score', avg_r2_score, sync_dist=True)

    def MSE_loss(self, logits, labels):
        return F.mean_squared_error(logits, labels)

    def l1_loss(self, logits, labels):
        return F.mean_absolute_error(logits, labels)
