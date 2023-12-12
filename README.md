# Crater Labs ODE-RNN

Repository implementing ‘Improved Batching Strategy For Irregular Time-Series ODE’ https://arxiv.org/abs/2207.05708

This includes the implementation of our models, dataloaders, as well as a wrapper for ODE-RNN by Rubanova et al. (referred to as odeint).


## Installation

First, install Pytorch and its related packages, and then install `requirements.txt`.
```bash
pip install --upgrade pip wheel
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/torch/ -f https://download.pytorch.org/whl/torchvision -f https://download.pytorch.org/whl/torchaudio/
pip3 install -r requirements.txt
```

## Requirements

* Python 3.6 - 3.9 and CUDA 11.3
* Nvidia NGC PyTorch container 21.10 (nvcr.io/nvidia/pytorch:21.10-py3)
* MIMIC experiments requires minimum of 163 GB of CPU memory

## Running
* Ensure you are in the source folder (`src/` for our model, `src/odeint` for odeint)
* To run a single experiment, use `train.py` for our model, specifying which experiment you want to run with `--experiment [syn|mujoco|mimic]` flag  
* To run multiple experiments recreating the paper, run `src/schedule.sh` for our model or `src/odeint/schedule_int.sh` scripts for odeint.
* The test result after training will be recorded in the `.out` output file within the `lightning_logs` directory  


## Contents
All files with `_int` suffix are for Duvenaud's ODE-RNN model, otherwise are for our own model.

* `train.py` and `odeint/train_int.py` are execution files for our model and odeint respectively.
* `model.py` and `odeint/model_int.py` are the model files.
* `schedule.sh` and `odeint/schedule_int.sh` are scripts to run sample experiments for our model and odeint respectively.
* `dataloader_mujoco.py`, `dataloader_phy.py` and `dataloader_syn.py` are the dataloaders for Mujoco, MIMIC, and synthetic data respectively used to train the models.
* `data` folder contains the data for MuJoCo experiments.
* `params` folder contains the default hyperparameters for the experiments. 

## Data processing for MIMIC-IV
* Request access to MIMIC-IV data and download MIMIC-IV v1.0 files  
https://physionet.org/content/mimiciv/1.0/

* Following the procedure from the repo of the paper "Neural flows: Efficient alternative to neural ODEs" by Biloˇs et al. (<https://arxiv.org/abs/2110.13040>), run all the notebooks except `mimic_prep.ipynb` from the following repo. Finally run `datamerging.ipynb` to yield `full_dataset.csv`  
https://github.com/mbilos/neural-flows-experiments/tree/master/nfe/experiments/gru_ode_bayes/data_preproc

* Modify `get_data.py` at line 64 and insert a line to save `full_data` as a csv, then run it to obtain the csv file  
https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/lib/get_data.py
```
full_data.to_csv('data/MIMIC-IV-intercepted.csv')
```
* Run the python script `src/MIMIC-IV_processing.py` from this repo using the csv file obtained above

Reference:

* Marin Biloˇs, Johanna Sommer, Syama Sundar Rangapuram, Tim Januschowski, and Stephan G¨unnemann. Neural flows: Efficient alternative to neural ODEs. Advances in Neural Information Processing Systems, 2021.
* Yulia Rubanova, Ricky T. Q. Chen, and David K Duvenaud. Latent ordinary differential equations for irregularly-sampled time series.
