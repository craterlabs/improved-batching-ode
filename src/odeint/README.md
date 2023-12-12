# odeint model by Duvenaud et al.

This folder contains a wrapper we created of the ODE-RNN model by Duvenaud et al. using the odeint evolver from the `torchdiffeq` package  
The `train_int.py` and `model_int.py` files replicates the corresponding train and model files from the parent directory and works similarly  
Please cd into this directory `cd src/odeint` and run `train_int.py`,  
or run `schedule_int.sh` to run the experiments you want

## Reference
Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud.
Neural ordinary differential equations. In S. Bengio, H. Wallach,
H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors,
Advances in Neural Information Processing Systems, volume 31. Curran
Associates, Inc., 2018.