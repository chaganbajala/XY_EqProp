# XY_EqProp
## Paper:
  - Code used for paper: https://arxiv.org/abs/2402.08579
  - The background and the algorithm is described in https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full
  - This project run EqProp on XY model
  - This part of code is un-encapsulated, function-oriented. JAX is used for acceleartion on GPU.
  Classes of networks are used to generate initial parameters and initial states of specific topology. 
  - Functions of:
    - sp_func: functions construct with SciPy.
    - _func: functions for network with all-to-all activity.
    - layer_func/layered_func: functions for networks with layer architecture.
    - lattice_func: functions used for lattice
   
## MNIST_study_spoch:
  -  Encapsulated and reconstructed code. Easier to read, understand and extend. 
  -  Run with full size MNIST
  -  Run "python -u main.py". One may need to change the parameters and directory in main.py. 
