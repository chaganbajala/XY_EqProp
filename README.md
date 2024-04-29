# XY_EqProp
## Paper:
  - Code used for paper: https://arxiv.org/abs/2402.08579
  - This part of code is un-encapsulated, function-oriented. JAX is used for acceleartion on GPU.
  Classes of networks are used to generate initial parameters and initial states of specific topology. 
  - Functions of:
    - sp_func: functions construct with SciPy.
    - _func: functions for network with all-to-all activity.
    - layer_func/layered_func: functions for networks with layer architecture.
    - lattice_func: functions used for lattice
