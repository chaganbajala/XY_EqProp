import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import jaxopt
import time
import pickle
import gc
# Code for training
from functools import partial

class EP_grad:
    '''
    This implement equilibrium propagation training task. The training task consists of:
    nn: network to be trained 
    total_force(self, t, y, target, beta, nn, network_params): calculate the total force
    run_network(self, y0, target, beta, nn, network_params, T): run network
    '''
    
    def __init__(self, grad_params, sample_args):
        self.beta, self.runtime, self.rtol, self.atol = grad_params
        self.sample_method, self.batch_size, self.M_init = sample_args
        if self.sample_method == 'full':
            self.grad_func = self.full_gradient
        elif self.sample_method == 'mini_batch': 
            self.grad_func = self.mini_batch_gradient
        elif self.sample_method == 'random_init_mini_batch':
            self.grad_func = self.radnom_init_mini_batch_gradient
        
    
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def total_force(self, t, y, target, beta, nn, network_params):
        return jax.vmap(nn.internal_force, (0, None))(y, network_params) + beta*jax.vmap(nn.external_force, (0, 0, None))(y, target, network_params)
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Solve the equation with diffrax
        # Set parameter for diffrax
        
        rtol = self.rtol
        atol = self.atol
        t_span = [0, self.runtime]
        
        '''
        func1 = jax.jit(lambda y: nn.internal_force(y, network_params))
        func2 = jax.jit(lambda y, target: nn.external_force(y, target, network_params))
        
        @jax.jit
        def total_force(t, y, target, beta):
            return jax.vmap(func1, (0))(y) + beta * jax.vmap(func2, (0, 0))(y, target)
        
        
        #saveat = np.linspace(*t_span, 10).tolist()
        odefunc = lambda t, y0, args: total_force(t, y0, target, beta)
        eqs = diffrax.ODETerm(odefunc)
        '''
        
        odefunc = lambda t, y0, args: self.total_force(t, y0, target, beta, nn, network_params)
        eqs = diffrax.ODETerm(odefunc)
        
        # Use 4th Runge Kutta
        solver = diffrax.Tsit5()
        
        #Use 5th Kvaerno for stiff case
        #solver = diffrax.Kvaerno4()
        
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        #t = diffrax.SaveAt(ts=saveat)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0 = None, y0=y0,
                                stepsize_controller=stepsize_controller, max_steps=10000000)
        
        L = len(solution.ts)
        
        y = solution.ys[L-1,:]
        del solution

        return y
    
    def devided_by_beta(self, x):
        return x/self.beta
    
    def get_params_gradient(self, y0, target, nn, network_params):
        # Get free equilibrium
        #run_func = jax.jit(lambda y0, target, beta: self.run_network(y0, target, nn, network_params, beta))
        #free_equi = run_func(y0, target, 0)
        #nudge_equi = run_func(free_equi, target, beta)
        N_data = y0.shape[0]
        N_params = sum(jax.tree_util.tree_leaves(jax.tree_map(jnp.size, network_params)))
        
        mean_fun = lambda x: jnp.mean(x, axis=0)
        sum_func = lambda x: jnp.sum(x, axis=0)
        mean_divide_func = lambda x: jnp.divide(x, N_data)
        zero_func = lambda x: jnp.multiply(x, 0)
        
        t0 = time.time()
        free_equi = self.thermalize_network(y0, target, nn, network_params, 0.)
        t1 = time.time()
        
        nudge_equi = self.thermalize_network(free_equi, target, nn, network_params, self.beta)
        '''
        memory_limit = 2.5 * 1e8
        
        if N_data*N_params < memory_limit:
            dEdParams_array_free = jax.vmap(nn.params_derivative, (0,None))(free_equi, network_params)
            dEdParams_array_nudge = jax.vmap(nn.params_derivative, (0,None))(nudge_equi, network_params)

            mean_dEdParams_free = jax.tree_map(mean_fun, dEdParams_array_free)
            mean_dEdParams_nudge = jax.tree_map(mean_fun, dEdParams_array_nudge)
            
            del dEdParams_array_free, dEdParams_array_nudge

        elif N_params < memory_limit: 
            group_size = int(memory_limit/N_params)
            group_num = int(N_data/group_size)+1
            dEdParams_free = jax.tree_map(zero_func, network_params)
            dEdParams_nudge = jax.tree_map(zero_func, network_params)
            for k in range(0, group_num):
                dEdParams_array_free = jax.vmap(nn.params_derivative, (0,None))(free_equi[k*group_size:min((k+1)*group_size, N_data), :], network_params)
                dEdParams_array_nudge = jax.vmap(nn.params_derivative, (0,None))(nudge_equi[k*group_size:min((k+1)*group_size, N_data), :], network_params)
                dEdParams_free = jax.tree_map(jnp.add, dEdParams_free, jax.tree_map(sum_func, (dEdParams_array_free)))
                dEdParams_nudge = jax.tree_map(jnp.add, dEdParams_nudge, jax.tree_map(sum_func, (dEdParams_array_nudge)))

            mean_dEdParams_free = jax.tree_map(mean_divide_func, dEdParams_free)
            mean_dEdParams_nudge = jax.tree_map(mean_divide_func, dEdParams_nudge)
            del dEdParams_array_free, dEdParams_array_nudge
        else:
            raise Exception("Network size too large")
        '''
        
        dEdParams_free = nn.params_derivative(free_equi[0,:], network_params)
        dEdParams_nudge = nn.params_derivative(nudge_equi[0,:], network_params)

        for k in range(1, N_data):
            dEdParams_free = jax.tree_map(jnp.add, dEdParams_free, nn.params_derivative(free_equi[k,:], network_params))
            dEdParams_nudge = jax.tree_map(jnp.add, dEdParams_nudge, nn.params_derivative(nudge_equi[k,:], network_params))
        
        mean_func = lambda x: jnp.divide(x, N_data)
        mean_dEdParams_free = jax.tree_map(mean_func, dEdParams_free)
        mean_dEdParams_nudge = jax.tree_map(mean_func, dEdParams_nudge)
        
        gradient = jax.tree_map(jnp.subtract, mean_dEdParams_nudge, mean_dEdParams_free)
        
        cost = jax.vmap(nn.distance_function, (0, 0, None))(free_equi, target, network_params)
        del free_equi, nudge_equi
        #n = gc.collect()
        #jax.clear_backends()
        #jax.clear_caches()
        
        return jnp.mean(cost), jax.tree_map(self.devided_by_beta, gradient)
    
    def full_gradient(self, input_data, target, nn, network_params, *args):

        y0 = nn.get_initial_state(input_data)
        cost, params_g = self.get_params_gradient(y0, target, nn, network_params)
        del y0
        return cost, params_g
    
    def mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, *args):

        y0, running_target = nn.get_initial_state_mini_batch(input_data, target, batch_size)
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g
    
    
    def radnom_init_mini_batch_gradient(self, input_data, target, nn, network_params, batch_size, M_init):

        y0, running_target = nn.get_multiple_init_initial_state(input_data, target, batch_size, M_init)            
        cost, params_g = self.get_params_gradient(y0, running_target, nn, network_params)
        del y0
        return cost, params_g
    
    def load_training_data(self, params_nf, costL_nf):
        with open(params_nf, 'rb') as f1:
            paramsL = pickle.load(f1)
        with open(costL_nf, 'rb') as f2:
            costL = pickle.load(f2)
            
        return costL, paramsL
        
    

class EP_grad_vmap(EP_grad):
    '''
    This implement equilibrium propagation training task which deals with data with vmap. 
    '''
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def total_force(self, t, y, target, beta, nn, network_params):
        # Calculate total force for single piece of data
        return nn.internal_force(y, network_params) + beta * nn.external_force(y, target, network_params)
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def single_run_func(self, y0, target, nn, network_params, beta):
        # Find equilibrium for single piece of data
        
        t_span = [0, self.runtime]
        
        odefunc = lambda t, y0, args: self.total_force(t, y0, target, beta, nn, network_params)
        eqs = diffrax.ODETerm(odefunc)
        
        # Use 4th Runge Kutta
        solver = diffrax.Tsit5()
        
        #Use 5th Kvaerno for stiff case
        #solver = diffrax.Kvaerno4()
        
        stepsize_controller = diffrax.PIDController(rtol = self.rtol, atol = self.atol)
        #t = diffrax.SaveAt(ts=saveat)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0 = None, y0=y0,
                                stepsize_controller=stepsize_controller, max_steps=10000000)
        
        L = len(solution.ts)
        
        y = solution.ys[L-1,:]
        del solution

        return y
        
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Tree_map single_run_func with multiple data and target
        run_func = lambda y0, target: self.single_run_func(y0, target, nn, network_params, beta)
        y = jax.vmap(run_func, (0,0))(y0, target)

        return y
    
class EP_grad_tree(EP_grad):
    '''
    This implement equilibrium propagation training task which deals with data with tree strategy.
    The data are dealt one by one in a tree_mapped single_run_func 
    '''
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def total_force(self, t, y, target, beta, nn, network_params):
        # Calculate total force for single piece of data
        return nn.internal_force(y, network_params) + beta * nn.external_force(y, target, network_params)
    
    #@partial(jax.jit, static_argnames=['self', 'nn'])
    def single_run_func(self, y0, target, nn, network_params, beta):
        # Find equilibrium for single piece of data
        
        t_span = [0, self.runtime]
        
        odefunc = lambda t, y0, args: self.total_force(t, y0, target, beta, nn, network_params)
        eqs = diffrax.ODETerm(odefunc)
        
        # Use 4th Runge Kutta
        solver = diffrax.Tsit5()
        
        #Use 5th Kvaerno for stiff case
        #solver = diffrax.Kvaerno4()
        
        stepsize_controller = diffrax.PIDController(rtol = self.rtol, atol = self.atol)
        #t = diffrax.SaveAt(ts=saveat)

        # Solve the ODE
        solution = diffrax.diffeqsolve(eqs, solver, t0=t_span[0], t1=t_span[1], dt0 = None, y0=y0,
                                stepsize_controller=stepsize_controller, max_steps=10000000)
        
        L = len(solution.ts)
        
        y = solution.ys[L-1,:]
        del solution

        return y
        
    #@partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Tree_map single_run_func with multiple data and target
        run_func = lambda y0, target: self.single_run_func(y0, target, nn, network_params, beta)
        y = jax.tree_map(run_func, list(y0), list(target))

        return jnp.asarray(y)
    
    
class EP_grad_loop(EP_grad_tree):
    '''
    This implement equilibrium propagation training task which deals with data with jax.lax.fori_loop strategy.
    The data are dealt one by one in a tree_mapped single_run_func 
    '''
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Tree_map single_run_func with multiple data and target
            
        run_func = lambda y0, target: self.single_run_func(y0, target, nn, network_params, beta)
        y = np.zeros(y0.shape)
        

        
        def loop_func(k, vals):
            y0, target, y = vals
            y = y.at[k,:].set(run_func(y0[k], target[k]))
            
            return y0, target, y
        
        vals_0 = y0, target, y
        
        vals= jax.lax.fori_loop(0, y0.shape[0], loop_func, vals_0)        
        
        '''
        for k in range(0, y0.shape[0]):
            y[k] = self.single_run_func(y0[k], target[k], nn, network_params, beta)
        
        '''

        return vals[-1]
    
class EP_grad_pyloop(EP_grad_tree):
    '''
    This implement equilibrium propagation training task which deals with data with python loop
    The data are dealt one by one in a tree_mapped single_run_func 
    '''
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Tree_map single_run_func with multiple data and target
            
        run_func = lambda y0, target: self.single_run_func(y0, target, nn, network_params, beta)
        y = jnp.zeros(y0.shape)    
        
        for k in range(0, y0.shape[0]):
            y = y.at[k,:].set(run_func(y0[k], target[k], nn, network_params, beta))

        return y
    
class opt_EP_training(EP_grad):
    '''
    This implement a EP training task using optimizer from jaxopt to search for the equilibrium.
    '''
    
    def __init__(self, training_params):
        self.N_epoch, self.beta, self.learning_rate, self.runtime, self.rtol, self.atol, self.optimizer = training_params

    def total_energy(self, y, target, beta, nn, network_params):
        return nn.internal_energy(y, network_params) + beta * nn.external_energy(y, target, network_params)
    
    def total_force(self, y, target, beta, nn, network_params):
        # Calculate total force for single piece of data
        return nn.internal_force(y, network_params) + beta * nn.external_force(y, target, network_params)   
    
    #@partial(jax.jit, static_argnames=['self', 'nn'])
    def energy_and_grad(self, y, target, beta, nn, network_params):
        return self.total_energy(y, target, beta, nn, network_params), -self.total_force(y, target, beta, nn, network_params)
    
    #@partial(jax.jit, static_argnames=['self', 'nn'])
    def single_run_func(self, y0, target, nn, network_params, beta):
        # Find equilibrium for single piece of data
        optfunc = lambda y: self.energy_and_grad(y, target, beta, nn, network_params)
        
        solver = self.optimizer(optfunc, value_and_grad=True, maxiter=10000, tol=1e-6, maxls=1000, max_stepsize=0.1)
        #solver = jaxopt.GradientDescent(optfunc, value_and_grad=True, maxiter=10000, tol=1e-6)

        #args = input_data, WL, bias, target, structure_shape, beta

        res = solver.run(y0)

        y = res.params
        del res

        return y   
    
    @partial(jax.jit, static_argnames=['self', 'nn'])
    def thermalize_network(self, y0, target, nn, network_params, beta):
        # Tree_map single_run_func with multiple data and target
            
        run_func = lambda y0, target: self.single_run_func(y0, target, nn, network_params, beta)
        
        y = jax.vmap(run_func, (0,0))(y0, target)

        return y
    