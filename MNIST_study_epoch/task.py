import numpy as np
import jax
import jax.numpy as jnp

import optax
import diffrax
import jaxopt

import time
import pickle
import gc

from functools import partial

class optimizer:
    '''
    This define optimizers to do the training.
    Task: get gradient
    '''
    
    def __init__(self, grad_method, nn, network_params_0):
        self.grad_method = grad_method
        self.nn = nn
        self.network_params_0 = network_params_0
        
    def optimize_step(self, network_params, g_params):
        # Single iteration step during training
        pass
    
    def train(self, input_data, target, show_process = False, dynamical_saving = False, suffix = None):
        pass

class gradient_descent(optimizer):
    
    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate):
        
        def add_func(x, gx, learning_rate):
            return x - gx * learning_rate
    
        return jax.tree_map(lambda x, gx: add_func(x, gx, learning_rate), network_params, g_params)
    
    def train(self, N_epoch, learning_rate, input_data, target, show_process = False, dynamical_save = False, suffix = None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        for k in range(0, N_epoch):
            #grad_args = input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init)
            t1 = time.time()
            running_params = self.optimize_step(running_params, params_g, learning_rate)
            t2 = time.time()
            
            paramsL.append(running_params)
            costL.append(cost)
            
            if show_process:
                print(k, "current cost = ", cost, "thermolization time:", t1-t0, "update time:", t2-t1)
                
            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)
        return costL, paramsL
        
class moment_gradient_descent(gradient_descent):
    
    @partial(jax.jit, static_argnames=['self'])
    def optimize_step(self, network_params, g_params, learning_rate, last_grad, r):
        def moment_func(gx, lgx, r): return gx + r*lgx
        def add_func(x, tot_gx, learning_rate): return x - tot_gx * learning_rate
        
        new_grad = jax.tree_map(lambda gx, lgx: moment_func(gx, lgx, r), g_params, last_grad)
        new_params = jax.tree_map(lambda x, tot_gx: add_func(x, tot_gx, learning_rate), network_params, new_grad)
        
        return new_grad, new_params
    
    
    def train(self, N_epoch, learning_rate, input_data, target, r = 0.9, show_process = False, dynamical_save = False, suffix = None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        def zero_func(x): return 0. * x
        
        last_grad = jax.tree_map(zero_func, running_params)
        
        for k in range(0, N_epoch):
            #grad_args = input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init)
            t1 = time.time()
            last_grad, running_params = self.optimize_step(running_params, params_g, learning_rate, last_grad, r)
            t2 = time.time()
            
            gc.collect()
            paramsL.append(running_params)
            costL.append(cost)
            
            if show_process:
                print(k, "current cost = ", cost, "thermolization time:", t1-t0, "update time:", t2-t1)
                
            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)
        return costL, paramsL


class layerwise_gradient_descent(gradient_descent):
    '''
    This realize SGD with layerwise learning rate. That is, for different layer the learning rate is different. 
    '''
    
    def __init__(self, grad_method, nn, network_params_0):
        super().__init__(grad_method, nn, network_params_0)
        if nn.structure_name !='layered':
            raise ValueError('The network does not have layer architecture!')
    
    def optimize_step(self, network_params, g_params, learning_rate_list):
        
        def add_func(x, gx, learning_rate):
            return x - learning_rate * gx
                
        WL, bias = network_params
        biasL = np.split(bias, self.nn.split_points, axis=1)
                
        g_WL, g_bias = g_params
        g_biasL = np.split(g_bias, self.nn.split_points, axis=1)
                
        new_WL = jax.tree_map(add_func, WL, g_WL, learning_rate_list)
        new_biasL = jax.tree_map(add_func, biasL[1:len(biasL)], g_biasL[1:len(biasL)], learning_rate_list)
        new_bias = np.concatenate((biasL[0], *new_biasL), axis=1)
        
        #jax.profiler.save_device_memory_profile("memory.txt")
        
        return new_WL, new_bias
    
class layerwise_moment_gradient_descent(moment_gradient_descent):
    def get_tot_grad(self, g_params, old_params, r):
        def g_func(g, lg, r):
            return g + r*lg
        
        return jax.tree_map(lambda g, lg: g_func(g, lg, r), g_params, old_params)
    
    def layer_wise_update(self, network_params, g_params, learning_rate_list):
        def add_func(x, gx, learning_rate):
            return x - learning_rate * gx
                
        WL, bias = network_params
        biasL = np.split(bias, self.nn.split_points, axis=1)
                
        g_WL, g_bias = g_params
        g_biasL = np.split(g_bias, self.nn.split_points, axis=1)
                
        new_WL = jax.tree_map(add_func, WL, g_WL, learning_rate_list)
        new_biasL = jax.tree_map(add_func, biasL[1:len(biasL)], g_biasL[1:len(biasL)], learning_rate_list)
        new_bias = np.concatenate((biasL[0], *new_biasL), axis=1)
        #del WL, bias, biasL, g_WL, g_bias, g_biasL, new_biasL
        #gc.collect()
        return new_WL, new_bias
    
    
    def optimize_step(self, running_params, params_g, learning_rate_list, last_grad, r):
        tot_g = self.get_tot_grad(params_g, last_grad, r)
        new_params = self.layer_wise_update(running_params, tot_g, learning_rate_list)
        
        return tot_g, new_params
    

class optax_optimize(optimizer):
    '''
    This use optax methods to train the netowrk
    '''
    def __init__(self, grad_method, nn, network_params_0, optimizer):
        '''
        Here optimizer refers to a optax optimizer, optax.adam for example. 
        '''
        super().__init__(grad_method, nn, network_params_0)
        self.optimizer = optimizer

    def train(self, N_epoch, learning_rate, input_data, target, show_process = False, dynamical_save = False, suffix = None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        
        solver = self.optimizer(learning_rate=learning_rate)
        opt_state = solver.init(running_params)
        
        for k in range(0, N_epoch):
            #grad_args = input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init)
            t1 = time.time()
            updates, opt_state = solver.update(params_g, opt_state, running_params)
            running_params = optax.apply_updates(running_params, updates)
            t2 = time.time()
            
            paramsL.append(running_params)
            costL.append(cost)
            
            if show_process:
                print(k, "current cost = ", cost, "thermolization time:", t1-t0, "update time:", t2-t1)
                
            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)
        
        return costL, paramsL
    
class self_ad_layerwise_gd(layerwise_gradient_descent):
        
    def train(self, N_epoch, learning_rate, input_data, target, r = 0.9, show_process = False, dynamical_save = False, suffix = None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        
        def norm_ratio(x, gx):
            return jnp.linalg.norm(x)/jnp.linalg.norm(gx)
            
        def list_multiply(L, x):
            a0 = L[-1]
            for k in range(0, len(L)):
                L[k] = L[k]/a0 * x
            return L

            
        for k in range(0, N_epoch):
            #grad_args = input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(input_data, target, self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init)
            t1 = time.time()
            norm_ratio_list = jax.tree_map(norm_ratio, running_params[0], params_g[0])
            learning_rate_list = list_multiply(norm_ratio_list, learning_rate)
            print(learning_rate_list)
            running_params = self.optimize_step(running_params, params_g, learning_rate_list)
            t2 = time.time()
            
            paramsL.append(running_params)
            costL.append(cost)
            
            if show_process:
                print(k, "current cost = ", cost, "thermolization time:", t1-t0, "update time:", t2-t1)
                    
            if dynamical_save:
                with open("paramsL_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(paramsL, f1)
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)
        return costL, paramsL
