#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import optax
import diffrax
import jaxopt

import time
import pickle

import model as model
import method as method
import task

import gc

!python -u set_mnist_data.py

class mnist_epoch_task(task.moment_gradient_descent):
    
    def separate_data(self, rng, data, target, batch_size):
        N_data = data.shape[0]
        N_set = N_data // batch_size
        perm_ind = jax.random.permutation(rng, N_data)
        
        data_shape = N_set, batch_size, 28*28
        target_shape = N_set, batch_size, 10
        data_batch = data[perm_ind][0 : N_set * batch_size].reshape(data_shape)
        target_batch = target[perm_ind][0 : N_set * batch_size].reshape(target_shape)
        
        return data_batch, target_batch
        
    
    def train_epoch(self, learning_rate, batch_size, input_data, target, costL, running_params, last_grad, r = 0.9, show_process = False, dynamical_save = False, suffix = None):
        rng = jax.random.key(0)
        data_batch, target_batch = self.separate_data(rng, input_data, target, batch_size)
        N_set = data_batch.shape[0]
        
        for k in range(0, N_set):
            t0 = time.time()
            cost, params_g = self.grad_method.grad_func(data_batch[k,...], target_batch[k,...], self.nn, running_params, self.grad_method.batch_size, self.grad_method.M_init)
            t1 = time.time()
            last_grad, running_params = self.optimize_step(running_params, params_g, learning_rate, last_grad, r)
            t2 = time.time()
            
            costL.append(cost)
            n = gc.collect()
            
            if show_process:
                print(k, "current cost = ", cost, "thermolization time:", t1-t0, "update time:", t2-t1, "fragment:", n)
                
            if dynamical_save:
                with open("costL_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(costL, f2)
    
        return running_params, costL
    
    def train(self, N_epoch, learning_rate, batch_size, input_data, target, r = 0.9, show_process = False, dynamical_save = False, suffix = None):
        running_params = self.network_params_0
        paramsL = [running_params]
        costL = []
        
        test_y, inference_res, performance = self.test_network(running_params)
        
        test_y_L = [test_y]
        inference_res_L = [inference_res]
        performance_L = [performance]
        
        print("Start training. Test performance = {0}".format(performance))
        
        def zero_func(x): return 0. * x
        
        last_grad = jax.tree_map(zero_func, running_params)
        
        for k in range(0, N_epoch):
            t0 = time.time()
            running_params, costL = self.train_epoch(learning_rate, batch_size, input_data, target, costL, running_params, last_grad, r, show_process, dynamical_save, suffix)
            t1 = time.time()
            
            test_y, inference_res, performance = self.test_network(running_params)
        
            test_y_L.append(test_y)
            inference_res_L.append(inference_res)
            performance_L.append(performance)
            paramsL.append(running_params)
            
            if dynamical_save:
                with open("test_y_L_{0}".format(suffix), 'wb') as f1:
                    pickle.dump(test_y_L, f1)
                with open("inference_res_L_{0}".format(suffix), 'wb') as f2:
                    pickle.dump(inference_res_L, f2)
                with open("performance_L_{0}".format(suffix), 'wb') as f3:
                    pickle.dump(performance_L, f3)
                with open("paramsL_{0}".format(suffix), 'wb') as f4:
                    pickle.dump(paramsL, f4)
            
            print("The {0}th epoch done. Time consumution = {1}. Test performance = {2}".format(k, t1-t0, performance))
            
        return costL, paramsL
    
    def test_network(self, network_params):
        N = self.nn.network_structure[0]
        all_test_data = np.concatenate(sorted_test_data)
        all_test_target = np.concatenate(sorted_test_target)

        ly0 = self.nn.get_initial_state(all_test_data).reshape(10, 1000, N)
        ltest_target = all_test_target.reshape(10, 1000, 10)

        y_out = []
        for k in range(0, 10):
            y = self.grad_method.thermalize_network(ly0[k,...], ltest_target[k,...], self.nn, network_params, 0.)
            y_out.append(np.asarray(y))
        y = np.concatenate(y_out)

        res = jnp.argmax(jnp.sin(y[:,output_index]), axis=1)
        correct_res = jnp.argmax(jnp.sin(all_test_target), axis=1)

        inference_res = res==correct_res
        performance = np.mean(inference_res)

        return y, inference_res, performance



class layer_network_for_mnist(model.layered_general_XY_network):
    
    def get_random_index(self, N_data, batch_size):
        ind_list = []
        while len(ind_list) < batch_size:
            running_ind = np.random.randint(0, N_data)
            if running_ind not in ind_list:
                ind_list.append(running_ind)
        
        return np.asarray(ind_list)
    
    def get_multiple_init_initial_state(self, input_data, target, batch_size, M_init):
        # The input data here should be the sorted data. For each sort the data size should be the same
        sort_num = len(input_data)
        
        batch_input = []
        batch_target = []
        for k in range(0, sort_num):
            sort_input, sort_target = self.get_multiple_init_data(input_data[k], target[k], M_init, batch_size)
            batch_input.append(sort_input)
            batch_target.append(sort_target)
        
        batch_input = jnp.concatenate(batch_input)
        batch_target = jnp.concatenate(batch_target)
        y0 = self.get_initial_state(batch_input)
        
        return y0, batch_target
    
class general_network_for_mnist(model.general_XY_network):
    
    def __init__(self, network_structure, coup_func, bias_func, cost_func, network_type='general XY', structure_name='all to all', structure_mask=1.):
        super().__init__(network_structure, coup_func, bias_func, cost_func, network_type, structure_name)
        self.structure_mask = structure_mask
    
    def get_random_index(self, N_data, batch_size):
        ind_list = []
        while len(ind_list) < batch_size:
            running_ind = np.random.randint(0, N_data)
            if running_ind not in ind_list:
                ind_list.append(running_ind)
        return np.asarray(ind_list)
    
    def get_multiple_init_initial_state(self, input_data, target, batch_size, M_init):
        # The input data here should be the sorted data. For each sort the data size should be the same
        sort_num = len(input_data)
        
        batch_input = []
        batch_target = []
        for k in range(0, sort_num):
            sort_input, sort_target = self.get_multiple_init_data(input_data[k], target[k], M_init, batch_size)
            batch_input.append(sort_input)
            batch_target.append(sort_target)
        
        batch_input = jnp.concatenate(batch_input)
        batch_target = jnp.concatenate(batch_target)
        y0 = self.get_initial_state(batch_input)
        
        return y0, batch_target

    def params_derivative(self, y, network_params):
        
        W, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))
        v_dbias_func = jax.vmap(self.d1bias_func, (0, 0))
        v_bias_func = jax.vmap(self.bias_func, (0, 0))
        
        # calculate dE/dW
        g_W = m_coup_func(my, jnp.transpose(my)) * self.structure_mask
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([v_bias_func(y, bias[1]), bias[0]*v_dbias_func(y, bias[1])])
        return g_W, g_bias



#Load data from sorted MNIST
import pickle
def save_list(data_nf, data):
    with open(data_nf, 'wb') as f:
        pickle.dump(data, f)

def load_data(data_nf):
    with open(data_nf, 'rb') as f:
        data = pickle.load(f)
    return data

directory = "MNIST_study_epoch/"

sorted_train_data = load_data(directory + "sorted_train_data")
sorted_train_target = load_data(directory + "sorted_train_target")
sorted_test_data = load_data(directory + "sorted_test_data")
sorted_test_target = load_data(directory + "sorted_test_target")




train_data = np.concatenate(sorted_train_data)
train_target = np.concatenate(sorted_train_target)
test_data = np.concatenate(sorted_test_data)
test_target = np.concatenate(sorted_test_target)


#Define network

def coup_func(x, y): return -jnp.cos(x-y)
def bias_func(x, bias): return -jnp.cos(x - bias)
def cost_func(y, target): return -jnp.log(1. + 1e-5 + jnp.cos(y-target))

layer_sizes = [28*28,10]
N = sum(layer_sizes)
input_index = np.arange(0, layer_sizes[0])
output_index = np.arange(N - layer_sizes[-1], N)
layer_XY_structure = N, input_index, output_index, layer_sizes

mnist_lxynn = layer_network_for_mnist(layer_XY_structure, coup_func, bias_func, cost_func)

XY_structure = N, input_index, output_index
mnist_gxynn = general_network_for_mnist(XY_structure, coup_func, bias_func, cost_func, structure_mask=mnist_lxynn.mask)


# Initial network parameters
WL, bias = mnist_lxynn.get_initial_params()
layer_network_params_0 = WL, bias
network_params_0 = mnist_lxynn.merge_params(WL, bias)



# Define gradient calculation with Equilibrium Propagation
N_epoch = 10
beta = 0.1
learning_rate = 0.0001
learning_rate_list = [0.01, 0.004, 0.001]
runtime = 300
rtol = 1e-3
atol = 1e-6

batch_size = 20
M_init = 1

sample_method_list = ['full', 'mini_batch', 'random_init_mini_batch']

grad_params = beta, runtime, rtol, atol
sample_args = sample_method_list[0], batch_size, M_init

para_task = method.EP_grad(grad_params, sample_args)
vmap_task = method.EP_grad_vmap(grad_params, sample_args)


epoch_proj = mnist_epoch_task(vmap_task, mnist_lxynn, layer_network_params_0)


suffix = 'h0_test'
costL0, paramsL0 = epoch_proj.train(N_epoch, learning_rate, batch_size, train_data, train_target, show_process=True, dynamical_save=True, suffix = suffix)





