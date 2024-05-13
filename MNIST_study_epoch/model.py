import numpy as np
import jax
import jax.numpy as jnp

from functools import partial

class network:
    '''
    This define a neural network. It should include:
    network_structure: network_type, structure_name, structure_parameter, activation
    __init__: initialization
    get_initial_params(): randomly generate a set of trainable parameter for the neural network
    get_initial_state(input_data): generate a random initial state with specific input_data
    internal_energy(y, network_params): calculate internal energy function
    external_energy(y, target, network_params): calculate external energy
    internal_force(y, network_paarams): 
    external_force(y, network_params):
    params_derivative(self, y, network_params): 
    '''
    
    def __init__(self) -> None:
        pass
    
    def get_initial_params(self):
        pass
    
    def get_initial_state(input_data):
        pass
    
    def internal_energy(self, y, network_params): 
        pass
    
    def distance_function(self, y, target, network_params):
        pass
    
    def external_energy(self, y, target, network_params): 
        return self.distance_function(y, target, network_params)
    
    def internal_force(self, y, network_params): 
        return -jax.grad(self.internal_energy, argnums=0)(y, network_params)
    
    def external_force(self, y, target, network_params):
        return -jax.grad(self.external_energy, argnums=0)(y, target, network_params)
    
    def params_derivative(self, y, network_params):
        d_energy = jax.grad(self.internal_energy, argnums=1)
        return d_energy(y, network_params)    
    
    def get_random_index(self, N_data, batch_size):
        if batch_size == N_data:
            return np.arange(0, N_data, dtype=np.int32)
        return np.random.randint(0, N_data, batch_size)
    
    
class Hopfield_network(network):
    '''
    This defines a hopfield-like network with all-to-all connectivity
    '''
    
    def __init__(self, network_structure, network_type='Hopfield', structure_name='all to all'):
        
        # network_structure = N, input_index, output_index, activation
        self.network_type = network_type
        self.structure_name = structure_name
        self.network_structure = network_structure
        self.activation = network_structure[-1]
        self.d_activation = jax.grad(self.activation, 0)
    
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        W = 1/np.sqrt(N) * np.random.randn(N, N)
        W = (W + np.transpose(W))/2
        
        bias = np.random.randn(N)
        
        return W, bias
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index, activation = self.network_structure
        
        y0 = 2 * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    def get_initial_state_mini_batch(self, input_data, target, batch_size):
        N_data = input_data.shape[0]
        N, input_index, output_index, activation = self.network_structure
        
        data_ind = np.random.randint(0, N_data, batch_size)
        y0 = 2 * (np.random.rand(batch_size, N) - 0.5)
        y0[:, input_index] = input_data[data_ind, :]
        
        return y0, target[data_ind, :]
    
    def internal_energy(self, y, network_params):
        W, bias = network_params
        N = bias.shape[0]
        
        E0 = jnp.dot(y, y)/2
        
        my = jnp.tensordot(self.activation(y), jnp.ones(N), 0)
        E1 = - jnp.tensordot(W, my * jnp.transpose(my))/2
        E2 = - jnp.dot(bias, self.activation(y))
        
        return E0 + E1 + E2

    def internal_force(self, y, network_params):
        W, bias = network_params
        input_index = self.network_structure[1]
        #N = bias.shape[0]
        
        sy = jax.vmap(self.d_activation,(0))(y)
        
        F = y - jnp.dot(jnp.asarray(W), self.activation(y)) * sy - bias * sy
        F = F.at[input_index].set(0)
        
        return -F
    
    
    def distance_function(self, y, target, network_params):
        W, bias = network_params
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = jnp.dot(dy, dy)/2
        return cost
    
    def external_energy(self, y, target, network_params):
        
        W, bias = network_params
        
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = jnp.sum(jnp.log(1 - jnp.power(dy, 2)))/2
        return cost
    
    def external_force(self, y, target, network_params):
        output_index = self.network_structure[2]
        F = 0*y
        #F = F.at[output_index].set(-(y[output_index] - target)/(1 - jnp.power(y[output_index]-target, 2)))
        F = F.at[output_index].set(- y[output_index] + target)
        
        return F
    
    def params_derivative(self, y, network_params):
        N = y.shape[0]
        sy = self.activation(y)
        my = jnp.tensordot(sy, jnp.ones(N), 0)
        return -my * jnp.transpose(my), -sy
    
    

class XY_network(network):
    '''
    This defince a XY model viewed as an neural network with all-to-all connectivity
    '''
    def __init__(self, network_structure, network_type='XY', structure_name='all to all'):
        
        # network_structure = N, input_index, output_index
        self.network_type = network_type
        self.structure_name = structure_name
        self.network_structure = network_structure
    
    #--------------------Initialization of the network-----------------------
    
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        W = 1/np.sqrt(N) * np.random.randn(N, N)
        W = (W + np.transpose(W))/2
        
        bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        
        return W, bias
    
    #--------------------Initialization of states network-----------------------
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure
        
        # the initial state follows a uniform distribution over (-\pi, \pi)
        y0 = 2 * np.pi * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    def get_initial_state_mini_batch(self, input_data, target, batch_size):
        #select a random mini-batch of data from total dataset
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        data_ind = self.get_random_index(N_data, batch_size)
        y0 = 2 * np.pi * (np.random.rand(batch_size, N) - 0.5)
        y0[:, input_index] = input_data[data_ind, :]
        
        return y0, target[data_ind, :]
    
    def get_multiple_init_data(self, input_data, target, M_init, batch_size):
        # prepare folded mini-batch dataset for multiple random initialization
        N_data = input_data.shape[0]
        
        data_ind = self.get_random_index(N_data, batch_size)
        
        mini_input = input_data[data_ind, :]
        mini_target = target[data_ind, :]
        
        batch_input = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_input, 0))
        batch_target = jnp.concatenate(jnp.tensordot(jnp.ones(M_init), mini_target, 0))
        
        return batch_input, batch_target
    
    def get_multiple_init_initial_state(self, input_data, target, batch_size, M_init):
        
        batch_input, batch_target = self.get_multiple_init_data(input_data, target, M_init, batch_size)
        y0 = self.get_initial_state(batch_input)
        return y0, batch_target
    
    #----------------------------Dynamics of the network-------------------------------
    
    def internal_energy(self, y, network_params):
        W, bias = network_params
        N = W.shape[0]
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        E0 = - jnp.tensordot(W, jnp.cos(my - jnp.transpose(my)))/2
        
        E1 = - jnp.dot(bias[0], jnp.cos(y-bias[1]))
        
        return E0 + E1

    def internal_force(self, y, network_params):
        W, bias = network_params
        input_index = self.network_structure[1]
        N = W.shape[0]
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        F = -jnp.sum(W * jnp.sin(my - jnp.transpose(my)), axis=1) - bias[0] * jnp.sin(y - bias[1])
        F = F.at[input_index].set(0)
        
        return F
    
    def distance_function(self, y, target, network_params):
        W, bias = network_params
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = jnp.sum(1-jnp.cos(dy))/2
        return cost
    
    def external_energy(self, y, target, network_params):
        
        W, bias = network_params
        
        output_index = self.network_structure[2]
        dy = y[output_index] - target
        cost = -jnp.sum(jnp.log(1 + jnp.cos(dy)))
        return cost
    
    def external_force(self, y, target, network_params):
        output_index = self.network_structure[2]
        F = 0 * y
        #F = F.at[output_index].set(-(y[output_index] - target)/(1 - jnp.power(y[output_index]-target, 2)))
        dy = y[output_index] - target
        
        F = F.at[output_index].set(-jnp.sin(dy)/(1 + 1e-5 + jnp.cos(dy)))
        
        return F
    
    def params_derivative(self, y, network_params):
        W, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        dmy = my - jnp.transpose(my)
        
        # calculate dE/dW
        g_W = - jnp.cos(dmy)
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([-jnp.cos(y - bias[1]), -bias[0]*jnp.sin(y - bias[1])])
        return g_W, g_bias
    
class general_XY_network(XY_network):
    """
    This define a network implementing energy function of arbitrary two-point interaction, bias and cost function
    coup_func, bias_func: guarantee that it reaches minimum when all the other interactions are screened, e.g., for XY, coup_func = bias_func = -cos(* - *)
    """
        
    def __init__(self, network_structure, coup_func, bias_func, cost_func, network_type='general XY', structure_name='all to all'):
        super().__init__(network_structure, network_type, structure_name)
        
        self.coup_func, self.bias_func, self.cost_func = coup_func, bias_func, cost_func
        self.d0coup_func = jax.grad(coup_func, 0)
        self.d0bias_func = jax.grad(bias_func, 0)
        self.d1bias_func = jax.grad(bias_func, 1)
        self.dcost_func = jax.grad(cost_func, 0)
    
    #========================Internal dynamics=========================
    def internal_energy(self, y, network_params):
        W, bias = network_params
        N = W.shape[0]
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))
        
        v_bias_func = jax.vmap(self.bias_func, (0,0))
        
        E0 = jnp.tensordot(W, m_coup_func(my, jnp.transpose(my)))/2
        
        E1 = jnp.dot(bias[0], v_bias_func(y, bias[1]))
        
        return E0 + E1
    
    def internal_force(self, y, network_params):
        W, bias = network_params
        input_index = self.network_structure[1]
        N = W.shape[0]
        
        m_dcoup_func = jax.vmap(jax.vmap(self.d0coup_func, (0, 0)), (0, 0))
        v_dbias_func = jax.vmap(self.d0bias_func, (0, 0))
        
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        F = -jnp.sum(W * m_dcoup_func(my, jnp.transpose(my)), axis=1) - bias[0] * v_dbias_func(y, bias[1])
        F = F.at[input_index].set(0)
        
        return F
    
    #========================External dynamics=====================================
    
    def external_energy(self, y, target, network_params):
        
        #W, bias = network_params
        
        output_index = self.network_structure[2]
        
        cost = -jnp.sum(jax.vmap(self.cost_func, (0,0))(y[output_index], target))
        return cost
    
    def external_force(self, y, target, network_params):
        output_index = self.network_structure[2]
        F = 0 * y
        #F = F.at[output_index].set(-(y[output_index] - target)/(1 - jnp.power(y[output_index]-target, 2)))
        dy = y[output_index] - target
        
        F = F.at[output_index].set(-jax.vmap(self.dcost_func, (0,0))(y[output_index], target))
        
        return F
    
    def params_derivative(self, y, network_params):
        W, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))
        v_dbias_func = jax.vmap(self.d1bias_func, (0, 0))
        v_bias_func = jax.vmap(self.bias_func, (0, 0))
        
        # calculate dE/dW
        g_W = m_coup_func(my, jnp.transpose(my))
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([v_bias_func(y, bias[1]), bias[0]*v_dbias_func(y, bias[1])])
        return g_W, g_bias
    
class layered_general_XY_network(general_XY_network):
    '''
    This implement a generall XY network with layered structure
    '''
    
    def __init__(self, network_structure, coup_func, bias_func, cost_func, network_type='general XY', structure_name='layered'):
        super().__init__(network_structure, coup_func, bias_func, cost_func, network_type, structure_name)
        
        #Here network_structure = N, input_index, output_index, layer_sizes
        
        self.split_points = [network_structure[-1][0]]
        for k in range(1, len(network_structure[-1])-1):
            self.split_points.append(self.split_points[-1]+network_structure[-1][k])
        
        index_list = [0]
        for k in range(0, len(network_structure[-1])):
            index_list.append(index_list[-1]+network_structure[-1][k])
        
        self.mask = np.zeros([network_structure[0], network_structure[0]])
        for k in range(0, len(index_list)-2):
            self.mask[index_list[k]:index_list[k+1], index_list[k+1]:index_list[k+2]] = 1
        
        self.mask = self.mask + np.transpose(self.mask)
        
        self.layer_shape = jax.tree_map(jnp.zeros, self.network_structure[-1])
        self.structure_shape = jax.tree_map(jnp.zeros, self.split_points)
        self.index_list = index_list
            
    
    #==============================================initial netowrk parameters========================
    def get_initial_params(self):
        
        # get a set of weights and bias. The weight matrix is symmetric. 
        
        N = self.network_structure[0]
        N_list = self.network_structure[-1]
        depth = len(N_list)
        
        WL = []
        for k in range(0, depth-1):
            WL.append( 1/np.sqrt(N_list[k] + N_list[k+1]) * np.random.randn(N_list[k], N_list[k+1]) )
        
        bias = np.asarray([0*np.random.randn(N), 2*np.pi*(np.random.rand(N) - 0.5)])
        
        return WL, bias
    
    #========================initial states===========================
    
    def get_initial_state(self, input_data):
        # generate initial state for a set of input data
        
        N_data = input_data.shape[0]
        N, input_index, output_index = self.network_structure[0:3]
        
        # the initial state follows a uniform distribution over (-\pi, \pi)
        y0 = 2 * np.pi * (np.random.rand(N_data, N) - 0.5)
        y0[:, input_index] = input_data
        
        return y0
    
    #======================calculate interactions between neighbor layers=================
    
    def adjacent_energy(self, y1, W, y2):
        # This calculate \sum W_ij * f(y1_i, y2_j)
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        return jnp.tensordot(W, jax.vmap(jax.vmap(self.coup_func, (0,0)), (0,0))(my1, my2))
    
    
    def adjacent_forces(self, y1, W, y2):
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        # This force acts on y2
        forward_force = jnp.sum(W * jax.vmap(jax.vmap(self.d0coup_func, (0,0)), (0,0))(my1, my2), axis=0)
        
        # This force acts on y2
        backward_force = -jnp.sum(W * jax.vmap(jax.vmap(self.d0coup_func, (0,0)), (0,0))(my1, my2), axis=1)
        
        return forward_force, backward_force
    
    def internal_energy(self, y, network_params):
        
        WL, bias = network_params
        N = bias.shape[1]
        layer_sizes = self.network_structure[-1]
        
        m_coup_func = jax.vmap(jax.vmap(self.coup_func, (0, 0)), (0, 0))  
        v_bias_func = jax.vmap(self.bias_func, (0,0))
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        E0 = jnp.sum(jnp.asarray(jax.tree_map(self.adjacent_energy, yl1, WL, yl2)))
        
        E1 = jnp.dot(bias[0], v_bias_func(y, bias[1]))
        
        return E0+E1
    
    def internal_force(self, y, network_params):
        WL, bias = network_params
        input_index = self.network_structure[1]
        N = bias.shape[1]
        
        m_dcoup_func = jax.vmap(jax.vmap(self.d0coup_func, (0, 0)), (0, 0))
        v_dbias_func = jax.vmap(self.d0bias_func, (0, 0))
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]

        res = jax.tree_map(self.adjacent_forces, yl1, WL, yl2)
        ff = jnp.concatenate(list(zip(*res))[0])
        bf = jnp.concatenate(list(zip(*res))[1])
        
        F1 = jnp.zeros(N)
        F2 = jnp.zeros(N)
        
        #back and forward force from 2-body interaction
        F1 = F1.at[jnp.arange(self.split_points[0], N)].set(ff)
        F2 = F2.at[jnp.arange(0, self.split_points[-1])].set(bf)
        
        #force from bias
        F3 = - bias[0] * v_dbias_func(y, bias[1])
        
        F = F1 + F2 + F3
        F = F.at[input_index].set(0)
        return F
    
    #=================Calculate dynamical terms from external energy============
    # This part is not interferred by the layer atchitecture and is therefore not necessary to change
    
    #=================Calculate parameter derivatives==============
    def W_derivative(self, y1, y2):
        N1, N2 = y1.shape[0], y2.shape[0]
        my1 = jnp.tensordot(y1, jnp.ones(N2), 0)
        my2 = jnp.tensordot(jnp.ones(N1), y2, 0)
        
        return jax.vmap(jax.vmap(self.coup_func, (0,0)), (0,0))(my1, my2)
    
    @partial(jax.jit, static_argnames=['self'])
    def params_derivative(self, y, network_params):
        WL, bias = network_params
        N = y.shape[0]

        # dmy[i,j] = y[i] - y[j]
        my = jnp.tensordot(y, jnp.ones(N), 0)
        
        v_dbias_func = jax.vmap(self.d1bias_func, (0, 0))
        v_bias_func = jax.vmap(self.bias_func, (0, 0))
        
        # calculate dE/dW
        layer_sizes = self.network_structure[-1]
        
        yl = jnp.split(y, self.split_points)
        
        yl1 = yl.copy()
        del yl1[-1]
        
        yl2 = yl.copy()
        del yl2[0]
        
        g_W = jax.tree_map(self.W_derivative, yl1, yl2)
        
        # calculate dE/dh
        
        g_bias = jnp.asarray([v_bias_func(y, bias[1]), bias[0]*v_dbias_func(y, bias[1])])
        return g_W, g_bias
    
    #=====================Correlate layered network to a all-to-all network=================
    def merge_params(self, WL, bias):
        N = bias.shape[1]
        depth = len(self.index_list) - 1
        
        W = np.zeros([N, N])
        
        for n in range(0, depth-1):
            W[self.index_list[n]:self.index_list[n+1], self.index_list[n+1]:self.index_list[n+2]] = WL[n]
        
        W = W + np.transpose(W)
        
        return W, bias
        
