# Load MNIST with torchvision
import torch
import torchvision

trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=None)
testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=None)

# Classify and transform mnist data
import numpy as np
import jax

def sorted_data(data, labels):
    # Sort mnist images with different labels
    sorted_data = []
    sorted_target = []
    N_data = data.shape[0]
    
    for k in range(0, 10):
        sorted_data.append([])
        sorted_target.append([])
    
    for k in range(0, N_data):
        sorted_data[labels[k]].append(np.concatenate(data[k]))
        sorted_target[labels[k]].append(jax.nn.one_hot(labels[k], num_classes=10))
    
    for k in range(0, 10):
        sorted_data[k] = np.asarray(sorted_data[k])
        sorted_target[k] = np.asarray(sorted_target[k])
    
    
    return sorted_data, sorted_target

def rescale(data, a, b):
    # a: lower limit
    # b: upper limit
    max_val = np.max(data)
    min_val = np.min(data)
    return data/(max_val - min_val) * (b-a) + a

mnist_train_data = np.asarray(trainset.data)
mnist_train_target = np.asarray(trainset.targets)

mnist_test_data = np.asarray(testset.data)
mnist_test_target = np.asarray(testset.targets)

np.save("mnist_train_data", mnist_train_data)
np.save("mnist_train_target", mnist_train_target)
np.save("mnist_test_data", mnist_test_data)
np.save("mnist_test_target", mnist_test_target)

sorted_train_data, sorted_train_target = jax.tree_map(lambda x: rescale(x, -np.pi/2, np.pi/2), sorted_data(mnist_train_data, mnist_train_target))
sorted_test_data, sorted_test_target = jax.tree_map(lambda x: rescale(x, -np.pi/2, np.pi/2), sorted_data(mnist_test_data, mnist_test_target))

# Save data
import pickle
def save_list(data_nf, data):
    with open(data_nf, 'wb') as f:
        pickle.dump(data, f)

def load_data(data_nf):
    with open(data_nf, 'rb') as f:
        data = pickle.load(f)
    return data

save_list('sorted_train_data', sorted_train_data)
save_list('sorted_train_target', sorted_train_target)
save_list('sorted_test_data', sorted_test_data)
save_list('sorted_test_target', sorted_test_target)