# -----------------------------------------------------------------------------
# Working memory with an echo state network
# Copyright (C) 2018  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
import numpy as np


def generate_data(size=10000, n_input=1, p=0.01):
    """
    size : int
        Number of data to generate
    n_input : int
        Number of signal/tick couples in the input
    p : float 
        Probabilities of a tick in the input
    """
    
    data = np.zeros(size, dtype = [ ("input",  float, (2*n_input,)),
                                    ("output", float, (  n_input,))])
    # Input signals
    data["input"][:,:n_input] = np.random.uniform(-1, +1, (size, n_input))
    
    # Input ticks
    data["input"][:, n_input:] = np.random.uniform(0, 1, (size, n_input)) < p
    data["input"][0, n_input:] = 1

    # Output values according to ticks
    wm = np.zeros(n_input)
    wm[:] = data["input"][0,n_input:]
    for i in range(size):
        for j in range(len(wm)):
            if data["input"][i,n_input+j] > 0:
                wm[j] = data["input"][i,j]
            data["output"][i,j] = wm[j]

    return data

def train(data):
    n = len(data)
    inputs  = data["input"]
    outputs = data["output"]
    internals = np.zeros((n, n_unit))

    # Gathering internal states over all samples
    for i in tqdm.trange(1,n):
        z = ( np.dot(W, internals[i-1]) +
              np.dot(W_in, inputs[i]) +
              scaling*np.dot(W_fb, outputs[i-1]) )
        internals[i,:] = np.tanh(z) + noise*(np.random.uniform(-1.0, 1.0, n_unit))
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]

    # Computing output weights
    W_out = np.dot(np.linalg.pinv(internals), outputs).T
    error = np.sqrt(np.mean((np.dot(internals, W_out.T) - outputs)**2))
    print("Training error : {0}".format(error))

    I_ = internals[:,np.random.randint(0,n_unit,100)]
    W_ = np.dot(np.linalg.pinv(I_), outputs).T
    error = np.sqrt(np.mean((np.dot(I_, W_.T) - outputs)**2))
    print("Training error using 100 random units : {0}".format(error))

    return W_out, (inputs[-1], internals[-1], outputs[-1])


def test(data, state):
    n = len(data)
    inputs  = data["input"]
    outputs = data["output"]
    last_input, last_internal, last_output = state
    inputs    = np.vstack([last_input, inputs])
    internals = np.vstack([last_internal, np.zeros((n, n_unit))])
    outputs   = np.vstack([last_output, np.zeros((n, n_input))])
    for i in tqdm.trange(1,n+1):
        z = ( np.dot(W, internals[i-1]) +
              np.dot(W_in, inputs[i]) +
              scaling*np.dot(W_fb, outputs[i-1]) )
        internals[i] = np.tanh(z) + noise*(np.random.uniform(-1.0, +1.0, n_unit))
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]
        outputs[i] = np.dot(W_out, internals[i])
    return outputs[1:], internals[1:].T


# Parameters
# np.random.seed(3)       # Random number generator seed
n_unit      = 1000      # Number of unit in the reservoir
n_input     = 1         # Number of inputs (input = tick + signal)
sparsity    = 0.25      # Connectivity sparsity inside the reservoir
radius      = 0.01      # Spectral radius
noise       = 0.0001    # Noise level
scaling     = 0.25      # Feedback scaling
leak        = 1.00      # Leak rate

# Reservoir building
W_in = np.random.uniform(-1.0, 1.0, (n_unit, 2*n_input))
W = np.random.uniform(-0.5, 0.5, (n_unit, n_unit))
W[np.random.uniform(0.0, 1.0, W.shape) > sparsity] = 0.0
W *= radius / np.max(np.abs(np.linalg.eigvals(W)))
W_fb = np.random.uniform(-1.0, 1.0, (n_unit, n_input))

# Training
data = generate_data(25000, n_input, p=0.01)
W_out, state = train(data)

# Testing
data = generate_data(2500, n_input, p=0.01)
O = data["output"]
P,R = test(data, state)
error = np.sqrt(np.mean((P - O)**2))
print("Testing error : {0}".format(error))
