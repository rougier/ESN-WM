# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
import numpy as np

 
def generate_model(shape, sparsity=0.25, radius=1.0, scaling=0.25,
                   leak=1.0, noise=0.0, seed=0):
    """
    Generate a reservoir according to parameters

    shape: tuple
        shape of the reservoir as (n_input, n_reservoir, n_output)

    sparsity: float
        Connectivity sparsity inside the reservoir
         (percentage of non null connexions)

    radius: float
        Spectral radius

    scaling: float
        Feedback scaling

    leak: float
        Neuron leak

    noise: float
        Noise leve inside the reservoir

    seed: int
        Seed for the random generator
    """

    # Get a random generator
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)
    else:
        rng = np.random
    
    # Reservoir building
    W_in = rng.uniform(-1.0, 1.0, (shape[1], shape[0]))
    W_rc = rng.uniform(-0.5, 0.5, (shape[1], shape[1]))
    W_rc[rng.uniform(0.0, 1.0, W_rc.shape) > sparsity] = 0.0
    W_rc *= radius / np.max(np.abs(np.linalg.eigvals(W_rc)))
    W_fb = rng.uniform(-1.0, 1.0, (shape[1], shape[2]))

    return { "shape"    : shape,
             "sparsity" : sparsity,
             "scaling"  : scaling,
             "leak"     : leak,
             "noise"    : noise,
             "W_in"     : W_in,
             "W_rc"     : W_rc,
             "W_fb"     : W_fb }


def generate_data(size=10000, p=0.01):
    """
    This function generates input/output data for a gated working memory task:

      Considering an input signal S(t) and a tick signal T(t), the output
      signal O(t) is defined as: O(t) = S(tᵢ) where i = argmax(T(t) = 1).

    size : int
        Number of data to generate
    p : float 
        Probabilities of a tick in the gating signal
    """

    idx_value, idx_tick = 0, 1
    data = np.zeros(size, dtype = [ ("input",  float, (2,)),
                                    ("output", float, (1,))])
    # Input signals
    data["input"][:, idx_value] = np.random.uniform(-1, +1, size)
    
    # # Input ticks
    data["input"][:, idx_tick] = np.random.uniform(0, 1, size) < p

    # Always set tick=1 at time t=0
    data["input"][0, idx_tick] = 1.0

    # Set output values according to ticks
    wm = data["input"][0, idx_value]
    for i in range(size):
        if data["input"][i, idx_tick] > 0:
            wm = data["input"][i, idx_value]
        data["output"][i,0] = wm
        
    return data


def train(data): 
    """
    """
    
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
    outputs   = np.vstack([last_output, outputs])
    for i in tqdm.trange(1,n+1):
        z = ( np.dot(W, internals[i-1]) +
              np.dot(W_in, inputs[i]) +
              scaling*np.dot(W_fb, outputs[i-1]) )
        internals[i] = np.tanh(z) + noise*(np.random.uniform(-1.0, +1.0, n_unit))
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]
        outputs[i] = np.dot(W_out, internals[i])
    return outputs[1:], internals[1:].T


# Parameters
np.random.seed(3)    # Random number generator seed
n_unit      = 1000   # Number of unit in the reservoir
n_input     = 2      # Number of input
n_output    = 1      # Number of output
sparsity    = 0.50   # Connectivity sparsity inside the reservoir
radius      = 0.01   # Spectral radius
noise       = 0.0001 # Noise level
scaling     = 0.25   # Feedback scaling
leak        = 1.00   # Leak rate

# Reservoir building
W_in = np.random.uniform(-1.0, 1.0, (n_unit, n_input))
W = np.random.uniform(-0.5, 0.5, ( n_unit, n_unit))
W[np.random.uniform(0.0, 1.0, W.shape) > sparsity] = 0.0
W *= radius / np.max(np.abs(np.linalg.eigvals(W)))
W_fb = np.random.uniform(-1.0, 1.0, (n_unit, n_output))

# Training
data = generate_data(25000, p=0.01)
W_out, state = train(data)


# Testing
data = generate_data(2500, p=0.01)


O = data["output"]
P,R = test(data, state)
error = np.sqrt(np.mean((P - O)**2))
print("Testing error : {0}".format(error))


# Find the most correlated unit in the reservoir (during testing)
from scipy.stats.stats import pearsonr
C = np.zeros(n_unit)
for i in range(n_unit):
    C[i], p = pearsonr(R[i].ravel(), O.ravel())
mc = np.argsort(C)[-1]


# Display
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

ax1 = plt.subplot(5,1,1)
ax1.plot(O,  color='0.75', lw=1.0)
ax1.plot(P, color='0.00', lw=1.5)
ax1.set_ylim(-1.1,1.1)
ax1.yaxis.tick_right()
ax1.set_ylabel("Output")

ax2 = plt.subplot(5,1,2, sharex=ax1)
ax2.plot(P-O,  color='red', lw=1.0)
ax2.set_ylim(-0.015, +0.015)
ax2.yaxis.tick_right()
ax2.axhline(0, color='.75', lw=.5)
ax2.set_ylabel("Error")

ax3 = plt.subplot(5,1,3, sharex=ax1)
ax3.plot(R[mc], color='k', alpha=.5, lw=.5)
ax3.yaxis.tick_right()
ax3.set_ylabel("Reservoir[i={0}]".format(mc))

ax4 = plt.subplot(5,1,4)
ax4.plot(np.sort(W_out.ravel()))
ax4.set_ylabel("Out weights\n(sorted)")
ax4.yaxis.tick_right()

ax5 = plt.subplot(5,1,5)
ax5.plot(np.sort(C[np.argsort(W_out)[0]]))
ax5.set_ylabel("Correlation\n(sorted)")
ax5.yaxis.tick_right()

plt.tight_layout()
plt.show()


plt.imshow(R[np.argsort(C),:])
plt.show()
