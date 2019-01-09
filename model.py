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
        shape of the reservoir as (a,b,c) where:
          a: number of input
          b: number of reservoir units
          d: number of output

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
        Noise level inside the reservoir

    seed: int
        Seed for the random generator
    """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)
    
    # Reservoir building
    W_in = rng.uniform(-1.0, 1.0, (shape[1], shape[0]))
    W_rc = rng.uniform(-0.5, 0.5, (shape[1], shape[1]))
    W_rc[rng.uniform(0.0, 1.0, W_rc.shape) > sparsity] = 0.0
    W_rc *= radius / np.max(np.abs(np.linalg.eigvals(W_rc)))

    W_fb = rng.uniform(-1.0, 1.0, (shape[1], shape[2]))
    # W_fb *= rng.uniform(0.0, 1.0, (shape[1], shape[2])) < 0.25
    

    return { "shape"    : shape,
             "sparsity" : sparsity,
             "scaling"  : scaling,
             "leak"     : leak,
             "noise"    : noise,
             "W_in"     : W_in,
             "W_rc"     : W_rc,
             "W_fb"     : W_fb,
             "W_out"    : None }


def train_model(model, data, seed=None): 
    """ Train the model using provided data and seed (noise). """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)

    inputs, outputs = data["input"], data["output"]
    internals = np.zeros((len(data), model["shape"][1]))
    leak = model["leak"]

    # Gathering internal states over all samples
    for i in tqdm.trange(1, len(data)):
        z = (np.dot(model["W_rc"], internals[i-1]) +
             np.dot(model["W_in"], inputs[i]) +
             model["scaling"] * np.dot(model["W_fb"], outputs[i-1]))
        noise = model["noise"] * rng.uniform(-1, 1, z.shape)
        internals[i,:] = np.tanh(z) + noise
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]

    # Computing W_out over a subset of reservoir units
    W_out = np.dot(np.linalg.pinv(internals), outputs).T
    error = np.sqrt(np.mean((np.dot(internals, W_out.T) - outputs)**2))
    model["W_out"] = W_out
    model["last_state"] = inputs[-1], internals[-1], outputs[-1]
    return error



def test_model(model, data, seed=None):
    """ Test the model using provided data and seed (noise). """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)

    # W_ = model["W_rc"] + model["scaling"] * (model["W_out"]*model["W_fb"])
        
    last_input, last_internal, last_output = model["last_state"]
    inputs    = np.vstack([last_input, data["input"]])
    internals = np.vstack([last_internal,
                           np.zeros((len(data), model["shape"][1]))])
    outputs   = np.vstack([last_output, data["output"]])
    leak = model["leak"]
    
    for i in tqdm.trange(1,len(data)+1):

        # Regular version
        z = ( np.dot(model["W_rc"], internals[i-1]) +
              np.dot(model["W_in"], inputs[i]) +
              model["scaling"]*np.dot(model["W_fb"], outputs[i-1]) )

        # Shortened version
        # z = np.dot(W_, internals[i-1]) + np.dot(model["W_in"], inputs[i])
        
        noise = model["noise"]*(rng.uniform(-1.0, +1.0, z.shape))
        internals[i] = np.tanh(z) + noise
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]
        outputs[i] = np.dot(model["W_out"], internals[i])

    model["state"] = internals[1:].T
    model["input"] = inputs[1:]
    model["output"] = outputs[1:]
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))

    return error

