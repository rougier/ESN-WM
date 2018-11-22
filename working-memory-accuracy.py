# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
def generate_model(shape, sparsity=0.25, radius=1.0, scaling=0.25,
                   leak=1.0, noise=0.0, seed=None):
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
        Noise leve inside the reservoir

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

    return { "shape"    : shape,
             "sparsity" : sparsity,
             "scaling"  : scaling,
             "leak"     : leak,
             "noise"    : noise,
             "W_in"     : W_in,
             "W_rc"     : W_rc,
             "W_fb"     : W_fb,
             "W_out"    : None }



# -----------------------------------------------------------------------------
def generate_data(size=10000, p=0.01, seed=None):
    """
    This function generates input/output data for a gated working memory task:

      Considering an input signal S(t) and a tick signal T(t), the output
      signal O(t) is defined as: O(t) = S(tᵢ) where i = argmax(T(t) = 1).

    size : int
        Number of data to generate

    p : float 
        Probabilities of a tick in the gating signal
    """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)

    idx_value, idx_tick = 0, 1
    data = np.zeros(size, dtype = [ ("input",  float, (2,)),
                                    ("output", float, (1,))])
    # Input signals
    data["input"][:, idx_value] = rng.uniform(-1, +1, size)
    
    # # Input ticks
    data["input"][:, idx_tick] = rng.uniform(0, 1, size) < p

    # Always set tick=1 at time t=0
    data["input"][0, idx_tick] = 1.0

    # Set output values according to ticks
    wm = data["input"][0, idx_value]
    for i in range(size):
        if data["input"][i, idx_tick] > 0:
            wm = data["input"][i, idx_value]
        data["output"][i,0] = wm
        
    return data



# -----------------------------------------------------------------------------
def train(model, data, seed=None): 
    """ Train the model using provided data. """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)

    inputs, outputs = data["input"], data["output"]
    internals = np.zeros((len(data), model["shape"][1]))

    # Gathering internal states over all samples
    for i in range(1, len(data)):
        z = (np.dot(model["W_rc"], internals[i-1]) +
             np.dot(model["W_in"], inputs[i]) +
             model["scaling"] * np.dot(model["W_fb"], outputs[i-1]))
        internals[i,:] = np.tanh(z) + model["noise"] * rng.uniform(-1, 1, z.shape)
        internals[i,:] = (1-model["leak"])*internals[i-1] + model["leak"]*internals[i,:]

    # Computing W_out over a subset of reservoir units
    W_out = np.dot(np.linalg.pinv(internals), outputs).T
    error = np.sqrt(np.mean((np.dot(internals, W_out.T) - outputs)**2))

    model["W_out"] = W_out
    model["last_state"] = inputs[-1], internals[-1], outputs[-1]

    # print("Training error : {0}".format(error))
    return error


# -----------------------------------------------------------------------------
def test(model, data, seed=None):
    """ Test the model using provided data. """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)
    
    last_input, last_internal, last_output = model["last_state"]
    inputs    = np.vstack([last_input, data["input"]])
    internals = np.vstack([last_internal, np.zeros((len(data), model["shape"][1]))])
    outputs   = np.vstack([last_output, data["output"]])
    for i in range(1,len(data)+1):
        z = ( np.dot(model["W_rc"], internals[i-1]) +
              np.dot(model["W_in"], inputs[i]) +
              model["scaling"]*np.dot(model["W_fb"], outputs[i-1]) )
        internals[i] = np.tanh(z) + model["noise"]*(rng.uniform(-1.0, +1.0, z.shape))
        internals[i,:] = (1-model["leak"])*internals[i-1] + model["leak"]*internals[i,:]
        outputs[i] = np.dot(model["W_out"], internals[i])

    model["state"] = internals[1:].T
    model["output"] = outputs[1:]
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))

    return error



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(1)
    for n in range(10,1000+1,5):
        error = []
        for j in range(10):
            model = generate_model(shape=(2,n,1), sparsity=0.5, radius=0.1,
                                   scaling=0.25, leak=1.0, noise=0.0001)
            data = generate_data(25000, p=0.01)
            train(model, data)
            data = generate_data(2500, p=0.01)
            error.append(test(model, data))

        print("%d, %f, %f" % (n, np.mean(error), np.std(error)))

