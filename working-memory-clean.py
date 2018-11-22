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



def smoothen(Z, window='hanning', length=25):
    """
    Smoothen a signal by averaging it over a fixed-size window


    Z : np.array
        Signal to smoothen

    window: string
        Specify how to compute the average over neighbours
        One of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'

    length: int
        Size of the averaging window
    """
    
    # window in 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    S = np.r_[Z[length-1:0:-1], Z, Z[-2:-length-1:-1]]
    if window == 'flat': 
        W = np.ones(length,'d')
    else:
        W = eval('np.' + window + '(length)')
    Z = np.convolve(W/W.sum(), S, mode='valid')
    return 2*Z[(length//2-1):-(length//2)-1]



def generate_data(size=10000, p=0.01, seed=None, smooth=False):
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
    if smooth:
        length = 25
        V = rng.uniform(-1, +1, size)
        S = smoothen(V, "hanning", length)
        data["input"][:, idx_value] = S[:]
    else:
        V = rng.uniform(-1, +1, size)
        data["input"][:, idx_value] = V
    
    # Input ticks
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



def train(model, data, seed=None): 
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



def test(model, data, seed=None):
    """ Test the model using provided data and seed (noise). """

    # Get a random generator
    rng = np.random
    if seed is not None:
        rng = np.random.mtrand.RandomState(seed)


    W_ = model["W_rc"] + model["scaling"] * (model["W_out"]*model["W_fb"])
        
    last_input, last_internal, last_output = model["last_state"]
    inputs    = np.vstack([last_input, data["input"]])
    internals = np.vstack([last_internal,
                           np.zeros((len(data), model["shape"][1]))])
    outputs   = np.vstack([last_output, data["output"]])
    leak = model["leak"]
    
    for i in tqdm.trange(1,len(data)+1):

        # Regular version
        # z = ( np.dot(model["W_rc"], internals[i-1]) +
        #       np.dot(model["W_in"], inputs[i]) +
        #       model["scaling"]*np.dot(model["W_fb"], outputs[i-1]) )

        # Shortened version
        z = np.dot(W_, internals[i-1]) + np.dot(model["W_in"], inputs[i])
        
        noise = model["noise"]*(rng.uniform(-1.0, +1.0, z.shape))
        internals[i] = np.tanh(z) + noise
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]
        outputs[i] = np.dot(model["W_out"], internals[i])

    model["state"] = internals[1:].T
    model["input"] = inputs[1:]
    model["output"] = outputs[1:]
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))

    return error



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Random generator initialization
    np.random.seed(3)
    
    # Build the model
    model = generate_model(shape=(2,1000,1), sparsity=0.25, radius=0.1,
                           scaling=0.25, leak=0.75, noise=0.0001)

    # Training
    data = generate_data(25000, p=0.01, smooth=False)
    error = train(model, data)
    print("Training error : {0}".format(error))
    
    # Testing
    data = generate_data(2500, p=0.01, seed=1, smooth=True)
    error = test(model, data)
    print("Testing error : {0}".format(error))


    
    # Display
    fig = plt.figure(figsize=(14,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 4


    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,1]
    ax1.scatter(X, -0.9*Y, s=1, facecolors=C, edgecolors=None)
    ax1.text(-25, -0.9, "Ticks:",
             fontsize=8, transform=ax1.transData,
             horizontalalignment="right", verticalalignment="center")
    ax1.set_ylim(-1.1,1.1)
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Input & Output")
    ax1.text(0.01, 0.9, "A",
             fontsize=16, fontweight="bold", transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")


    ax2 = plt.subplot(n_subplots, 1, 2, sharex=ax1)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(model["output"]-data["output"],  color='red', lw=1.0)
    ax2.set_ylim(-0.011, +0.011)
    ax2.yaxis.tick_right()
    ax2.axhline(0, color='.75', lw=.5)
    ax2.set_ylabel("Output error")
    ax2.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")


    # Find the most correlated unit in the reservoir (during testing)
    from scipy.stats.stats import pearsonr
    n = len(model["state"])
    C = np.zeros(n)
    for i in range(n):
        C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    I = np.argsort(np.abs(C))


    ax3 = plt.subplot(n_subplots, 1, 3, sharex=ax1)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    n = 20
    for i in range(n):
        ax3.plot(model["state"][I[-1-i]], color='k', alpha=.25, lw=.5)
    ax3.yaxis.tick_right()
    ax3.set_ylim(-1.1, +1.1)
    ax3.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    ax3.text(0.01, 0.9, "C",
             fontsize=16, fontweight="bold", transform=ax3.transAxes,
             horizontalalignment="left", verticalalignment="top")


    ax4 = plt.subplot(n_subplots, 1, 4, sharex=ax1)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    n = 20
    for i in range(n):
        ax4.plot(model["state"][I[i]], color='k', alpha=.25, lw=.5)
    ax4.yaxis.tick_right()
    ax4.set_ylim(-1.1, +1.1)
    ax4.set_ylabel("Least correlated\n internal units (n={0})".format(n))
    ax4.text(0.01, 0.9, "D",
             fontsize=16, fontweight="bold", transform=ax4.transAxes,
             horizontalalignment="left", verticalalignment="top")
    
    plt.tight_layout()
    plt.savefig("working-memory-clean.pdf")
    plt.show()

    
