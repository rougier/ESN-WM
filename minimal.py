# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
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
        S = smoothen(rng.uniform(-1, +1, size+length), "hanning", length)
        data["input"][:, idx_value] = S[length:]
    else:
        data["input"][:, idx_value] = rng.uniform(-1, +1, size)
    
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



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = generate_data(2500, p=0.01, seed=1, smooth=True)

    y = data["input"][0,0]
    output = []
    states = []
    scale = 0.005
    norm = np.tanh(1-scale)-np.tanh(1+scale)+np.tanh(scale)-np.tanh(-scale)

    for i in range(len(data)):
        x,t = data["input"][i]
        
        y_ff = (np.tanh(t-scale*x) - np.tanh(t+scale*x) +
                np.tanh(  scale*x) - np.tanh( -scale*x)) / norm

        y_fb = (np.tanh(t+scale*y) - np.tanh(t-scale*y)
              - np.tanh(1+scale*y) + np.tanh(1-scale*y) ) / norm
        
        y = y_ff + y_fb
        output.append(y)

    model = {"output" : np.array(output).reshape(len(output),1),
             "states" : np.array(states) }
             
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))
    print("Error: {0}".format(error))

    
    # Display
    fig = plt.figure(figsize=(14,4))
    fig.patch.set_alpha(0.0)
    n_subplots = 2
    
    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)
    ax1.set_ylim(-1.1,1.1)
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Output")

    ax2 = plt.subplot(n_subplots, 1, 2, sharex=ax1)
    ax2.plot(model["output"]-data["output"],  color='red', lw=1.0)
    ax2.set_ylim(-0.015, +0.015)
    ax2.yaxis.tick_right()
    ax2.axhline(0, color='.75', lw=.5)
    ax2.set_ylabel("Error")
   
    plt.tight_layout()
    plt.show()
