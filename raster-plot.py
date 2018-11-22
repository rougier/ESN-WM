# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import tqdm
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
    
    data = generate_data(500, p=0.01, seed=1, smooth=True)
    data = data["input"][:,0]
    
    # Display
    fig = plt.figure(figsize=(10,3))
    fig.patch.set_alpha(0.0)
    n_subplots = 1

    ax = plt.subplot(n_subplots, 1, 1)
        
    X = np.arange(len(data))
    Y = np.ones(len(data))
    data = (data - data.min()) / (data.max() - data.min())
    for i in range(100):
        D = data + np.random.uniform(0.0,0.1, len(data))
        C = np.ones((len(data), 3))
        C = C * (1-(np.random.uniform(0, 1, len(D)) < 0.25*D)).reshape(len(D),1)
        ax.scatter(X, Y+i, s=1, edgecolor="None", facecolor=C)
        
        
    plt.tight_layout()
    plt.show()

    
