# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

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


def generate_data(values, ticks, last=None):
    """
    This function generates output data for a gated working memory task:

      Considering an input signal S(t) and a tick signal T(t), the output
      signal O(t) is defined as: O(t) = S(tᵢ) where i = argmax(T(t) = 1).

    values : np.array
        Input signal as a single sequence of random float

    ticks : np.array
        Gating signal(s) as one (or several) sequence(s) of 0 and 1 
    """

    values = np.array(values).ravel()
    ticks = np.array(ticks)
    if len(ticks.shape) == 1:
        ticks = ticks.reshape(len(ticks), 1)
    n_gate = ticks.shape[1]
    size = len(values)
    
    data = np.zeros(size, dtype = [ ("input",  float, (1 + n_gate,)),
                                    ("output", float, (    n_gate,))])
    # Input signals
    data["input"][:,0 ] = values
    data["input"][:,1:] = ticks


    wm = np.zeros(n_gate)
    # If no last activity set tick=1 at time t=0
    if last is None:
        wm[:] = data["input"][0, 0]
        data["input"][0, 1:] = 1
    else:
        wm[:] = last

    # Output value(s) according to ticks
    for i in range(size):
        for j in range(n_gate):
            if data["input"][i,1+j] > 0:
                wm[j] = data["input"][i,0]
            data["output"][i,j] = wm[j]

    return data


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n,2)) < 0.01
    data = generate_data(values, ticks)
    print("Data size: {0}".format(len(data)))
    print("Data dtype: {0}".format(data.dtype))    

    plt.figure(figsize=(12,2.5))
    plt.plot(data["input"][:,0],  color='0.75', lw=1.0)
    plt.plot(data["output"][:,0], color='0.00', lw=1.5)
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.show()
    """


    
    n = 50
    np.random.seed(6)
    values = np.random.uniform(0, +1, n)

    ticks = np.random.uniform(0, 1, (n,1)) < 0.05
    data1 = generate_data(values, ticks)

    ticks = np.random.uniform(0, 1, (n,3)) < 0.05
    data3 = generate_data(values, ticks)

    cmap = "magma"
    S  = [
        ( 6, data1["input"][:,0],  cmap, 0.75, "Value (V)"),
        ( 5, data3["input"][:,1],  "gray_r",  1.00, "Trigger (T₁)"),
        ( 4, data3["output"][:,0], cmap, 0.75, "Output (M₁)"),
        ( 3, data3["input"][:,2],  "gray_r",  1.00, "Trigger (T₂)"),
        ( 2, data3["output"][:,1], cmap, 0.75, "Output (M₂)"),
        ( 1, data3["input"][:,3],  "gray_r",  1.00, "Trigger (T₃)"),
        ( 0, data3["output"][:,2], cmap, 0.75, "Output (M₃)"),

        (10, data1["input"][:,0],  cmap, 0.75, "Value (V)"),
        ( 9, data1["input"][:,1],  "gray_r",  1.00, "Trigger (T)"),
        ( 8, data1["output"][:,0], cmap, 0.75, "Output (M)") ]
    
    fig = plt.figure(figsize=(10,2.5))
    ax = plt.subplot(1,1,1, frameon=False)
    ax.tick_params(axis='y', which='both', length=0)
    
    X = np.arange(n)
    Y = np.ones(n)
    yticks = []
    ylabels = []
    for (index, V, cmap, alpha, label) in S:
        ax.scatter(X, index*Y, s=100, vmin=0, vmax=1, alpha=alpha,
                   edgecolor="None", c=V, cmap=cmap)
        ax.scatter(X, index*Y, s=100, edgecolor="k", facecolor="None",
                   linewidth=0.5)
        yticks.append(index)
        ylabels.append(label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.5,10.5)

    ax.set_xticks([])
    ax.set_xlim(-0.5,n-0.5)


    plt.savefig("data.pdf")
    plt.show()
