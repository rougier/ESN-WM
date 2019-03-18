# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp


def convert_data(data_, size):
    values = (data_["input"][:, 0]).astype(int)
    text = [chr(ord("0")+i) for i in values]
    Z, I = str_to_bmp(text, size = size)
    data = np.zeros(Z.shape[1], dtype = [ ("input",  float, (1 + Z.shape[0],)),
                                          ("output", float, (    n_gate,))])
    data["input"][:, :-1] = Z.T
    n = Z.shape[1]//len(text)
    data["input"][:,-1] = np.repeat(data_["input"][:, 1], n)
    data["output"][:, 0] = np.repeat(data_["output"], n) / 10
    return data


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Random generator initialization
    np.random.seed(123)
    
    # Build memoryticks
    n_gate = 1
    fontsize = 11

    # Training data
    n = 20
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
    train_data_ = generate_data(values, ticks)
    train_data = convert_data(train_data_, fontsize)

    # Testing data
    n = 9
    values = np.random.randint(0, 10, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.5
    test_data_ = generate_data(values, ticks, last = train_data_["output"][-1])
    test_data = convert_data(test_data_, fontsize)


    data = test_data
    n = len(data)
    value = data["input"][:,3:-4]
    trigger = data["input"][:,-1]
    output = data["output"][:].ravel()

    X = np.arange(n)
    Y = np.ones(n)
    yticks = []
    ylabels = []
    
    S  = [ (0, output,     "magma",  1.0, "Output (M)"),
           (1, trigger,    "gray_r", 1.0, "Trigger (T)")]
    for index in range(value.shape[1]):
        label = "Value (V"+chr(ord("₁")+index)+")"
        S.append((8-index, value[:,index], "gray_r", 1.0, label))

    
    fig = plt.figure(figsize=(11,2))
    ax = plt.subplot(1,1,1, frameon=False)
    ax.tick_params(axis='y', which='both', length=0)
    

    yticks, ylabels = [], []
    for (index, values, cmap, alpha, label) in S:
        ax.scatter(X, index*Y, s=100, vmin=0, vmax=1, alpha=alpha,
                   edgecolor="None", c=values, cmap=cmap)
        ax.scatter(X, index*Y, s=100, edgecolor="k", facecolor="None",
                   linewidth=0.25)
        yticks.append(index)
        ylabels.append(label)

    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.5,8.5)
    ax.set_xlim(-0.5,len(data)+.5)

    plt.savefig("task-literal-1-gate.pdf")
    plt.show()

