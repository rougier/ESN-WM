# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen
from model import generate_model, train_model, test_model


if __name__ == '__main__':
    
    # Random generator initialization
    np.random.seed(123)
    
    # Testing data
    n_gate = 1
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    data = generate_data(values, ticks)

    y = data["input"][0,0]
    output = []
    states = np.zeros((3,len(data)))
    a = 1000
    b = .001
    
    for i in range(len(data)):
        v,t = data["input"][i]
        x0 = states[0,i] = b*v
        x1 = states[1,i] = b*v + a*t
        x2 = states[2,i] = a*t + b*y
        y = (np.tanh(x0) - np.tanh(x1) + np.tanh(x2))/b
        output.append(y)

    model = {"output" : np.array(output).reshape(len(output),1),
             "state" : np.array(states) }
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))
    print("Error: {0}".format(error))

    # Display
    fig = plt.figure(figsize=(14,6))
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


    for i in range(3):
        ax = plt.subplot(n_subplots, 1, 2+i, sharex=ax1)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_ylim(-0.001, +0.001)
        ax.yaxis.tick_right()
        ax.text(0.01, 0.9, chr(ord("B")+i),
             fontsize=16, fontweight="bold", transform=ax.transAxes,
             horizontalalignment="left", verticalalignment="top")
        ax.plot(model["state"][i,:], color='k', alpha=.5, lw=.5)
        ax.set_ylabel("Activity")
        ax.set_yticks([-0.001,0.001])
        ax.set_yticklabels(["$-10^{-3}$","$+10^{-3}$"])
    
   
    plt.tight_layout()
    plt.savefig("result-RM-1-gate.pdf")
    plt.show()
