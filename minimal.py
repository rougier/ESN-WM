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
    np.random.seed(1)
    
    # Testing data
    n_gate = 1
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    data = generate_data(values, ticks)
    

    y = data["input"][0,0]
    output = []
    states = np.zeros((8,len(data)))
    scale = 0.001
    norm = np.tanh(1-scale)-np.tanh(1+scale)+np.tanh(scale)-np.tanh(-scale)
    
    for i in range(len(data)):
        x,t = data["input"][i]

        y0 = states[0,i] = +np.tanh(t-scale*x)/norm
        y1 = states[1,i] = -np.tanh(t+scale*x)/norm
        y2 = states[2,i] = +np.tanh(  scale*x)/norm
        y3 = states[3,i] = -np.tanh( -scale*x)/norm
        y_ff = y0+y1+y2+y3

        y0 = states[4,i] = +np.tanh(t+scale*y)/norm
        y1 = states[5,i] = -np.tanh(t-scale*y)/norm
        y2 = states[6,i] = -np.tanh(1+scale*y)/norm
        y3 = states[7,i] = +np.tanh(1-scale*y)/norm
        y_fb = y0+y1+y2+y3
        
        y = y_ff + y_fb
        output.append(y)

    model = {"output" : np.array(output).reshape(len(output),1),
             "state" : np.array(states) }
    error = np.sqrt(np.mean((model["output"] - data["output"])**2))
    print("Error: {0}".format(error))


    # Display
    fig = plt.figure(figsize=(14,6))
    fig.patch.set_alpha(0.0)
    n_subplots = 3

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

    
    ax3 = plt.subplot(n_subplots, 1, 3, sharex=ax1)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    n = 8
    for i in range(n):
        ax3.plot(model["state"][i], color='k', alpha=.25, lw=.5)
    ax3.yaxis.tick_right()
    ax3.set_ylim(-1.1, +1.1)
    ax3.set_ylabel("Internal units (n={0})".format(n))
    ax3.text(0.01, 0.9, "C",
             fontsize=16, fontweight="bold", transform=ax3.transAxes,
             horizontalalignment="left", verticalalignment="top")
   
    plt.tight_layout()
    plt.show()
