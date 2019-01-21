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

    # Build memory
    n_gate = 3
    model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5, radius=0.01,
                        scaling=0.25, leak=1.0, noise=0.0001)

    # Training data
    n = 25000
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    # Testing data
    n = 2500
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    error = train_model(model, train_data)
    print("Training error : {0}".format(error))

    error = test_model(model, test_data)
    print("Testing error : {0}".format(error))


    # Display
    fig = plt.figure(figsize=(14,5))
    fig.patch.set_alpha(0.0)
    n_subplots = 2

    data = test_data

    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    # ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)

    X, Y = np.arange(len(data)), np.ones(len(data))
    for i in range(n_gate):
        C = np.zeros((len(data),4))
        C[:,3] = data["input"][:,1+i]
        ax1.scatter(X, -1.05*Y-0.04*i, s=1.5, facecolors=C, edgecolors=None)

    ax1.text(-25, -1.05, "Ticks:",
             fontsize=8, transform=ax1.transData,
             horizontalalignment="right", verticalalignment="center")
    ax1.set_ylim(-1.25,1.25)
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Input & Output")
    ax1.text(0.01, 0.9, "A",
             fontsize=16, fontweight="bold", transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")


    ax2 = plt.subplot(n_subplots, 1, 2, sharex=ax1)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(model["output"]-data["output"],  color='red', lw=1.0)
    #ax2.set_ylim(-0.011, +0.011)
    ax2.yaxis.tick_right()
    ax2.axhline(0, color='.75', lw=.5)
    ax2.set_ylabel("Output error")
    ax2.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")


    plt.tight_layout()
#    plt.savefig("working-memory.pdf")
    plt.show()
