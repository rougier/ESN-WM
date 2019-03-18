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
    n_gate = 1
    model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5, radius=0.1,
                        scaling=0.25, leak=1.0, noise=0.0001)

    # Training data
    n = 10000
    values = np.random.choice(np.linspace(-1,1,5), n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    # Testing data
    n = 2500
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])
    
    error = train_model(model, train_data)
    print("Training error : {0}".format(error))
    
    error = test_model(model, test_data)
    print("Testing error : {0}".format(error))

    
    # Display
    fig = plt.figure(figsize=(14,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 4

    data = test_data

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
    plt.savefig("result-ESN-scalar-1-gate-discrete.pdf")
    plt.show()
