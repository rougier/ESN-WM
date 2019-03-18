# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import smoothen
from model import generate_model, train_model, test_model


def generate_data(values, gates, last=None):
    """
    This function generates output data for a gated working memory task:

    Considering an input signal S(t) and a gate signal T(t), the output
    signal O(t) is defined as: O(t) = S(tᵢ) where i = argmax(T(t) = 1).

    values : np.array
        Input signal(s) as one (or several) sequence(s) of random float

    gates : np.array
        Gating signal(s) as one (or several) sequence(s) of 0 and 1
    """


    values = np.array(values)
    if len(values.shape) == 1:
        values = values.reshape(len(values), 1)
    n_values = values.shape[1]
        
    gates = np.array(gates)
    if len(gates.shape) == 1:
        gates = gates.reshape(len(gates), 1)
    n_gates = gates.shape[1]

    size = len(values)
    
    data = np.zeros(size, dtype = [ ("input",  float, (n_values + n_gates,)),
                                    ("output", float, (           n_gates,))])
    # Input signals
    data["input"][:, 0:n_values ] = values
    data["input"][:, n_values: ] = gates


    wm = np.zeros(n_gates)
    # If no last activity set gate=1 at time t=0
    if last is None:
        wm[:] = data["input"][0, 0]
        data["input"][0, 1:] = 1
    else:
        wm[:] = last

    # Output value(s) according to gates
    for i in range(size):
        for j in range(n_gates):
            # Output at time of gate is not changed
            # data["output"][i,j] = wm[j]
            if data["input"][i,n_values+j] > 0:
                wm[j] = data["input"][i,0]
            # Output at time of gate is changed
            data["output"][i,j] = wm[j]

    return data


if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(1)

    # Parameters
    # ----------
    n_gates = 1
    n_values = 3
    sparsity = 0.5
    radius = 0.1
    scaling = 1.0
    leak = 1.0
    noise = 0.0, 1e-4, 0.0
    
    
    # Build memory
    model = generate_model(shape=(n_values+n_gates, 1000, n_gates),
                           sparsity=0.5, radius=0.1, scaling=1.0, leak=1.0,
                           noise=(0, 1e-4, 0))

    # Sparsify W_in & W_fb
    model["W_in"] *= np.random.uniform(0, 1, model["W_in"].shape) < sparsity
    model["W_fb"] *= np.random.uniform(0, 1, model["W_fb"].shape) < sparsity

    # Training data
    n = 50000
    
    values = np.random.uniform(-1, +1, (n,n_values))
    for i in range(n_values):
        values[:,i] = smoothen(values[:,i])
    gates = np.random.uniform(0, 1, (n, n_gates)) < 0.01
    train_data = generate_data(values, gates)

    # Testing data
    n = 2500
    values = np.random.uniform(-1, +1, (n,n_values))
    for i in range(n_values):
        values[:,i] = smoothen(values[:,i])
        
    gates = np.random.uniform(0, 1, (n, n_gates)) < 0.01
    test_data = generate_data(values, gates, last = train_data["output"][-1])

    print("Training")
    rmse_train = train_model(model, train_data)
    print("Training error : {0:.5f}".format(rmse_train))

    print("Testing")
    rmse_test = test_model(model, test_data)
    print("Testing error : {0:.5f}".format(rmse_test))


    # Display
    fig = plt.figure(figsize=(14,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 4

    data = test_data

    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.patch.set_alpha(1.0)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax1.plot(data["output"],  color='0.75', lw=1.0)
    ax1.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,n_values]
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
    ax3.set_ylim(-0.59, +0.59)
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

    # Display
    filename = "test-n-values.png"
    plt.savefig(filename) #, transparent=True)

    # OSX/iTerm specific
    from subprocess import call
    call(["imgcat", filename])

    plt.savefig("test-n-values.pdf")
    # plt.show()
