# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This script tests stability of the reservoir. The protocol is as follows:
#
# 1. Train the model using teacher forcing (-> Wout)
# 2. For output o in [-5,+5]
#      Choose a random reservoir state
#      Remove input and force output o at t=0
#      Iterate over 500 timesteps
#
# Expected behavior (after 500 timesteps):
#  For output(t=0) in [-1,1], no change in output
#  For output(t=0  > +1, output converges towards +1
#  For output(t=0) < -1, output converges towards -1
# -----------------------------------------------------------------------------
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen
from model import generate_model, train_model, test_model
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':
    
    # Random generator initialization
    np.random.seed(123)
    
    # Build memory
    n_gate = 1
    model = generate_model(shape=(1+n_gate,1000,n_gate),
                           sparsity=0.5,
                           radius=0.1,
                           scaling=0.25,
                           leak=1.0,
                           noise=0.0001)

    # Training data
    n = 25000
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    error = train_model(model, train_data)
    print("Training error : {0}".format(error))

    
    # Test model with random initial state and no input
    np.random.seed(123)
    n_value = 100
    n_epoch = 500
    vmin, vmax = -5, +5
    outputs = np.zeros((n_value,n_epoch))
    
    for i in tqdm.trange(n_value):
        output = vmin + (vmax-vmin)*(i/(n_value-1)) * np.ones(1)
        internals = np.random.uniform(-0.5,+0.5,1000)
        for j in range(n_epoch):
            outputs[i,j] = output
            internals_ = np.tanh((np.dot(model["W_rc"], internals) +
                                  np.dot(model["W_fb"], output)))
            internals = (1-model["leak"])*internals + model["leak"]*internals_
            output = np.dot(model["W_out"], internals)
            
            
    # Display results
    plt.figure(figsize=(10,5))

    ax = plt.subplot(1,1,1)
    for i in range(n_value):
        ax.plot(outputs[i], color="k", alpha=0.25, lw=0.5)
    ax.text(0.98, 0.02, "A", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=24, weight="bold")

        
    divider = make_axes_locatable(ax)
    ax = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    ax.plot(np.abs(outputs[:,-1] - outputs[:,0]),
            np.linspace(-5,+5,n_value), color="k")
    for label in ax.get_yticklabels():
        label.set_visible(False)
    ax.axhline(+1.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.axhline(-1.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.axvline( 0.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.text(0.125, 0.02, "B", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=24, weight="bold")
    plt.plot([0,0], [-1,1], lw="1.5", color="red", zorder=-10)

        
    plt.tight_layout()
    plt.savefig("analysis-stability.pdf")
    plt.show()



