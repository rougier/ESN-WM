# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
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
    model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5,
                           radius=0.01, scaling=0.25, leak=1.0, noise=0.0001)

    # Training data
    n = 10000
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
    internals_init = np.zeros((n_value,model["shape"][1]))
    internals_end = np.zeros((n_value,model["shape"][1]))
    
    for i in tqdm.trange(n_value):
        output = vmin + (vmax-vmin)*(i/(n_value-1)) * np.ones(1)
        internals = np.random.uniform(-0.5,+0.5,1000)
        for j in range(n_epoch):
            outputs[i,j] = output
            internals_ = np.tanh((np.dot(model["W_rc"], internals) +
                                  np.dot(model["W_fb"], output)))
            internals = (1-model["leak"])*internals + model["leak"]*internals_
            output = np.dot(model["W_out"], internals)
 #           if j == 1:
 #               internals_init[i] = internals
 #       internals_end[i] = internals
    
            
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
    """
    ax = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    ax.plot(np.linalg.norm(internals_init - internals_end, axis = 1),
            outputs[:,0], color="k")
    for label in ax.get_yticklabels():
        label.set_visible(False)
    ax.axhline(+1.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.axhline(-1.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.axvline( 0.0, color="0.75", linewidth=0.75, zorder=-10)
    ax.text(0.125, 0.02, "C", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=24, weight="bold")
    plt.plot([0,0], [-1,1], lw="1.5", color="red", zorder=-10)
    """
        
    plt.tight_layout()
    plt.savefig("WM-stability.pdf")
    plt.show()



