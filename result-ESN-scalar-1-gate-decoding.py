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


if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(1)

    # Build memory
    n_gate = 1
    model = generate_model(shape=(1+n_gate,1000,n_gate),
                           sparsity=0.5, radius=0.1, scaling=1.0, leak=1.0,
                           noise=(0, 1e-4, 1e-4))

    # Training data
    n = 25000
    values = np.random.uniform(-1, +1, n)
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


    sizes = []
    sizes.extend(np.arange(  1,   50,   1).tolist())
    sizes.extend(np.arange( 50,  100,  10).tolist())
    sizes.extend(np.arange(100,  500,  50).tolist())
    sizes.extend(np.arange(500, 1001, 100).tolist())

    errors = np.zeros((len(sizes),10))
    outputs = test_data["output"]
    for i,size in tqdm.tqdm(enumerate(sizes)):
        for run in range(errors.shape[1]):
            indices = np.random.choice(1000, size=size, replace=False)
            states = (model["state"][indices]).T
            W_out = np.dot(np.linalg.pinv(states), outputs).T
            errors[i,run] = np.sqrt(np.mean((np.dot(states,W_out.T)-outputs)**2))

    print("Re-Testing error : {0}".format(errors[-1].mean()))
            
    fig = plt.figure(figsize=(8,6))
    fig.patch.set_alpha(0.0)
    ax = plt.subplot(1, 1, 1)
    
    ax.plot(sizes, errors.mean(axis=1), lw=1.5)
    ax.fill_between(sizes, errors.mean(axis=1) + errors.std(axis=1),
                            errors.mean(axis=1) - errors.std(axis=1),
                            alpha=0.25, zorder=-10)
    ax.axhline(error, color=".75", lw=0.75, zorder=-20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("# units used for linear decoding")
    ax.set_ylabel("Mean error")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig("result-ESN-scalar-1-gate-decoding.pdf")
    plt.show()

