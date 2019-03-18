# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This script tests whether it is possible to have a better reading of the
# output after learning. The protocol is as follows:
#
# 1. Train the model using teacher forcing (-> Wout)
# 2. Run the model on the training data using Wout & 
#    record all the internal states of the reservoir
# 3. Test the model on the test data
# 4. Measure performance (test data) of a linear decoder using several
#    restricted (size) sets of reservoir units using training states
#    from step 2.
#
# Condition A: Sets vary in size and are random
# Condition B: Sets vary in size and use the less correlated units (/ output)
# -----------------------------------------------------------------------------
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen
from model import generate_model, train_model, test_model


if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(1)


    # Sizes to be tested
    sizes = []
    sizes.extend(np.arange(  1,   50,   1).tolist())
    sizes.extend(np.arange( 50,  100,  10).tolist())
    sizes.extend(np.arange(100,  500,  50).tolist())
    sizes.extend(np.arange(500, 1001, 100).tolist())

    errors_random_filename = "data/decoding-error-random.npy"
    errors_sorted_1_filename = "data/decoding-error-sorted_1.npy"
    errors_sorted_2_filename = "data/decoding-error-sorted_2.npy"

    # Don't recompute if things have been already saved
    if os.path.exists(errors_random_filename):
        errors_random = np.load(errors_random_filename)
        errors_sorted_1 = np.load(errors_sorted_1_filename)
        errors_sorted_2 = np.load(errors_sorted_2_filename)
    else:

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

        # Train model
        error = train_model(model, train_data)
        print("Training error : {0}".format(error))

        # Test model and collect internal states
        error = test_model(model, test_data)
        print("Testing error : {0}".format(error))
        testing_states = model["state"].copy()


        
        # Run model on training data to collect internal states
        error = test_model(model, train_data)
        print("Training error : {0}".format(error))
        training_states = model["state"].copy()

        # Find reservoir units correlation / output on training data
        from scipy.stats.stats import pearsonr
        n = len(model["state"])
        C = np.zeros(n)
        for i in range(n):
            C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())

        # Sort units from the less to the most correlated
        I = np.argsort(np.abs(C))

        errors_random = np.zeros((len(sizes),10))
        errors_sorted_1 = np.zeros((len(sizes),1))
        errors_sorted_2 = np.zeros((len(sizes),1))
        
        for i,size in tqdm.tqdm(enumerate(sizes)):
            for run in range(errors_random.shape[1]):

                # Random variable set of reservoir units
                outputs = train_data["output"]
                indices = np.random.choice(1000, size=size, replace=False)
                states = (training_states[indices]).T
                W_out = np.dot(np.linalg.pinv(states), outputs).T
            
                outputs = test_data["output"]
                states = (testing_states[indices]).T
                errors_random[i,run] = \
                    np.sqrt(np.mean((np.dot(states,W_out.T)-outputs)**2))

            # Sorted variable size subset of reservoir units
            outputs = train_data["output"]
            indices = I[:size]
            states = (training_states[indices]).T
            W_out = np.dot(np.linalg.pinv(states), outputs).T
        
            outputs = test_data["output"]
            states = (testing_states[indices]).T
            errors_sorted_1[i,0] = \
                np.sqrt(np.mean((np.dot(states,W_out.T)-outputs)**2))

            # Sorted variable size subset of reservoir units
            outputs = train_data["output"]
            indices = I[-size:]
            states = (training_states[indices]).T
            W_out = np.dot(np.linalg.pinv(states), outputs).T
        
            outputs = test_data["output"]
            states = (testing_states[indices]).T
            errors_sorted_2[i,0] = \
                np.sqrt(np.mean((np.dot(states,W_out.T)-outputs)**2))

        # np.save(errors_random_filename, errors_random)
        np.save(errors_sorted_1_filename, errors_sorted_1)
        np.save(errors_sorted_2_filename, errors_sorted_2)

            
    fig = plt.figure(figsize=(5,5))
    fig.patch.set_alpha(0.0)
    ax = plt.subplot(1, 1, 1)
    
    ax.plot(sizes, errors_random.mean(axis=1), lw=1.5, color="k",
            label = "Random")
    ax.fill_between(sizes, errors_random.mean(axis=1) + errors_random.std(axis=1),
                           errors_random.mean(axis=1) - errors_random.std(axis=1),
                           alpha=0.15, zorder=-10, facecolor="k", edgecolor="None")
    ax.plot(sizes, errors_sorted_1.mean(axis=1), lw=1.5,
            label = "Least correlated")

    ax.plot(sizes, errors_sorted_2.mean(axis=1), lw=1.5,
            label = "Most correlated")
    
    
    # ax.axhline(error, color=".75", lw=0.75, zorder=-20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.0)
    ax.set_xlabel("# units in the decoder")
    ax.set_ylabel("Mean error", labelpad=-10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig("figure10.pdf")
    plt.show()

