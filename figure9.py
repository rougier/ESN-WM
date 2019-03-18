# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import generate_data, smoothen, str_to_bmp, convert_data 
from model import generate_model, train_model, test_model
import sys
import os

if __name__ == '__main__':

    # Display
    n_bin = 50
    width_curve = 10
    width_histogram = 2
    total_width = width_curve + width_histogram
    n_subplots = 4
    fig = plt.figure(figsize=(total_width,2*n_subplots))
    fig.patch.set_alpha(0.0)

    gs = gridspec.GridSpec(n_subplots, 2,
                           width_ratios=[width_curve, width_histogram])

    directory = "data/results"
    # -------------------------------------------------------------------------
    # 1-1-1 scalar task
    task = "1-1-1-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 1
    if not np.all([os.path.exists(f) for f in files]):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Random generator initialization
        np.random.seed(1)

        # Build memory
        model = generate_model(shape=(1+n_gate,1000,n_gate),
                               sparsity=0.5, radius=0.1, scaling=(1.0,1.0),
                               leak=1.0, noise=(0.0000, 0.0001, 0.0001))

        # Training data
        n = 25000 # 300000
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
        np.save(files[0], test_data)
        np.save(files[1], model["output"])
        np.save(files[2], model["state"])
    else:
        test_data = np.load(files[0])
        model = {}
        model["output"] = np.load(files[1])
        model["state"] = np.load(files[2])

    # Display
    data = test_data
    threshold = 0.99

    # Find the most correlated unit in the reservoir (during testing)
    from scipy.stats.stats import pearsonr
    n = len(model["state"])
    C = np.zeros(n)
    idx = np.where(data["input"][:, 1] == 0)[0]
    for i in range(n):
        C[i], p = pearsonr(model["state"][i, idx].ravel(), model["output"][idx].ravel())
        #C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    I = np.argsort(np.abs(C))

    ax1 = plt.subplot(gs[0,0])
    ax1.tick_params(axis='both', which='major', labelsize=8)
    #n = np.min(np.where(np.abs(C[I[::-1]])<threshold)[0])
    n = 20
    threshold = np.abs(C[I[-20]])
    for i in range(n):
        ax1.plot(model["state"][I[-1-i]], color='k', alpha=.25, lw=.5)
    ax1.plot(model["output"], color = "red", lw = 1.)
    ax1.yaxis.tick_right()
    # ax3.set_ylim(-0.25, +0.25)
    ax1.set_ylim(-1.1, +1.1)
    ax1.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    ax1.text(0.01, 0.9, "A",
             fontsize=16, fontweight="bold", transform=ax1.transAxes,
             horizontalalignment="left", verticalalignment="top")
    ax12 = plt.subplot(gs[0,1])
    ax12.tick_params(axis='both', which='major', labelsize=8)
    ax12.yaxis.tick_right()
    ax12.axvline(threshold, 0, 1, color = "black", linestyle = "--")
    ax12.axvline(-threshold, 0, 1, color = "black", linestyle = "--")
    ax12.hist(C, bins = n_bin, color = "0.5", range = [-1, 1])
    ax12.set_xlim([-1.1, 1.1])
    ax12.set_ylim([0, 130])

    # -------------------------------------------------------------------------
    # 1-3-3 scalar task
    # Random generator initialization
    task = "1-3-3-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 3
    if not np.all([os.path.exists(f) for f in files]):
        np.random.seed(1)

        # Build memory
        model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5,
                               radius=0.1, scaling=(1.0, 0.33), leak=1.0,
                               noise=(0.000, 0.0001, 0.000))

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
        np.save(files[0], test_data)
        np.save(files[1], model["output"])
        np.save(files[2], model["state"])
    else:
        test_data = np.load(files[0])
        model = {}
        model["output"] = np.load(files[1])
        model["state"] = np.load(files[2])

    # Display
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data = test_data
    threshold = 0.8

    from scipy.stats.stats import pearsonr
    n = len(model["state"])
    C = np.zeros((n, n_gate))
    idx = np.where(np.sum(data["input"][:, 1:], axis = 1) == 0)[0]
    for j in range(n_gate):
        for i in range(n):
            C[i,j], p = pearsonr(model["state"][i, idx].ravel(), model["output"][idx, j].ravel())
        #C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    I = np.empty_like(C, dtype = np.int)
    for j in range(n_gate):
        I[:,j] = np.argsort(np.abs(C[:,j]))
    ax2 = plt.subplot(gs[1,0])
    ax2.tick_params(axis='both', which='major', labelsize=8)
    #n = np.min(np.where(np.abs(C[I[::-1, 0], 0])<threshold)[0])
    n = 20
    threshold = np.abs(C[I[-20, 0], 0])
    for i in range(n):
        ax2.plot(model["state"][I[-1-i, 0]], color="k", alpha=.25, lw=.5)
    ax2.plot(model["output"][:,0], color = "red", lw = 1.)
    ax2.yaxis.tick_right()
    # ax3.set_ylim(-0.25, +0.25)
    ax2.set_ylim(-1.1, +1.1)
    ax2.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    ax2.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")
    ax22 = plt.subplot(gs[1,1])
    ax22.tick_params(axis='both', which='major', labelsize=8)
    ax22.yaxis.tick_right()
    ax22.axvline(threshold, 0, 1, color = "black", linestyle = "--")
    ax22.axvline(-threshold, 0, 1, color = "black", linestyle = "--")
    ax22.hist(C[:,0], bins = n_bin, color = "0.5", range = [-1, 1])
    ax22.set_xlim([-1.1, 1.1])
    ax22.set_ylim([0, 130])
    # -------------------------------------------------------------------------
    # 3-1-1 scalar task
    task = "3-1-1-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gates = 1
    n_values = 3
    if not np.all([os.path.exists(f) for f in files]):
        # Random generator initialization
        np.random.seed(1)
        
        # Build memory
        model = generate_model(shape=(n_values+n_gates, 1000, n_gates),
                               sparsity=0.5, radius=0.1, scaling=(1.0,1.0),
                               leak=1.0, noise=(0, 1e-4, 0))

        # Training data
        n = 25000
        
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

        rmse_train = train_model(model, train_data)
        print("Training error : {0:.5f}".format(rmse_train))

        rmse_test = test_model(model, test_data)
        print("Testing error : {0:.5f}".format(rmse_test))
        np.save(files[0], test_data)
        np.save(files[1], model["output"])
        np.save(files[2], model["state"])
    else:
        test_data = np.load(files[0])
        model = {}
        model["output"] = np.load(files[1])
        model["state"] = np.load(files[2])

    # Display
    data = test_data
    threshold = 0.8

    # Find the most correlated unit in the reservoir (during testing)
    from scipy.stats.stats import pearsonr
    n = len(model["state"])
    C = np.zeros(n)
    idx = np.where(data["input"][:, -1] == 0)[0]
    for i in range(n):
        C[i], p = pearsonr(model["state"][i, idx].ravel(), model["output"][idx].ravel())
        #C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    I = np.argsort(np.abs(C))

    ax3 = plt.subplot(gs[2,0])
    ax3.tick_params(axis='both', which='major', labelsize=8)
    #n = np.min(np.where(np.abs(C[I[::-1]])<threshold)[0])
    n = 20
    threshold = np.abs(C[I[-20]])
    for i in range(n):
        ax3.plot(model["state"][I[-1-i]], color='k', alpha=.25, lw=.5)
    ax3.plot(model["output"], color = "red", lw = 1.)
    ax3.yaxis.tick_right()
    # ax3.set_ylim(-0.25, +0.25)
    ax3.set_ylim(-1.1, +1.1)
    ax3.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    ax3.text(0.01, 0.9, "C",
             fontsize=16, fontweight="bold", transform=ax3.transAxes,
             horizontalalignment="left", verticalalignment="top")
    ax32 = plt.subplot(gs[2,1])
    ax32.tick_params(axis='both', which='major', labelsize=8)
    ax32.yaxis.tick_right()
    ax32.axvline(threshold, 0, 1, color = "black", linestyle = "--")
    ax32.axvline(-threshold, 0, 1, color = "black", linestyle = "--")
    ax32.hist(C, bins = n_bin, color = "0.5", range = [-1, 1])
    ax32.set_xlim([-1.1, 1.1])
    ax32.set_ylim([0, 130])


    # -------------------------------------------------------------------------
    # 1-1-1 digit task
    task = "1-1-1-digit"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 1
    size = 11
    if not np.all([os.path.exists(f) for f in files]):
        # Random generator initialization
        np.random.seed(1)

        # Training data
        n = 25000
        values = np.random.randint(0, 10, n)
        ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
        train_data_ = generate_data(values, ticks)
        train_data = convert_data(train_data_, size, noise = 0.)


        # Testing data
        n = 50
        values = np.random.randint(0, 10, n)
        ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.1
        test_data_ = generate_data(values, ticks)
        test_data = convert_data(test_data_, size, noise = 0.)

        # Model
        model = generate_model(shape=(train_data["input"].shape[1],1000,n_gate),
                               sparsity=0.5,
                               radius=0.1,
                               scaling=(1.0, 1.0),
                               leak=1.0,
                               noise=0.0001)
        
        error = train_model(model, train_data)
        print("Training error : {0}".format(error))
        
        error = test_model(model, test_data)
        print("Testing error : {0}".format(error))
        np.save(files[0], test_data)
        np.save(files[1], model["output"])
        np.save(files[2], model["state"])
    else:
        test_data = np.load(files[0])
        model = {}
        model["output"] = np.load(files[1])
        model["state"] = np.load(files[2])


    # Display
    data = test_data

    # Find the most correlated unit in the reservoir (during testing)
    from scipy.stats.stats import pearsonr
    n = len(model["state"])
    C = np.zeros(n)
    idx = np.where(data["input"][:, -1] == 0)[0]
    for i in range(n):
        C[i], p = pearsonr(model["state"][i, idx].ravel(), model["output"][idx].ravel())
        #C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    I = np.argsort(np.abs(C))


    ax4 = plt.subplot(gs[3,0])
    ax4.tick_params(axis='both', which='major', labelsize=8)
    n = 20
    threshold = np.abs(C[I[-20]])
    for i in range(n):
        ax4.plot(model["state"][I[-1-i]], color='k', alpha=.25, lw=.5)
    ax4.plot(model["output"], color = "red", lw = 1.)
    ax4.yaxis.tick_right()
    # ax3.set_ylim(-0.25, +0.25)
    ax4.set_ylim(-1.1, +1.1)
    ax4.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    ax4.text(0.01, 0.9, "D",
             fontsize=16, fontweight="bold", transform=ax4.transAxes,
             horizontalalignment="left", verticalalignment="top")
    ax42 = plt.subplot(gs[3,1])
    ax42.tick_params(axis='both', which='major', labelsize=8)
    ax42.yaxis.tick_right()
    ax42.axvline(threshold, 0, 1, color = "black", linestyle = "--")
    ax42.axvline(-threshold, 0, 1, color = "black", linestyle = "--")
    ax42.hist(C, bins = n_bin, color = "0.5", range = [-1, 1])
    ax42.set_xlim([-1.1, 1.1])
    ax42.set_ylim([0, 130])



    plt.tight_layout()
    plt.savefig("figure9.pdf")
    plt.show()
