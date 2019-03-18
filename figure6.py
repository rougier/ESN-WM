# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen, str_to_bmp, convert_data 
from model import generate_model, train_model, test_model
import sys
import os

if __name__ == '__main__':
    # Display
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 4

    directory = "data/results"
    # -------------------------------------------------------------------------
    # 1-1-1 scalar task
    task = "1-1-1-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 1
    print(task)
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
        error = np.sqrt(np.mean((model["output"] - test_data["output"])**2))
        print("Testing error : {0}".format(error))

    # Display
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

    # -------------------------------------------------------------------------
    # 1-3-3 scalar task
    # Random generator initialization
    task = "1-3-3-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 3
    print(task)
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
        error = np.sqrt(np.mean((model["output"] - test_data["output"])**2))
        print("Testing error : {0}".format(error))

    # Display
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data = test_data

    ax2 = plt.subplot(n_subplots, 1, 2)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(data["input"][:,0],  color='0.75', lw=1.0)

    X, Y = np.arange(len(data)), np.ones(len(data))
    for i in range(n_gate):
        C = np.zeros((len(data),4))
        r = eval("0x"+colors[i][1:3])
        g = eval("0x"+colors[i][3:5])
        b = eval("0x"+colors[i][5:7])
        C[:,0] = r/255
        C[:,1] = g/255
        C[:,2] = b/255
        C[:,3] = data["input"][:,1+i]
        ax2.scatter(X, -1.05*Y-0.04*i, s=1.5, facecolors=C, edgecolors=None)
        ax2.plot(data["output"][:,i],  color='0.75', lw=1.0)
        ax2.plot(model["output"][:,i], lw=1.5, zorder=10)

    ax2.text(-25, -1.05, "Ticks:",
             fontsize=8, transform=ax2.transData,
             horizontalalignment="right", verticalalignment="center")
    ax2.set_ylim(-1.25,1.25)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Input & Output")
    ax2.text(0.01, 0.9, "B",
             fontsize=16, fontweight="bold", transform=ax2.transAxes,
             horizontalalignment="left", verticalalignment="top")
    # -------------------------------------------------------------------------
    # 3-1-1 scalar task
    task = "3-1-1-scalar"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gates = 1
    n_values = 3
    print(task)
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
        error = np.sqrt(np.mean((model["output"] - test_data["output"])**2))
        print("Testing error : {0}".format(error))

    # Display
    data = test_data

    ax3 = plt.subplot(n_subplots, 1, 3)
    ax3.patch.set_alpha(1.0)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.plot(data["input"][:,1],  color='0.9', lw=1.0)
    ax3.plot(data["input"][:,2],  color='0.9', lw=1.0)
    ax3.plot(data["input"][:,0],  color='0.75', lw=1.0)
    ax3.plot(data["output"],  color='0.75', lw=1.0)
    ax3.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,n_values]
    ax3.scatter(X, -0.9*Y, s=1, facecolors=C, edgecolors=None)
    ax3.text(-25, -0.9, "Ticks:",
             fontsize=8, transform=ax3.transData,
             horizontalalignment="right", verticalalignment="center")
    ax3.set_ylim(-1.1,1.1)
    ax3.yaxis.tick_right()
    ax3.set_ylabel("Input & Output")
    ax3.text(0.01, 0.9, "C",
             fontsize=16, fontweight="bold", transform=ax3.transAxes,
             horizontalalignment="left", verticalalignment="top")


    # -------------------------------------------------------------------------
    # 1-1-1 digit task
    task = "1-1-1-digit"
    files = ["{:s}/{:s}_{:s}.npy".format(directory, task, var) for var in ["desired", "model", "state"]]
    n_gate = 1
    size = 11
    print(task)
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
        error = np.sqrt(np.mean((model["output"] - test_data["output"])**2))
        print("Testing error : {0}".format(error))


    # Display
    data = test_data

    ax4 = plt.subplot(n_subplots,1,4)

    Z = test_data["input"][:, :-1].T
    ax4.imshow(Z, interpolation='nearest', origin='upper', cmap="gray_r",
               extent=[0,len(data),1.225,1.4], aspect='auto')

    
    ax4.tick_params(axis='both', which='major', labelsize=8)
    ax4.plot(data["output"],  color='0.75', lw=1.0)
    ax4.plot(model["output"], color='0.00', lw=1.5)
    X, Y = np.arange(len(data)), np.ones(len(data))
    C = np.zeros((len(data),4))
    C[:,3] = data["input"][:,-1]

    ax4.scatter(X, 1.1*Y, s=1, facecolors=C, edgecolors=None)
    ax4.text(-3, 1.1, "Ticks:",
             fontsize=8, transform=ax4.transData,
             horizontalalignment="right", verticalalignment="center")

    ax4.yaxis.tick_right()
    ax4.set_ylabel("Input & Output")
    ax4.text(0.01, 0.95, "D",
             fontsize=16, fontweight="bold", transform=ax4.transAxes,
             horizontalalignment="left", verticalalignment="top")

    ax4.set_ylim(-0.1,1.5)
    ax4.set_xlim(-15, 315)
    ax4.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])


    plt.tight_layout()
    plt.savefig("figure6.pdf")
    plt.show()
