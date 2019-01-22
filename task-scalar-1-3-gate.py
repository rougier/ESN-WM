# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from data import generate_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    
    n = 50
    np.random.seed(6)
    values = np.random.uniform(0, +1, n)

    ticks = np.random.uniform(0, 1, (n,1)) < 0.05
    data1 = generate_data(values, ticks)

    ticks = np.random.uniform(0, 1, (n,3)) < 0.05
    data3 = generate_data(values, ticks)

    cmap = "magma"
    S  = [
        ( 6, data1["input"][:,0],  cmap, 0.75, "Value (V)"),
        ( 5, data3["input"][:,1],  "gray_r",  1.00, "Trigger (T₁)"),
        ( 4, data3["output"][:,0], cmap, 0.75, "Output (M₁)"),
        ( 3, data3["input"][:,2],  "gray_r",  1.00, "Trigger (T₂)"),
        ( 2, data3["output"][:,1], cmap, 0.75, "Output (M₂)"),
        ( 1, data3["input"][:,3],  "gray_r",  1.00, "Trigger (T₃)"),
        ( 0, data3["output"][:,2], cmap, 0.75, "Output (M₃)"),

        (10, data1["input"][:,0],  cmap, 0.75, "Value (V)"),
        ( 9, data1["input"][:,1],  "gray_r",  1.00, "Trigger (T)"),
        ( 8, data1["output"][:,0], cmap, 0.75, "Output (M)") ]
    
    fig = plt.figure(figsize=(10,2.5))
    ax = plt.subplot(1,1,1, frameon=False)
    ax.tick_params(axis='y', which='both', length=0)
    
    X = np.arange(n)
    Y = np.ones(n)
    yticks = []
    ylabels = []
    for (index, V, cmap, alpha, label) in S:
        ax.scatter(X, index*Y, s=100, vmin=0, vmax=1, alpha=alpha,
                   edgecolor="None", c=V, cmap=cmap)
        ax.scatter(X, index*Y, s=100, edgecolor="k", facecolor="None",
                   linewidth=0.5)
        yticks.append(index)
        ylabels.append(label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.5,10.5)

    ax.set_xticks([])
    ax.set_xlim(-0.5,n-0.5)


    plt.savefig("task-scalar-1-3.pdf")
    plt.show()
