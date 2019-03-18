# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
from data import generate_data, smoothen, str_to_bmp


def convert_data(data_, size):
    values = (data_["input"][:, 0]).astype(int)
    text = [chr(ord("0")+i) for i in values]
    Z, I = str_to_bmp(text, size = size)
    data = np.zeros(Z.shape[1], dtype = [ ("input",  float, (1 + Z.shape[0],)),
                                          ("output", float, (    1,))])
    data["input"][:, :-1] = Z.T
    n = Z.shape[1]//len(text)
    data["input"][:,-1] = np.repeat(data_["input"][:, 1], n)
    data["output"][:, 0] = np.repeat(data_["output"], n) / 10
    return data


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
    import matplotlib.pyplot as plt
    
    n = 54

    np.random.seed(1)
    values = np.random.uniform(0, +1, n)
    ticks = np.random.uniform(0, 1, (n,1)) < 0.1
    data_scalar_1_1_1 = generate_data(values, ticks)

    np.random.seed(2)
    values = np.random.uniform(0, +1, n)
    ticks = np.random.uniform(0, 1, (n,3)) < 0.1
    data_scalar_1_3_3 = generate_data(values, ticks)

    np.random.seed(3)
    values = np.random.uniform(0, +1, (n,3))
    ticks = np.random.uniform(0, 1, (n,1)) < 0.1
    data_scalar_3_1_1 = generate_data(values, ticks)


    np.random.seed(5)
    values = np.random.randint(0, 10, 9)
    ticks = np.random.uniform(0, 1, (9, 1)) < 0.1
    data = convert_data(generate_data(values, ticks), 11)
    n = len(data)
    value = data["input"][:,3:-4]
    trigger = data["input"][:,-1]
    output = data["output"][:].ravel()

    X = np.arange(n)
    Y = np.ones(n)
    yticks = []
    ylabels = []

    y0 = 18
    y1 = 8
    y2 = 0
    y3 = -12
    cmap1 = "magma"
    cmap2 = "gray_r"
    
    S  = [
        ( y0+2, data_scalar_1_1_1["input"][:,0],  cmap1, 0.75, "Value (V)"),
        ( y0+1, data_scalar_1_1_1["input"][:,1],  cmap2, 1.00, "Trigger (T)"),
        ( y0+0, data_scalar_1_1_1["output"][:,0], cmap1, 0.75, "Output (M)"),

        ( y1+6, data_scalar_1_3_3["input"][:,0],  cmap1, 0.75, "Value (V)"),
        ( y1+5, data_scalar_1_3_3["input"][:,1],  cmap2, 1.00, "Trigger (T₁)"),
        ( y1+4, data_scalar_1_3_3["output"][:,0], cmap1, 0.75, "Output (M₁)"),
        ( y1+3, data_scalar_1_3_3["input"][:,2],  cmap2, 1.00, "Trigger (T₂)"),
        ( y1+2, data_scalar_1_3_3["output"][:,1], cmap1, 0.75, "Output (M₂)"),
        ( y1+1, data_scalar_1_3_3["input"][:,3],  cmap2, 1.00, "Trigger (T₃)"),
        ( y1+0, data_scalar_1_3_3["output"][:,2], cmap1, 0.75, "Output (M₃)"),


        ( y2+4, data_scalar_3_1_1["input"][:,0],  cmap1, 0.75, "Value (V₁)"),
        ( y2+3, data_scalar_3_1_1["input"][:,1],  cmap1, 0.75, "Value (V₂)"),
        ( y2+2, data_scalar_3_1_1["input"][:,2],  cmap1, 0.75, "Value (V₃)"),
        ( y2+1, data_scalar_3_1_1["input"][:,3],  cmap2, 1.00, "Trigger (T₁)"),
        ( y2+0, data_scalar_3_1_1["output"][:,0], cmap1, 0.75, "Output (M₁)")

    ]

    S.append((y3+0, output,     "magma",  1.0, "Output (M)"))
    S.append((y3+1, trigger,    "gray_r", 1.0, "Trigger (T)"))
    for index in range(value.shape[1]):
        label = "Value (V"+chr(ord("₁")+index)+")"
        S.append((y3+8-index, value[:,index], "gray_r", 1.0, label))

    
    fig = plt.figure(figsize=(10,7))
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
    ax.set_ylim(-12.5,20.5)

    ax.set_xticks([])
    ax.set_xlim(-0.5,n-0.5)


    ax.text(-7, y0+3, "A: 1-value 1-gate scalar task", fontsize=12, va="bottom", weight="bold")
    ax.text(-7, y1+7, "B: 1-value 3-gate scalar task", fontsize=12, va="bottom", weight="bold")
    ax.text(-7, y2+5, "C: 3-value 1-gate scalar task", fontsize=12, va="bottom", weight="bold")
    ax.text(-7, y3+9, "D: 1-value 1-gate digit task", fontsize=12, va="bottom", weight="bold")

    plt.savefig("figure2.pdf")
    plt.show()
