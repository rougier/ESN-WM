# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from data import generate_data, smoothen
from model import generate_model, train_model, test_model
from multiprocessing import Process, Lock, Value, Pool, Array
import sharedmem
from time import sleep
import string

def launch_parallel(*f_s):
    p_s = []
    for f in f_s:
        p = Process(target=f)
        p.start()
        p_s.append(p)
    for p in p_s:
        p.join()


def asses_param(param_name, values, n_res, train_data, test_data, base_param, test_errors, stop, ext = ""):
    files = ["data/{:s}{:s}_train.npy".format(param_name, ext), "data/{:s}{:s}_test.npy".format(param_name, ext)]
    np.save("data/{:s}{:s}_values.npy".format(param_name, ext), values)
    if not np.all([os.path.exists(path) for path in files]):
        if os.path.exists("tmp/{:s}{:s}_test.npy".format(param_name,ext)):
            a = np.load("tmp/{:s}{:s}_test.npy".format(param_name,ext))
            test_errors[:a.shape[0]] = a 
            print(param_name, "loaded")
        param = base_param.copy()
        n = len(values)
        train_errors = np.nan*np.empty((n,n_res))
        for i, v in enumerate(values):
            param[param_name] = v
            for j in range(n_res):
                if np.isnan(test_errors[i, j]):
                    print("{:s}{:s} {:d}/{:d}, {:d}/{:d}".format(param_name,ext, i,n,j,n_res))
                    #print(param)
                    model = generate_model(**param)

                    train_errors[i,j] = train_model(model, train_data)
                    test_errors[i,j] = test_model(model, test_data)
            #print(param_name, test_errors)
            np.save("tmp/{:s}{:s}_train.npy".format(param_name,ext), train_errors[:i+1])
            np.save("tmp/{:s}{:s}_test.npy".format(param_name,ext), test_errors[:i+1])
        np.save("data/{:s}{:s}_train.npy".format(param_name,ext), train_errors)
        np.save("data/{:s}{:s}_test.npy".format(param_name,ext), test_errors)
        #print(param_name, test_errors)
    else:
        train_errors = np.load("data/{:s}{:s}_train.npy".format(param_name,ext))
        test_errors[...] = np.load("data/{:s}{:s}_test.npy".format(param_name,ext))
        #print(param_name, test_errors)
    stop.value += 1

def draw(stop, delay, n_data, datas, base_shape = (3,3)):
    s = string.ascii_uppercase
    p = [5,50,95]
    n_fig = max([len(data) for data in datas]),len(datas)
    fig = plt.figure(figsize= (base_shape[0]*n_fig[0], base_shape[1]*n_fig[1]))
    fig.patch.set_alpha(0.0)
    fig.set_tight_layout(True)
    axs = []
    n_subplots = 0
    for i in range(len(datas)):
        axs.append([])
        for j, (name, data, values, scale, xlim) in enumerate(datas[i]):
            ax = fig.add_subplot(n_fig[1], n_fig[0], i*n_fig[0]+j+1)
            ax.set_xlabel(name)
            if type(scale) != str:
                ax.set_xscale(scale[0], linthreshx = scale[1])
            else:
                ax.set_xscale(scale)
            ax.set_yscale("log")
            ax.set_ylim([1e-4, 1e0])
            ax.set_ylabel("Error")
            axs[i].append(ax)

    while stop.value<n_data:
        for i in range(len(datas)):
            for j, (name, data, values, scale, xlim) in enumerate(datas[i]):
                ax = axs[i][j]
                for l in ax.lines:
                    l.remove()
                if not xlim is None:
                    ax.set_xlim(xlim)
                percentiles = np.percentile(data, p, axis = 1)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.fill_between(values, percentiles[0], percentiles[2], facecolor = "black", edgecolor = None, alpha = 0.1, zorder = 10)
                ax.plot(values, percentiles[1], linestyle = "-", linewidth=1.0, color = "black", zorder = 20)
                ax.text(.9, 0.1, s[i*n_fig[0]+j],
                         fontsize=16, fontweight="bold", transform=ax.transAxes,
                         horizontalalignment="left", verticalalignment="top")
                ax.axhline(1e-2, 0, 1, color = "red", linewidth=1.0, zorder = 0, linestyle = "--")
        fig.savefig("tmp/hprobust.pdf")
        sleep(delay)
    for i in range(len(datas)):
        for j, (name, data, values, scale, xlim) in enumerate(datas[i]):
            ax = axs[i][j]
            for l in ax.lines:
                l.remove()
            if not xlim is None:
                ax.set_xlim(xlim)
            percentiles = np.percentile(data, p, axis = 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.fill_between(values, percentiles[0], percentiles[2], facecolor = "black", edgecolor = None, alpha = 0.1, zorder = 10)
            ax.plot(values, percentiles[1], linestyle = "-", linewidth=1.0, color = "black", zorder = 20)
            ax.text(.9, 0.1, s[i*n_fig[0]+j],
                     fontsize=16, fontweight="bold", transform=ax.transAxes,
                     horizontalalignment="left", verticalalignment="top")
            ax.axhline(1e-2, 0, 1, color = "red", linewidth=1.0, zorder = 0, linestyle = "--")
    fig.savefig("figure7hp.pdf")



if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(1)
    
    # Build memoryticks
    n_gate = 1

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


    base_param = {"shape":(1+n_gate,1000,n_gate), "sparsity":0.5, "radius":0.1, "scaling":(1.0,1.0), "leak":1.0, "noise":(0, 1e-4, 0), "seed":None}
    n_res = 20
    n_sample = 20

    noise_values = np.stack([np.zeros(n_sample), np.logspace(-8, 0, n_sample), np.zeros(n_sample)], axis = 1)
    noise_test_errors = sharedmem.empty((len(noise_values),n_res))
    noise_test_errors[...] = np.nan
    radius_values = np.concatenate([[0], np.logspace(-2, 1, n_sample)])
    radius_test_errors = sharedmem.empty((len(radius_values),n_res))
    radius_test_errors[...] = np.nan
    sparsity_values = np.logspace(-2, 0, n_sample)
    sparsity_test_errors = sharedmem.empty((len(sparsity_values),n_res))
    sparsity_test_errors[...] = np.nan
    leak_values = np.logspace(-2, 0, n_sample)
    leak_test_errors = sharedmem.empty((len(sparsity_values),n_res))
    leak_test_errors[...] = np.nan
    shape_values = np.stack([1+n_gate+np.zeros(n_sample, dtype = np.int64), np.logspace(1, 3, n_sample, dtype = np.int64), n_gate+np.zeros(n_sample, dtype = np.int64)], axis = 1)
    shape_test_errors = sharedmem.empty((len(shape_values),n_res))
    shape_test_errors[...] = np.nan
    fb_scaling_values = np.stack([base_param["scaling"][0]*np.ones(n_sample), np.logspace(-2, 1, n_sample)], axis = 1)
    fb_scaling_test_errors = sharedmem.empty((len(fb_scaling_values),n_res))
    fb_scaling_test_errors[...] = np.nan
    in_scaling_values = np.stack([np.logspace(-2, 1, n_sample), base_param["scaling"][1]*np.ones(n_sample)], axis = 1)
    in_scaling_test_errors = sharedmem.empty((len(in_scaling_values),n_res))
    in_scaling_test_errors[...] = np.nan


    stop = Value("i", 0)

    
    data_leak = "leak", leak_test_errors, leak_values, "log", [10**(-2.1), 10**(0.1)]
    data_radius = "radius", radius_test_errors, radius_values, ("symlog", 0.01), [-0.001, 10**(1.1)]
    data_n_unit = "# unit", shape_test_errors, shape_values[:,1], "log", [10**(0.9), 10**(3.1)]
    data_sparsity = "sparsity", sparsity_test_errors, sparsity_values, "log", [10**(-2.1), 10**(0.1)]
    data_inscaling = "input scaling", in_scaling_test_errors, in_scaling_values[:,0], "log", [10**(-2.1), 10**(1.1)]
    data_fbscaling = "feedback scaling", fb_scaling_test_errors, fb_scaling_values[:,1], "log", [10**(-2.1), 10**(1.1)]
    """
    data_leak = "leak", leak_test_errors, leak_values, "linear", [-0.1, 1.1]
    data_radius = "radius", radius_test_errors, radius_values, ("linear", 0.01), [-0.1, 1.1]
    data_n_unit = "# unit", shape_test_errors, shape_values[:,1], "linear", [0., 1010]
    data_sparsity = "sparsity", sparsity_test_errors, sparsity_values, "linear", [-0.1, 1.1]
    data_inscaling = "input scaling", in_scaling_test_errors, in_scaling_values[:,0], "linear", [-0.1, 1.1]
    data_fbscaling = "feedback scaling", fb_scaling_test_errors, fb_scaling_values[:,1], "linear", [-0.1, 1.1]
    """
    datas = [[data_radius, data_leak, data_n_unit],[data_sparsity, data_inscaling, data_fbscaling]]

    datas = [[data_radius, data_leak, data_n_unit],[data_sparsity, data_inscaling, data_fbscaling]]

    f_noise = lambda: asses_param("noise", noise_values, n_res, train_data, test_data, base_param, noise_test_errors, stop)
    f_radius = lambda: asses_param("radius", radius_values, n_res, train_data, test_data, base_param, radius_test_errors, stop)
    f_sparsity = lambda: asses_param("sparsity", sparsity_values, n_res, train_data, test_data, base_param, sparsity_test_errors, stop)
    f_leak = lambda: asses_param("leak", leak_values, n_res, train_data, test_data, base_param, leak_test_errors, stop)
    f_shape = lambda: asses_param("shape", shape_values, n_res, train_data, test_data, base_param, shape_test_errors, stop)
    f_fb_scaling = lambda: asses_param("scaling", fb_scaling_values, n_res, train_data, test_data, base_param, fb_scaling_test_errors, stop, ext = "fb")
    f_in_scaling = lambda: asses_param("scaling", in_scaling_values, n_res, train_data, test_data, base_param, in_scaling_test_errors, stop, ext = "in")
    f_draw = lambda: draw(stop, 600, 7, datas)
    
    launch_parallel(f_noise, f_radius, f_sparsity, f_leak, f_shape, f_fb_scaling, f_in_scaling, f_draw)
"""
"""