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


def asses_param(param_name, values, n_res, gen_data, gen_once, test_errors, stop, ext = ""):
    files = ["data/{:s}{:s}_train.npy".format(param_name, ext), "data/{:s}{:s}_test.npy".format(param_name, ext)]
    np.save("data/{:s}{:s}_values.npy".format(param_name, ext), values)
    if not np.all([os.path.exists(path) for path in files]):
        if os.path.exists("tmp/{:s}{:s}_test.npy".format(param_name,ext)):
            a = np.load("tmp/{:s}{:s}_test.npy".format(param_name,ext))
            test_errors[:a.shape[0]] = a 
            print(param_name, "loaded")
        n = len(values)
        train_errors = np.nan*np.empty((n,n_res))
        for i, v in enumerate(values):
            if gen_once:
                param, train_data, test_data = gen_data(v)
            for j in range(n_res):
                if not gen_once:
                    param, train_data, test_data = gen_data(v)
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
    print(param_name, "ended", stop.value)
    stop.value += 1

def draw(stop, delay, n_data, datas, base_shape = (3,3), first_letter = 0):
    print("DRAWING")
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
                ax.text(.9, 0.1, s[first_letter+i*n_fig[0]+j],
                         fontsize=16, fontweight="bold", transform=ax.transAxes,
                         horizontalalignment="left", verticalalignment="top")
                ax.axhline(1e-2, 0, 1, color = "red", linewidth=1.0, zorder = 0, linestyle = "--")
        fig.savefig("tmp/taskrobust.pdf")
        sleep(delay)
    for i in range(len(datas)):
        for j, (name, data, values, scale, xlim) in enumerate(datas[i]):
            ax = axs[i][j]
            for l in ax.lines:
                l.remove()
            if not xlim is None:
                ax.set_xlim(xlim)
            print(name, data.shape, values.shape)
            percentiles = np.percentile(data, p, axis = 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.fill_between(values, percentiles[0], percentiles[2], facecolor = "black", edgecolor = None, alpha = 0.1, zorder = 10)
            ax.plot(values, percentiles[1], linestyle = "-", linewidth=1.0, color = "black", zorder = 20)
            ax.text(.9, 0.1, s[first_letter+i*n_fig[0]+j],
                     fontsize=16, fontweight="bold", transform=ax.transAxes,
                     horizontalalignment="left", verticalalignment="top")
            ax.axhline(1e-2, 0, 1, color = "red", linewidth=1.0, zorder = 0, linestyle = "--")
    fig.savefig("figure7task.pdf")

def gen_discrete(n_train, n_test, base_param, k):
    n_gate = 1
    values = np.random.uniform(-1, +1, n_train)
    ticks = np.random.uniform(0, 1, (n_train, n_gate)) < 0.01
    discrete_values = np.random.uniform(-1, 1, k)
    idx = np.where(ticks == 1)[0]
    values[idx] = np.random.choice(discrete_values, len(idx))
    train_data = generate_data(values, ticks)

    values = smoothen(np.random.uniform(-1, +1, n_test))
    ticks = np.random.uniform(0, 1, (n_test, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])
    return base_param, train_data, test_data

def gen_gate(n_train, n_test, base_param, k):
    n_gate = k
    values = np.random.uniform(-1, +1, n_train)
    ticks = np.random.uniform(0, 1, (n_train, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    values = smoothen(np.random.uniform(-1, +1, n_test))
    ticks = np.random.uniform(0, 1, (n_test, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    param = base_param.copy()
    param["shape"] = (1+n_gate,param["shape"][1],n_gate)

    return param, train_data, test_data

def gen_value(n_train, n_test, base_param, k):
    n_gate = 1
    values = np.random.uniform(-1, +1, (n_train, k))
    ticks = np.random.uniform(0, 1, (n_train, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    values = np.empty((n_test, k))
    for i in range(k):
        values[:, i] = smoothen(np.random.uniform(-1, +1, n_test))
    ticks = np.random.uniform(0, 1, (n_test, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    param = base_param.copy()
    param["shape"] = (k+n_gate,param["shape"][1],n_gate)

    return param, train_data, test_data

def gen_bound(n_train, n_test, base_param, k):
    n_gate = 1
    values = np.random.uniform(-k, +k, n_train)
    ticks = np.random.uniform(0, 1, (n_train, n_gate)) < 0.01
    train_data = generate_data(values, ticks)

    values = smoothen(np.random.uniform(-1, +1, n_test))
    ticks = np.random.uniform(0, 1, (n_test, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    return base_param, train_data, test_data


def gen_trigger(n_train, n_test, base_param, k):
    n_gate = 1
    values = np.random.uniform(-1, +1, n_train)
    ticks_interval = np.random.randint(1, k+1, size =(n_train))
    ticks_time = np.cumsum(ticks_interval)
    i_max = np.max(np.where(ticks_time < n_train)[0])
    ticks = np.zeros((n_train,))
    ticks[ticks_time[:i_max]] = 1
    train_data = generate_data(values, ticks)

    values = smoothen(np.random.uniform(-1, +1, n_test))
    ticks = np.random.uniform(0, 1, (n_test, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    return base_param, train_data, test_data



if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(1)
    


    n_res = 20
    n_sample = 20
    n_train = 25000
    n_test = 2500
    
    base_param = {"shape":(2,1000,1), "sparsity":0.5, "radius":0.1, "scaling":(1.0,1.0), "leak":1.0, "noise":(0, 1e-4, 0), "seed":None}

    noise_values = np.load("data/noise_values.npy")[:,1]
    noise_test_errors = np.load("data/noise_test.npy")
    discrete_values = np.arange(1, 13, dtype = np.int64)
    discrete_test_errors = sharedmem.empty((len(discrete_values),n_res))
    discrete_test_errors[...] = np.nan
    gate_values = np.arange(1, 13, dtype = np.int64)
    gate_test_errors = sharedmem.empty((len(gate_values),n_res))
    gate_test_errors[...] = np.nan
    value_values = np.arange(1, 13, dtype = np.int64)
    value_test_errors = sharedmem.empty((len(value_values),n_res))
    value_test_errors[...] = np.nan
    bound_values = np.logspace(-2, 0, n_sample)
    bound_test_errors = sharedmem.empty((len(bound_values),n_res))
    bound_test_errors[...] = np.nan
    trigger_values = np.logspace(1, 3, n_sample, dtype = np.int64)
    trigger_test_errors = sharedmem.empty((len(trigger_values),n_res))
    trigger_test_errors[...] = np.nan

    stop = Value("i", 0)

    data_noise = "noise", noise_test_errors, noise_values, "log", [10**(-8.1), 10**(0.1)]
    data_discrete = "# discrete values", discrete_test_errors, discrete_values, "linear", [0, 13]
    data_gate = "# gate", gate_test_errors, gate_values, "linear", [0, 13]
    data_value = "# input", value_test_errors, value_values, "linear", [0, 13]
    data_bound = "value bound", bound_test_errors, bound_values, "log", [10**(-2.1), 10**(0.1)]
    data_trigger = "trigger range", trigger_test_errors, trigger_values, "log", [10**(0.9), 10**(3.1)]
    
    datas = [[data_noise, data_discrete, data_gate], [data_value, data_bound, data_trigger]]

    f_discrete = lambda: asses_param("discrete", discrete_values, n_res, lambda k: gen_discrete(n_train, n_test, base_param, k), False, discrete_test_errors, stop)
    f_gate = lambda: asses_param("gate", gate_values, n_res, lambda k: gen_gate(n_train, n_test, base_param, k), True, gate_test_errors, stop)
    f_value = lambda: asses_param("value", value_values, n_res, lambda k: gen_value(n_train, n_test, base_param, k), True, value_test_errors, stop)
    f_bound = lambda: asses_param("bound", bound_values, n_res, lambda k: gen_bound(n_train, n_test, base_param, k), True, bound_test_errors, stop)
    f_trigger = lambda: asses_param("trigger", trigger_values, n_res, lambda k: gen_trigger(n_train, n_test, base_param, k), True, trigger_test_errors, stop)
    f_draw = lambda: draw(stop, 6000, 1, datas, first_letter = 6)
    
    launch_parallel(f_discrete, f_gate, f_bound, f_value, f_trigger, f_draw)
