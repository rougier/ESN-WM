# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Gated working memory with an echo state network
# Copyright (c) 2018 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# from data import generate_data, smoothen
from data_old_for_3dupl import generate_data, smoothen
# from model import generate_model, train_model, test_model
from model_with_output_noise import generate_model, train_model, test_model

#TODO: test de mettre que du bruit sur une seule WM-unit, ou sur deux sortie
# est-ce qu'il continue à apprendre ce qui semble être un line attractor ?
#TODO: perturber beaucoup une seule WM-unit pour voir si grâce aux autres elle revient sur la bonne valeur
#TODO: faire un feedback partiel où seulement deux (voir une) gate est réellement connectée au reservoir
#      afin de voir si en test les activités des gates non connectées par feedback vont divergé ou non.
#      (à comparer au cas tout connecté où ça diverge pas).
#TODO:Vu qu'on apprend en offline et que les sorties sont les mêmes, c'est normal qu'elles soient complètement égales.
#   --> soit dit en passant : il faudrait qu'elles soient chacune connectées à des sous-parties différentes du réservoir pour que ce ne soit pas le cas -> à   tester un jour pour voir les propriétés
#   La seule influence différente qu'elles ont, c'est d'avoir des poids de feedback différents, donc elles peuvent avoir une influence différente sur le reservoir (mais en moyenne ça doit pas changer grand chose)
#   --> là encore, on pourrait tester des feedback concernant seulement une partie du réseau.
#   --> et si ces feedback changent au cours du temps, peut être est-il possible d'avoir un mécanisme non-supervisé qui adapte la sortie automatiquement

def duplicate_outputs(data, dupl_gate):
    """
    Duplicate the WM-units with the same values

    data is organized as follows:
        data = np.zeros(size, dtype = [ ("input",  float, (1 + n_gate,)),
                                    ("output", float, (    n_gate,))])
    """

    newdata = np.zeros(data["input"].shape[0], dtype = [ ("input",  float, (1 + n_gate,)),
                                       ("output", float, (    n_gate + dupl_gate,))])
    newdata["input"] = data["input"]
    newdata["output"][:,0] = data["output"][:,0]
    newdata["output"][:,1] = data["output"][:,0]
    newdata["output"][:,2] = data["output"][:,0]

    return newdata

def test_attractor(model, gain):
    """
    test the model with no input by perturbing the states of the reservoir during
    a run of the model to see if it comes back to a line attractor or not
    """
    pass

if __name__ == '__main__':

    # Random generator initialization
    np.random.seed(123)

    # Build memory
    n_gate = 1
    # model = generate_model(shape=(1+n_gate,1000,n_gate), sparsity=0.5, radius=0.01,
    #                     scaling=0.25, leak=1.0, noise=0.0001)

    # model_dupl_3g
    noise = 0.00001
    # out_noise = 0.005 #noise
    # out_noise = 0.001 #noise
    out_noise_train = 0.001 #noise
    # out_noise_test = 0.001 #noise
    out_noise_test = 0.0 #noise
    dupl_gate = 2
    data_mult_factor = 30 # length data multiplicative factor #30 max bon tradeoff pour mac book xav
    # tick_mult_factor = 1 # multiplying the number of ticks
    model = generate_model(shape=(1+n_gate,1000,n_gate+dupl_gate), sparsity=0.5, radius=0.01,
    # model = generate_model(shape=(1+n_gate,1000,n_gate+dupl_gate), sparsity=0.1, radius=0.01,
                        # scaling=0.25, leak=1.0, noise=noise)
                        # scaling=0.5, leak=1.0, noise=noise)
                        # scaling=0.9, leak=0.5, noise=noise)
                        # scaling=1., leak=0.5, noise=noise)
                        # scaling=1.1, leak=0.5, noise=noise)
                        # scaling=1.25, leak=0.5, noise=noise)
                        scaling=1.5, leak=0.5, noise=noise)
                        # scaling=1.75, leak=0.5, noise=noise)
                        # scaling=2., leak=0.5, noise=noise)
                        # scaling=4., leak=0.5, noise=noise)
                        # scaling=8., leak=0.5, noise=noise)
                        # scaling=16., leak=0.5, noise=noise)

    # Training data
    # n = 10000
    n = 10000 * data_mult_factor
    values = np.random.uniform(-1, +1, n)
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    # ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01 * tick_mult_factor
    train_data = generate_data(values, ticks)

    n_train_sample = n

    # Testing data
    # n = 2000
    n = 2000 * 5
    values = smoothen(np.random.uniform(-1, +1, n))
    ticks = np.random.uniform(0, 1, (n, n_gate)) < 0.01
    test_data = generate_data(values, ticks, last = train_data["output"][-1])

    n_test_sample = n

    # duplicate data (from 1 output gate -> 3 output gates)
    train_data = duplicate_outputs(train_data, dupl_gate)
    test_data = duplicate_outputs(test_data, dupl_gate)


    # error = train_model(model, train_data)
    error = train_model(model, train_data, output_noise_scale=out_noise_train) # add output noise only in feedback reinserted in reservoir (not in teacher)
    print("Training error : {0}".format(error))
    training_error = error #add xav

    # error = test_model(model, test_data)
    error = test_model(model, test_data, output_noise_scale=out_noise_test) # add output noise only in feedback reinserted in reservoir (not in teacher)
    print("Testing error : {0}".format(error))
    testing_error = error #add xav


    # Display
    fig = plt.figure(figsize=(14,8))
    fig.patch.set_alpha(0.0)
    n_subplots = 4

    data = test_data

    # ax1 = plt.subplot(n_subplots, 1, 1)
    # ax1.tick_params(axis='both', which='major', labelsize=8)
    # ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    # ax1.plot(data["output"],  color='0.75', lw=1.0)
    # ax1.plot(model["output"], color='0.00', lw=1.5)
    # X, Y = np.arange(len(data)), np.ones(len(data))
    # C = np.zeros((len(data),4))
    # C[:,3] = data["input"][:,1]
    # ax1.scatter(X, -0.9*Y, s=1, facecolors=C, edgecolors=None)
    # ax1.text(-25, -0.9, "Ticks:",
    #          fontsize=8, transform=ax1.transData,
    #          horizontalalignment="right", verticalalignment="center")
    # ax1.set_ylim(-1.1,1.1)
    # ax1.yaxis.tick_right()
    # ax1.set_ylabel("Input & Output")
    # ax1.text(0.01, 0.9, "A",
    #          fontsize=16, fontweight="bold", transform=ax1.transAxes,
    #          horizontalalignment="left", verticalalignment="top")

    ax1 = plt.subplot(n_subplots, 1, 1)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    # ax1.plot(data["input"][:,0],  color='0.75', lw=1.0)
    # ax1.plot(data["output"],  color='0.75', lw=3.0)
    ax1.plot(data["output"],  color='0.75', lw=2.0)
    # ax1.plot(model["output"], color='0.00', lw=1.5)
    ax1.plot(model["output"][:,0], lw=0.5)
    ax1.plot(model["output"][:,1], lw=0.5)
    ax1.plot(model["output"][:,2], lw=0.5)

    X, Y = np.arange(len(data)), np.ones(len(data))
    for i in range(n_gate):
        C = np.zeros((len(data),4))
        C[:,3] = data["input"][:,1+i]
        ax1.scatter(X, -1.05*Y-0.04*i, s=1.5, facecolors=C, edgecolors=None)

        ax1.text(-25, -1.05, "Ticks:",
        fontsize=8, transform=ax1.transData,
        horizontalalignment="right", verticalalignment="center")
        ax1.set_ylim(-1.25,1.25)
        ax1.yaxis.tick_right()
        ax1.set_ylabel("Input & Output")
        ax1.text(0.01, 0.9, "A",
        fontsize=16, fontweight="bold", transform=ax1.transAxes,
        horizontalalignment="left", verticalalignment="top")

    # ax2 = plt.subplot(n_subplots, 1, 2, sharex=ax1)
    # ax2.tick_params(axis='both', which='major', labelsize=8)
    # ax2.plot(model["output"]-data["output"],  color='red', lw=1.0)
    # ax2.set_ylim(-0.011, +0.011)
    # ax2.yaxis.tick_right()
    # ax2.axhline(0, color='.75', lw=.5)
    # ax2.set_ylabel("Output error")
    # ax2.text(0.01, 0.9, "B",
    #          fontsize=16, fontweight="bold", transform=ax2.transAxes,
    #          horizontalalignment="left", verticalalignment="top")

    ax2 = plt.subplot(n_subplots, 1, 2, sharex=ax1)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.plot(model["output"]-data["output"],  color='red', lw=1.0)
    #ax2.set_ylim(-0.011, +0.011)
    ax2.yaxis.tick_right()
    ax2.axhline(0, color='.75', lw=.5)
    ax2.set_ylabel("Output error")
    ax2.text(0.01, 0.9, "B",
          fontsize=16, fontweight="bold", transform=ax2.transAxes,
          horizontalalignment="left", verticalalignment="top")

    ax3 = plt.subplot(n_subplots, 1, 3, sharex=ax1)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    # ax3.plot(model["output"]-data["output"],  color='red', lw=1.0)
    ax3.plot(model["output"][:,0]-model["output"][:,1],   lw=1.0)
    ax3.plot(model["output"][:,1]-model["output"][:,2],  lw=1.0)
    ax3.plot(model["output"][:,2]-model["output"][:,0],  lw=1.0)
    #ax3.set_ylim(-0.011, +0.011)
    ax3.yaxis.tick_right()
    ax3.axhline(0, color='.75', lw=.5)
    ax3.set_ylabel("WM-unit differences")
    ax3.text(0.01, 0.9, "B",
          fontsize=16, fontweight="bold", transform=ax3.transAxes,
          horizontalalignment="left", verticalalignment="top")

    ax1.set_title("Noise internal: {0}".format(noise)+" -- Noise output train: {0}".format(out_noise_train)+" -- Noise output test: {0}".format(out_noise_test))
    ax2.set_title("Training error : {0}".format(training_error)+" -- Training samples: {0}".format(n_train_sample))
    ax3.set_title("Testing error : {0}".format(testing_error)+" -- Testing samples: {0}".format(n_test_sample))

    # # Find the most correlated unit in the reservoir (during testing)
    # from scipy.stats.stats import pearsonr
    # n = len(model["state"])
    # C = np.zeros(n)
    # for i in range(n):
    #     C[i], p = pearsonr(model["state"][i].ravel(), model["output"].ravel())
    # I = np.argsort(np.abs(C))
    #
    #
    # ax3 = plt.subplot(n_subplots, 1, 3, sharex=ax1)
    # ax3.tick_params(axis='both', which='major', labelsize=8)
    # n = 20
    # for i in range(n):
    #     ax3.plot(model["state"][I[-1-i]], color='k', alpha=.25, lw=.5)
    # ax3.yaxis.tick_right()
    # ax3.set_ylim(-1.1, +1.1)
    # ax3.set_ylabel("Most correlated\n internal units (n={0})".format(n))
    # ax3.text(0.01, 0.9, "C",
    #          fontsize=16, fontweight="bold", transform=ax3.transAxes,
    #          horizontalalignment="left", verticalalignment="top")
    #
    #
    # ax4 = plt.subplot(n_subplots, 1, 4, sharex=ax1)
    # ax4.tick_params(axis='both', which='major', labelsize=8)
    # n = 20
    # for i in range(n):
    #     ax4.plot(model["state"][I[i]], color='k', alpha=.25, lw=.5)
    # ax4.yaxis.tick_right()
    # ax4.set_ylim(-1.1, +1.1)
    # ax4.set_ylabel("Least correlated\n internal units (n={0})".format(n))
    # ax4.text(0.01, 0.9, "D",
    #          fontsize=16, fontweight="bold", transform=ax4.transAxes,
    #          horizontalalignment="left", verticalalignment="top")



    plt.tight_layout()
    plt.savefig("WM-one-gate-3-dupl.pdf")
    # plt.show()


    # save the model
    import cPickle
    with open("WM-one-gate__saved-model.pkl", "wb") as output_file:
        cPickle.dump(model, output_file)
