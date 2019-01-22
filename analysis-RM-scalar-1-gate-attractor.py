

"""
In this code we analyse the rewritten network W' of the 3-neuron network of Anthony
(i.e. the "canonical switch network") : W' = W + W_{fb}W_{out}

We do a dynamical system approach to study the stability properties of the network.
"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def linear_update(m,v):
    return np.dot(m, v)

def nonlinear_update(m,v):
    return np.tanh(np.dot(m, v))

def noisy_linear_update(m,v,noise_gain):
    return np.dot(m, v + noise_gain * (np.random.rand((v.shape[0]))-0.5) )

def noisypositive_linear_update(m,v,noise_gain):
    return np.dot(m, v + noise_gain * np.random.rand((v.shape[0])) )

def noisy_nonlinear_update(m,v,noise_gain):
    return np.tanh(np.dot(m, v + noise_gain * (np.random.rand((v.shape[0]))-0.5) ))

def noisypositive_nonlinear_update(m,v,noise_gain):
    return np.tanh(np.dot(m, v + noise_gain * np.random.rand((v.shape[0])) ))

def init_states(dim, init_scale, nr_timesteps):
    x = np.zeros((dim, nr_timesteps))
    # x[:,0] = np.random.rand((dim)) * init_scale
    x[:,0] = v1 * 0.7
    return x

def plot_states(x, title="", save=False):
    plt.figure()
    # plt.figure(figsize=(14,9))
    plt.plot(range(nr_timesteps), x[0,:], label='neuron 1')
    plt.plot(range(nr_timesteps), x[1,:], label='neuron 2')
    plt.plot(range(nr_timesteps), x[2,:], label='neuron 3')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Neuron activation")
    plt.title(title)
    if save:
        plt.savefig(title+".pdf")

if __name__=='__main__':
    # W'  network : rewritten form of "canonical switch network"
    m = np.array([[0,0,0],[0,0,0],[1,-1,1]])
    # Compute eigenvectors
    w, v = np.linalg.eig(m)
    v1 = v[:,0]
    # # Or define the eigenvector of non-zero eigenvalue
    # v1 = np.array([0,0,1]) # eigen vectore of matrix m

    """Run from random state to see if it converges"""
    init_scale = 1.0 #1000 #0.001 #10.0
    # nr_timesteps = 1000000 #max acceptable limite to wait for simulation (xav: 3 sec for linear and 9 sec for nonlinear)
    nr_timesteps = 10
    # noise_gain = 0.01
    noise_gain = 0.01
    positive_noise = False #True

    # Run linear update
    x = init_states(v.shape[0], init_scale, nr_timesteps)
    for i in tqdm.trange(0,nr_timesteps-1):
        x[:,i+1] = linear_update(m,x[:,i])
    # Display
    plot_states(x, "Linear update", save=True)
    plt.title("Linear update")
    linear_update_states = x[:,:]


    # Run nonlinear update
    x = init_states(v.shape[0], init_scale, nr_timesteps)
    for i in tqdm.trange(0,nr_timesteps-1):
        x[:,i+1] = nonlinear_update(m,x[:,i])
    # Display
    plot_states(x, "Nonlinear update", save=True)


    # Run linear noisy update
    x = init_states(v.shape[0], init_scale, nr_timesteps)
    for i in tqdm.trange(0,nr_timesteps-1):
        if positive_noise:
            x[:,i+1] = noisypositive_linear_update(m,x[:,i],noise_gain)
        else:
            x[:,i+1] = noisy_linear_update(m,x[:,i],noise_gain)
    # Display
    plot_states(x, "Linear noisy update", save=True)


    # Run nonlinear noisy update
    x = init_states(v.shape[0], init_scale, nr_timesteps)
    for i in tqdm.trange(0,nr_timesteps-1):
        if positive_noise:
            x[:,i+1] = noisypositive_nonlinear_update(m,x[:,i],noise_gain)
        else:
            x[:,i+1] = noisy_nonlinear_update(m,x[:,i],noise_gain)
    # Display
    plot_states(x, "Nonlinear noisy update", save=True)


    # Display difference between Linear and Noisy Nonlinear
    plot_states(linear_update_states - x, "Difference between Linear and Noisy Nonlinear", save=True)

    plt.show()
