import tqdm
import numpy as np

def generate_data(n=10000, p=0.01):
    data = np.zeros((n,4))
    data[:,0] = np.random.uniform(-1,+1,n)
    data[:,1] = np.random.uniform(0,1,n) < p
    # data[::50,1] = 1
    data[0,1] = 1.0
    wm = data[0,0]
    for i in range(n):
        if data[i,1] > 0:
            wm = data[i,0]
        data[i,2] = wm
    return data

def test(data, state):
    n = len(data)
    last_input, last_internal, last_output = state
    inputs    = np.vstack([last_input, data[:,:n_input]])
    internals = np.vstack([last_internal, np.zeros((n, n_unit))])
    outputs   = np.vstack([last_output, np.zeros((n, n_output))])
    for i in tqdm.trange(1,n+1):
        z = ( np.dot(W, internals[i-1]) +
              np.dot(W_in, inputs[i]) +
              scaling*np.dot(W_fb, outputs[i-1]) )
        internals[i] = np.tanh(z) + noise*(np.random.uniform(-1.0, +1.0, n_unit))
        internals[i,:] = (1-leak)*internals[i-1] + leak*internals[i,:]
        # outputs[i] = np.dot(W_out, np.concatenate([internals[i], inputs[i]]))
        outputs[i] = np.dot(W_out, internals[i])
    return outputs[1:], internals[1:].T

# Parameters
np.random.seed(3)    # Random number generator seed
n_unit      = 3   # Number of unit in the reservoir
n_input     = 2      # Number of input
n_output    = 1      # Number of output
noise       = 0.0 # Noise level
scaling     = 1.0   # Feedback scaling
leak        = 1.0    # Leak rate

a_s = [5, 2 , 1]
b_s = [0.01, 0.1, 1] # TODO test with b = 10
n = max(len(a_s), len(b_s))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
basesize = (6,2)
f = plt.figure(figsize=(2*basesize[0],n*basesize[1]))
g = f.add_gridspec(n, 2)
data = generate_data(2500, p=0.01)
I = data[:,:2]

b = min(b_s)
for i, a in enumerate(a_s):
    # Reservoir building
    W_in = np.array([[0,a],[b,a],[b,0]])
    W = np.zeros(( n_unit, n_unit))
    W_fb = np.array([[b],[0],[0]])
    W_out = np.array([[1,-1,1]])/b
    state = np.zeros((n_input,)), np.zeros((n_unit,)), np.zeros((n_output,))

    # Testing
    O = data[:,2:]
    P,R = test(data, state)
    error = np.sqrt(np.mean((P - O)**2))
    print("Testing error : {0}".format(error))


    # Display

    ax1 = f.add_subplot(g[i,0])
    ax1.plot(O,  color='0.75', lw=1.0)
    ax1.plot(P, color='0.00', lw=1.5)
    ax1.set_ylim(-1.1,1.1)
    ax1.set_xticks([])
    ax1.yaxis.tick_right()
    ax1.set_ylabel("Output")
    ax1.set_title("a = {:.0f}, b = {:.2f}".format(a,b))
ax1.set_xlabel("Time")

a = max(a_s)
for i, b in enumerate(b_s):
    # Reservoir building
    W_in = np.array([[0,a],[b,a],[b,0]])
    W = np.zeros(( n_unit, n_unit))
    W_fb = np.array([[b],[0],[0]])
    W_out = np.array([[1,-1,1]])/b
    state = np.zeros((n_input,)), np.zeros((n_unit,)), np.zeros((n_output,))

    # Testing
    O = data[:,2:]
    P,R = test(data, state)
    error = np.sqrt(np.mean((P - O)**2))
    print("Testing error : {0}".format(error))


    # Display

    ax1 = f.add_subplot(g[i,1])
    ax1.plot(O,  color='0.75', lw=1.0)
    ax1.plot(P, color='0.00', lw=1.5)
    ax1.set_ylim(-1.1,1.1)
    ax1.set_xticks([])
    ax1.yaxis.tick_right()
    ax1.set_title("a = {:.1g}, b = {:.1g}".format(a,b))
ax1.set_xlabel("Time")


plt.tight_layout()
plt.savefig("small_a{:s}_b{:s}.pdf".format(str(a_s), str(b_s)))

