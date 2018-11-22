#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer 
        self.layers.append(np.zeros(self.shape[0]))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][:] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw 
            self.dw[i] = dw
            # self.weights[i] = np.round(100*self.weights[i])/100
            
        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # np.random.seed(1)
    data = np.zeros(25000, dtype=[('input',  float, 3),
                                  ('output', float, 1)])
    data['input'][:,0] = A = np.random.uniform(-1,1,len(data))
    data['input'][:,1] = B = np.random.uniform(-1,1,len(data))
    data['input'][:,2] = T = np.random.randint(0,2,len(data)).astype(float)
    # data['input'][:,2] = T = np.random.uniform(0,1,len(data))
    data['output'] = T*A+(1-T)*B

    lrate = 0.1
    network = MLP(3,3,1)
    network.reset()

    E = []
    for i in np.random.randint(0,len(data),25000):
        network.propagate_forward(data['input'][i])
        error = network.propagate_backward(data['output'][i], lrate )
        E.append(error)
        
    # for i in np.random.randint(0,len(data),2500):
    #     network.propagate_forward(data['input'][i])
    #     error = (network.layers[-1] - data['output'][i])**2
    #     E.append(error)

    # print(network.weights)

    plt.figure(figsize=(14,3))
    plt.plot(E[::1], lw=.5)
    plt.tight_layout()
    plt.show()


    # n = 100
    # # A = +0.5 + 0.0*np.random.uniform(-1,+1,n)
    # # B = -0.5 + 0.0*np.random.uniform(-1,+1,n)

    # A = np.linspace(-1,1,n)
    # B = np.cos(np.linspace(-3*np.pi,3*np.pi,n))

    # T = np.random.randint(0,2,n).astype(float)
    # T = np.random.uniform(-1,+1,n)
    # O = T*A+(1-T)*B

    # h1 = np.tanh(-0.94*A + 0.27*B - 2.32*T)
    # h2 = np.tanh(-0.71*A + 0.07*B - 0.11*T)
    # h3 = np.tanh(-0.02*A + 0.78*B + 2.01*T)
    # O_ = np.tanh( 1.64*h1 - 2.02*h2 + 1.47*h3)
    # print (abs((O_ - O)).mean())

    # plt.figure(figsize=(14,3))
    # n=4
    
    # ax = plt.subplot(n,1,1)
    # plt.plot(h1)
    # plt.ylim(-1,1)
    
    # ax = plt.subplot(n,1,2)
    # plt.plot(h2)
    # plt.ylim(-1,1)

    # ax = plt.subplot(n,1,3)
    # plt.plot(h3)
    # plt.ylim(-1,1)
        
    # ax = plt.subplot(n,1,4)
    # plt.plot(O_)
    # plt.plot(A,alpha=.25)
    # plt.plot(B,alpha=.25)
        
    # plt.ylim(-1,1)
    
    # plt.show()
