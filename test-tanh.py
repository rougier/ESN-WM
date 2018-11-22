import numpy as np
import matplotlib.pyplot as plt

t = 0
x = np.linspace(-1.0, +1.0, 100)

scale = 0.1
norm = (np.tanh(1-scale) - np.tanh(1+scale)
      + np.tanh(  scale) - np.tanh( -scale))

y = (np.tanh(t-scale*x) - np.tanh(t+scale*x)
   + np.tanh(  scale*x) - np.tanh( -scale*x)) / norm


y = (np.tanh(t+scale*x) - np.tanh(t-scale*x)
   - np.tanh(1+scale*x) + np.tanh(1-scale*x) ) / norm

plt.plot(x, y)
plt.axhline(0, color='k', linewidth=.5)
plt.show()
