


import numpy as np

# Generating random reservoir states
# x1 = np.random.randn(5,1)*0.5
# x2 = np.random.randn(5,1)*0.5
# x3 = np.random.randn(5,1)*0.5
x1 = np.tanh(np.random.randn(5,1)*0.5)
x2 = np.tanh(np.random.randn(5,1)*0.5)
x3 = np.tanh(np.random.randn(5,1)*0.5)
X = np.hstack((x1,x2,x3))
print("X")
print(X)

# Generate a random Wfb
fb = np.random.randn(5,1)*0.5

# Computing Wout directly
out = np.linalg.pinv(fb).dot(np.arctanh(X)).dot(np.linalg.pinv(X))

# Computing output states
y1 = out.dot(x1)
y2 = out.dot(x2)
y3 = out.dot(x3)
print("Y", y1, y2, y3)

# Computing "new" x values
x1new = np.tanh(fb.dot(y1))
x2new = np.tanh(fb.dot(y2))
x3new = np.tanh(fb.dot(y3))
print("newX", x1new, x2new, x3new)

# Comparing values
print("x_i - tanh(Wfb.y_i)")
print(x1 - x1new)
print(x2 - x2new)
print(x3 - x3new)

# Rewritting of W when W is not recurrent: W <- Wfb.Wout
W = fb.dot(out)

print("W rewritten : Wfb.Wout")
print(W)
