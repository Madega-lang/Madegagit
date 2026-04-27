import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))


data = [float(input()) for _ in range(8)]


x = np.array(data[:5])
t = np.array(data[5:])


W1 = np.ones((5,10))
W2 = np.ones((10,3))
b1 = np.ones(10)
b2 = np.ones(3)

lr = 0.1


z1 = np.dot(x, W1) + b1
h = sigmoid(z1)

z2 = np.dot(h, W2) + b2
y = sigmoid(z2)

loss1 = 0.5 * np.sum((y - t)**2)


delta2 = (y - t) * y * (1 - y)
delta1 = np.dot(W2, delta2) * h * (1 - h)


W2 = W2 - lr * np.outer(h, delta2)
W1 = W1 - lr * np.outer(x, delta1)

b2 = b2 - lr * delta2
b1 = b1 - lr * delta1


z1 = np.dot(x, W1) + b1
h = sigmoid(z1)

z2 = np.dot(h, W2) + b2
y = sigmoid(z2)

loss2 = 0.5 * np.sum((y - t)**2)

print(round(loss1,4))
print(round(loss2,4))