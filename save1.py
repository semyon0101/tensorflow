import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
np.random.seed(0)


def sigmoid(x):
    return 1/(1+math.e**(-x))

def sigmoidDX(sig):
    return sig*(1-sig)

def calculate_loss():
    loss = np.sum(np.power((predict(dataI)-dataO), 2))
    return loss

def predict(x):
    a0 = np.append(np.matrix(x), np.ones((x.shape[0], 1)), 1)

    z1 = a0.dot(w1)
    a1 = np.append(sigFunc(z1), np.ones((x.shape[0], 1)), 1)

    z2 = a1.dot(w2)
    a2 = sigFunc(z2)

    return a2

def plot_decision_boundary(pred_func):
    x_min, x_max = np.array(dataI)[:, 0].min() - .5, np.array(dataI)[:, 0].max() + .5
    y_min, y_max = np.array(dataI)[:, 1].min() - .5, np.array(dataI)[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = np.round(Z.reshape(xx.shape))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)

dataI, dataO = datasets.make_moons(200, noise=0.20)

dataI = np.matrix(dataI)
dataO = np.matrix(dataO).T

learningRate = 0.03

sigFunc = np.vectorize(sigmoid)
sigDXFunc = np.vectorize(sigmoidDX)

w1 = np.random.rand(2 + 1, 30)
w2 = np.random.rand(30 + 1, 1)

for _ in range(20000):
    a0 = np.append(np.matrix(dataI), np.ones((dataI.shape[0], 1)), 1)

    z1 = a0.dot(w1)
    a1 = np.append(sigFunc(z1), np.ones((dataI.shape[0], 1)), 1)

    z2 = a1.dot(w2)
    a2 = sigFunc(z2)

    deltaZ2 = sigDXFunc(a2-dataO)
    deltaZ1 = np.multiply(deltaZ2.dot(np.delete(w2.T, w2.T.shape[1]-1, 1)), np.delete(sigDXFunc(a1), a1.shape[1]-1, 1))
    
    w2 -= a1.T.dot(deltaZ2) * learningRate
    w1 -= a0.T.dot(deltaZ1) * learningRate

    if _ % 1000 == 0:
        print (f"Loss after iteration {_}: { calculate_loss()}")
        

#for i in range(dataI.shape[0]):
#    print(dataI[i], dataO[i], predict(dataI[i]))
print(calculate_loss())



plot_decision_boundary(lambda x: predict(x))
plt.scatter(np.array(dataI)[:,0], np.array(dataI)[:,1], s=40, c=np.array(dataO.T), cmap=plt.cm.Spectral)

plt.show()