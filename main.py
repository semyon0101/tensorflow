"""import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import os
import skimage

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "D:/tensorflow"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

print(images.ndim)
print(images.size)

plt.hist(labels, 62)

plt.show()
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)

with tf.Session() as sess:
  output = sess.run(result)
  print(output)"""

"""import numpy as np
import math
np.random.seed(0)

def sigmoid(x):
    return 1/(1+math.e**(-x))

def sigmoidDX(sig):
    return sig*(1-sig)

def answer(a):
    return int((a[0] + a[1] - a[2])>0)

dataI = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dataO = np.array([answer(a) for a in dataI])
print(dataI)
print(dataO)
learningRate = 0.3

sigFunc = np.vectorize(sigmoid)
sigDXFunc = np.vectorize(sigmoidDX)

w1 = np.random.rand(4, 3)
w2 = np.random.rand(4, 2)
w3 = np.random.rand(3, 1)


for _ in range(1000):
    #print(_)
    for i in range(dataI.shape[0]):
        n1 = np.matrix(dataI[i]).T
        n1 = np.r_[n1, np.ones((1, 1))]
        n2 = sigFunc(np.dot(w1.T, n1))
        n2 = np.r_[n2, np.ones((1, 1))]
        n3 = sigFunc(np.dot(w2.T, n2))
        n3 = np.r_[n3, np.ones((1, 1))]
        n4 = sigFunc(np.dot(w3.T, n3))

        d4 = sigDXFunc(n4-dataO[i]).T
        w3 = w3 - np.dot(n3, d4) * learningRate

        d3 = (np.dot(d4, np.delete(w3.T, w3.T.shape[0]-1, 1)))
        w2 = w2 - np.dot(n2, d3) * learningRate

        d2 = (np.dot(d3, np.delete(w2.T, w2.T.shape[0]-1, 1)))
        w1 = w1 - np.dot(n1, d2) * learningRate
        

for i in range(dataI.shape[0]):
    n1 = np.matrix(dataI[i]).T
    n1 = np.r_[n1, np.ones((1, 1))]
    n2 = sigFunc(np.dot(w1.T, n1))
    n2 = np.r_[n2, np.ones((1, 1))]
    n3 = sigFunc(np.dot(w2.T, n2))
    n3 = np.r_[n3, np.ones((1, 1))]
    n4 = sigFunc(np.dot(w3.T, n3))
    print(dataI[i], dataO[i], n4[0,0])
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()