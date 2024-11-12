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

import numpy as np
import math
np.random.seed(0)

def answer1(a):
    return int((a[0, 0] + a[0, 1] - a[0, 2])>0)

def answer2(a):
    return int((-a[0, 0] - a[0, 1] + a[0, 2])>=0)

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
    a2 = np.append(sigFunc(z2), np.ones((x.shape[0], 1)), 1)

    z3 = a2.dot(w3)
    a3 = np.append(sigFunc(z3), np.ones((x.shape[0], 1)), 1)

    z4 = a3.dot(w4)
    a4 = np.append(sigFunc(z4), np.ones((x.shape[0], 1)), 1)
    
    z5 = a4.dot(w5)
    a5 = sigFunc(z5)

    return a5

dataI = np.matrix([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dataO = np.matrix([[answer1(a), answer2(a)] for a in dataI])
print(dataI)
print(dataO)
learningRate = 0.03

sigFunc = np.vectorize(sigmoid)
sigDXFunc = np.vectorize(sigmoidDX)

w1 = np.random.rand(3 + 1, 3)
w2 = np.random.rand(3 + 1, 3)
w3 = np.random.rand(3 + 1, 3)
w4 = np.random.rand(3 + 1, 3)
w5 = np.random.rand(3 + 1, 2)

for _ in range(100000):
    a0 = np.append(np.matrix(dataI), np.ones((dataI.shape[0], 1)), 1)

    z1 = a0.dot(w1)
    a1 = np.append(sigFunc(z1), np.ones((dataI.shape[0], 1)), 1)

    z2 = a1.dot(w2)
    a2 = np.append(sigFunc(z2), np.ones((dataI.shape[0], 1)), 1)

    z3 = a2.dot(w3)
    a3 = np.append(sigFunc(z3), np.ones((dataI.shape[0], 1)), 1)

    z4 = a3.dot(w4)
    a4 = np.append(sigFunc(z4), np.ones((dataI.shape[0], 1)), 1)
    
    z5 = a4.dot(w5)
    a5 = sigFunc(z5)

    deltaZ5 = sigDXFunc(a5-dataO)
    deltaZ4 = np.multiply(deltaZ5.dot(np.delete(w5.T, w5.T.shape[1]-1, 1)), np.delete(sigDXFunc(a4), a4.shape[1]-1, 1))
    deltaZ3 = np.multiply(deltaZ4.dot(np.delete(w4.T, w4.T.shape[1]-1, 1)), np.delete(sigDXFunc(a3), a3.shape[1]-1, 1))
    deltaZ2 = np.multiply(deltaZ3.dot(np.delete(w3.T, w3.T.shape[1]-1, 1)), np.delete(sigDXFunc(a2), a2.shape[1]-1, 1))
    deltaZ1 = np.multiply(deltaZ2.dot(np.delete(w2.T, w2.T.shape[1]-1, 1)), np.delete(sigDXFunc(a1), a1.shape[1]-1, 1))
    
    w5 -= a4.T.dot(deltaZ5) * learningRate
    w4 -= a3.T.dot(deltaZ4) * learningRate  
    w3 -= a2.T.dot(deltaZ3) * learningRate
    w2 -= a1.T.dot(deltaZ2) * learningRate
    w1 -= a0.T.dot(deltaZ1) * learningRate

    if _ % 1000 == 0:
        print (f"Loss after iteration {_}: { calculate_loss()}")
        

for i in range(dataI.shape[0]):
    print(dataI[i], dataO[i], predict(dataI[i]))
print(calculate_loss())
