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

w1 = np.random.rand(3, 2)
w2 = np.random.rand(2, 1)


for _ in range(1000):
    print(_)
    for i in range(dataI.shape[0]):
        n1 = np.matrix(dataI[i]).T
        n2 = sigFunc(np.dot(w1.T,n1))
        n3 = sigFunc(np.dot(w2.T,n2))
        #print(dataI[i], dataO[i], n3[0,0]) 

        d3 = sigDXFunc(n3-dataO[i]).T
        w2 = w2 - np.dot(n2,  d3) * learningRate

        d2 = sigDXFunc(np.dot(w2, d3)).T
        w1 = w1 - np.dot(n1, d2) * learningRate
        

for i in range(dataI.shape[0]):
    n1 = np.matrix(dataI[i]).T
    n2 = sigFunc(np.dot(w1.T,n1))
    n3 = sigFunc(np.dot(w2.T,n2))
    print(dataI[i], dataO[i], n3[0,0])""" 
import numpy as np
import math

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

w1 = np.random.rand(4, 1)


for _ in range(1000):
    print(_)
    for i in range(dataI.shape[0]):
        n1 = np.matrix(dataI[i]).T
        n1 = # нужен сдвиг  bias !!!!
        n2 = sigFunc(np.dot(w1.T,n1))

        d2 = sigDXFunc(n2-dataO[i]).T
        w1 = w1 - np.dot(n1,  d2) * learningRate
        

for i in range(dataI.shape[0]):
    n1 = np.matrix(dataI[i]).T
    n2 = sigFunc(np.dot(w1.T,n1))
    print(dataI[i], dataO[i], n2[0,0])

