# http://nghiaho.com/?p=1913
# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow

import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
cifar10 = './assignment1/cs231n/datasets/cifar-10-batches-py'
xTrain,yTrain,xTest,yTest = load_CIFAR10(cifar10)
xTrain = xTrain[:5000]
yTrain = yTrain[:5000]
xTest = xTest[:1000]
yTest = yTest[:1000]


numTrain = xTrain.shape[0]
numTest = xTest.shape[0]
nInput = 3072
filtersize = 4
filter1 = 32
filter2 = 32
filter3 = 64
numDense = 64
prob = 0.6
batch_size = 128
learningRate = 0.0003
epochs = 1000

x = tf.placeholder(tf.float32,shape = [None,32,32,3])
y = tf.placeholder(tf.int64,shape = [None,])
isTraining = tf.placeholder(tf.bool)

filter = {
	'f1': tf.Variable(tf.random_normal([4,4,3,filter1])),
	'f2': tf.Variable(tf.random_normal([7,7,filter1,filter2])),
	'f3': tf.Variable(tf.random_normal([5,5,filter2,filter3]))
}
w1 = tf.Variable(tf.random_normal([filter3,numDense]))
B1 = tf.Variable(tf.zeros([numDense]))

w2 = tf.Variable(tf.random_normal([numDense,10]))
B2 = tf.Variable(tf.zeros([10]))

z1 = tf.layers.conv2d(x,32,5,activation = tf.nn.relu)
z2 = tf.layers.max_pooling2d(z1,2,2)

z3 = tf.layers.conv2d(z2,32,5,activation = tf.nn.relu)
z4 = tf.layers.max_pooling2d(z3,2,2)

# z4 = z2
z5 = tf.layers.conv2d(z4,64,5,activation = tf.nn.relu)

z5 = tf.contrib.layers.flatten(z5)

z6 = tf.layers.dense(z5,numDense,activation = tf.nn.relu)
z6 = tf.layers.dropout(z6,rate = prob, training = isTraining)

z7 = tf.layers.dense(z6,10)



cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = z7, labels = y))
train = tf.train.AdamOptimizer(learningRate).minimize(cost)

predict = tf.argmax(tf.nn.softmax(z7),1)
accuracy = 100.0*tf.reduce_mean(tf.cast(tf.equal(predict,y),tf.float32))

with tf.Session() as sess:
	for e in xrange(epochs):
		n = numTrain/batch_size
		for count in xrange(numTrain/batch_size):
			indices = np.arange(numTrain)
			np.random.shuffle(indices)
			indices = indices[:batch_size]
			xT = xTrain[indices]
			yT = yTrain[indices]
			_,c = sess.run([train,cost],feed_dict = {
				x:xT,
				y:yT,
				isTraining:True
				})
			sys.stdout.write('\r')
			sys.stdout.write("Bar {}%".format(100.*(count+1)/n))
			sys.stdout.flush()
		if (e + 1)%100 == 0 or 1:
			acc,p = sess.run([accuracy,predict],feed_dict = {
				x:xTrain,
				y:yTrain,
				isTraining:False
				})
			print("")
			print("Epoch {}".format(e+1))
			print("Training Cost {}, Training Accuracy {}".format(c,acc))

			c,acc = sess.run([cost,accuracy],feed_dict = {
				x:xTest,
				y:yTest,
				isTraining:False
				})
			print("Testing cost {} testing accuracy {}".format(c,acc))