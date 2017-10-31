import pandas as pd
import numpy as np
import tensorflow as tf


train = pd.read_csv("/home/saurabh/Saurabh/Stanford/MNIST/MNIST_data/train2.csv")
test = pd.read_csv("/home/saurabh/Saurabh/Stanford/MNIST/MNIST_data/test2.csv")

train = np.array(train)
test = np.array(test)
trainX = train[:,1:]
label = train[:,0]
trainY = np.zeros((label.shape[0],10))
for i in xrange(label.shape[0]):
	trainY[i,label[i]] = 1

validX = trainX[-1000:]
validY = label[-1000:]
trainX = trainX[:-1000]
trainY = trainY[:-1000]

inputSize = trainX.shape[1]
hidden = inputSize/2;
output = trainY.shape[1]

W1 = tf.Variable(tf.random_normal([inputSize,hidden],stddev = 1/trainX.shape[0]**0.5),dtype = tf.float32)
W2 = tf.Variable(tf.random_normal([hidden,output],stddev = 1))
b1 = tf.Variable(tf.random_normal([hidden],dtype  =tf.float64,stddev = 1/trainX.shape[0]**0.5),)
b2 = tf.Variable(tf.random_normal([output],stddev = 1))
W1 = tf.cast(W1,tf.float64)
W2 = tf.cast(W2,tf.float64)
b1 = tf.cast(b1,tf.float64)
b2 = tf.cast(b2,tf.float64)

trainX = trainX.astype(np.float64)
trainY = trainY.astype(np.float64)
validX = validX.astype(np.float64)
validY = validY.astype(np.float64)
x = tf.placeholder(tf.float64,shape = [None,784])
y = tf.placeholder(tf.float64,shape = [None,10])
z = tf.placeholder(tf.float64,shape = None)
z1 = tf.sigmoid(tf.matmul(x,W1) + b1)
z2 = tf.nn.softmax(tf.matmul(z1,W2) + b2)
# cost = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = z2)
cost = -tf.reduce_mean(y*tf.log(z2))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

batchSize = 200

accurate = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in xrange(200):
		randomize = np.arange(trainX.shape[0])
		np.random.shuffle(randomize)
		trainX = trainX[randomize]
		trainY = trainY[randomize]
		for itr in xrange(0,trainX.shape[0],batchSize):
			sess.run(train,feed_dict={x : trainX[itr:itr+batchSize],y : trainY[itr:itr+batchSize]})
		# sess.run(train,feed_dict={x:trainX,y:trainY})
		accuracy = tf.equal(tf.cast(tf.argmax(z2,1),tf.float64),z)
		acc = tf.reduce_mean(tf.cast(accuracy,tf.float64))
		temp = sess.run(acc,feed_dict={ x:validX,z:validY})
		print("Accuracy for epoch {} is {}".format(i+1,temp))
		accurate.append(temp)
	print(sess.run(tf.argmax(z2,1),feed_dict ={ x:validX[:50]}))
	print(validY[:50])
	import matplotlib.pyplot as plt
	plt.plot(accurate)
	plt.show()