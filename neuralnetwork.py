import tensorflow as tf
import pandas as pd
import numpy as np
from math import sqrt
train = pd.read_csv('./Dataset/train.csv')
test = pd.read_csv('./Dataset/test.csv')
yTrain = np.array(train['label'],dtype = int)
xTrain = np.array(train.iloc[:,1:])
xTest = np.array(test.iloc[:,0:])


numTrain = int(xTrain.shape[0]*0.7)
xValid = xTrain[numTrain:]
yValid = yTrain[numTrain:]
xTrain = xTrain[:numTrain]
yTrain = yTrain[:numTrain]

x = tf.placeholder('float',shape = [None,784])
y = tf.placeholder(tf.int64,shape = [None,])

nInput = 784
nHidden = 50
nOutput = 10
learningRate = 0.01
epochs = 2000
weight = {
	'W1': tf.Variable(tf.random_normal([nInput,nHidden])),
	'W2': tf.Variable(tf.random_normal([nHidden,nOutput]))
}

bias = {
	'b1': tf.Variable(tf.zeros([nHidden])),
	'b2': tf.Variable(tf.zeros([nOutput]))
}

z1 = tf.add(tf.matmul(x,weight['W1']),bias['b1'])
a1 = tf.nn.relu(z1)
output = tf.add(tf.matmul(a1,weight['W2']),bias['b2'])

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output,labels = y))
train = tf.train.AdamOptimizer(learningRate).minimize(cost)

predict = tf.argmax(output,1)
accuracy = 100.0*tf.reduce_mean(tf.cast(tf.equal(predict,y),tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in xrange(epochs):
		_,c,acc = sess.run([train,cost,accuracy],feed_dict = {
				x:xTrain,
				y:yTrain
			})
		if (e+1)%100 == 0:
			print("At epoch {} ".format(e+1))
			print("Train cost {}, accuracy {} ".format(c,acc))
			c,acc = sess.run([cost,accuracy],feed_dict = {
					x:xValid,
					y:yValid
				})
			print("Validation cost {}, accuracy {}".format(c,acc))
		