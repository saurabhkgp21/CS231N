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
nHidden1 = 100
nHidden2 = 50
nOutput = 10
learningRate = 0.0001
epochs = 100
batch_size = 100
prob = 0.5
weight = {
	'W1': tf.Variable(tf.random_normal([nInput,nHidden1],stddev = 1/sqrt(nInput/2))),
	'W2': tf.Variable(tf.random_normal([nHidden1,nHidden2],stddev = 1/sqrt(nHidden1/2))),
	'W3': tf.Variable(tf.random_normal([nHidden2,nOutput],stddev = 1/sqrt(nHidden2/2)))
}

bias = {
	'b1': tf.Variable(tf.zeros([nHidden1])),
	'b2': tf.Variable(tf.zeros([nHidden2])),
	'b3': tf.Variable(tf.zeros([nOutput]))
}
isTraining = tf.placeholder(tf.bool)

a1 = tf.add(tf.matmul(x,weight['W1']),bias['b1'])
z1 = tf.nn.relu(a1)
z1 = tf.layers.dropout(z1,rate = prob, training = isTraining)
a2 = tf.add(tf.matmul(a1,weight['W2']),bias['b2'])
z2 = tf.nn.relu(a2)
z2 = tf.layers.dropout(z2,rate = prob,training = isTraining)
output = tf.add(tf.matmul(z2,weight['W3']),bias['b3'])

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output,labels = y))
train = tf.train.AdamOptimizer(learningRate).minimize(cost)

predict = tf.argmax(tf.nn.softmax(output),1)
accuracy = 100.0*tf.reduce_mean(tf.cast(tf.equal(predict,y),tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in xrange(epochs):
		for _ in xrange(numTrain/batch_size):
			indices = np.random.choice(np.arange(numTrain),batch_size)
			xT = xTrain[indices]
			yT = yTrain[indices]
			_,c,acc = sess.run([train,cost,accuracy],feed_dict = {
					x:xT,
					y:yT,
					isTraining:True
				})
		if (e+1)%100 == 0 or 1:
			c,acc = sess.run([cost,accuracy],feed_dict = {
					x:xTrain,
					y:yTrain,
					isTraining:False
				})
			print("At epoch {} ".format(e+1))
			print("Train cost {}, accuracy {} ".format(c,acc))
			c,acc = sess.run([cost,accuracy],feed_dict = {
					x:xValid,
					y:yValid,
					isTraining:False
				})
			print("Validation cost {}, accuracy {}".format(c,acc))
		