import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import *
from assignment1.cs231n.data_utils import load_CIFAR10

address = './assignment1/cs231n/datasets/cifar-10-batches-py/'

xInput,yInput,xTest,yTest = load_CIFAR10(address)
xInput = np.reshape(xInput,(xInput.shape[0],-1))
xInput -= np.mean(xInput,axis = 0)
numInput = xInput.shape[0]
numTrain = int(0.7*numInput)
print(numTrain)
xTrain = xInput[:numTrain]
yTrain = yInput[:numTrain]
xValid = xInput[numTrain:]
yValid = yInput[numTrain:]
# xTrain = np.reshape(xTrain,(xTrain.shape[0],-1))
xTest = np.reshape(xTest,(xTest.shape[0],-1))
# xTrain = (xTrain - np.mean(xTrain,axis = 0))/np.std(xTrain,axis = 0)
xTest = (xTest - np.mean(xTest,axis = 0))/np.std(xTest,axis = 0)
# print(yTrain.dtype)
# exit()

# numTrain = xTrain.shape[0]
nInput = xTrain.shape[1]
nHidden1 = 600
nHidden2 = 200
nOutput = 10
learningRate = 0.001
epochs = 2000
batch_size = 200

W = {
	'W1' : tf.Variable(tf.random_normal([nInput,nHidden1],stddev = 1/sqrt(nInput))),
	'W2' : tf.Variable(tf.random_normal([nHidden1,nHidden2],stddev = 1/sqrt(nHidden1))),
	'W3' : tf.Variable(tf.random_normal([nHidden2,nOutput],stddev = 1/sqrt(nHidden2)))
}
B = {
	'b1' : tf.Variable(tf.zeros([nHidden1])),
	'b2' : tf.Variable(tf.zeros([nHidden2])),
	'b3' : tf.Variable(tf.zeros([nOutput]))
}

def Output(x):
	a1 = tf.add(tf.matmul(x,W['W1']), B['b1'])
	z1 = tf.nn.relu(a1)
	a2 = tf.add(tf.matmul(z1,W['W2']), B['b2'])
	z2 = tf.nn.relu(a2)
	a3 = tf.add(tf.matmul(z2,W['W3']), B['b3'])
	return a3


x = tf.placeholder(tf.float32,[None,nInput])
y = tf.placeholder(tf.int32,[None,])
logits = Output(x)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
train = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

predict = tf.nn.softmax(logits)
labels = tf.cast(tf.argmax(predict,1),tf.int32)
correct = tf.reduce_sum(tf.cast(tf.equal(labels,y),tf.int32))
accuracy = 10000*correct/tf.shape(x)[0]


for it in xrange(epochs):
	indices = np.random.choice(np.arange(numTrain),batch_size)
	xT = xTrain[indices]
	yT = yTrain[indices]
	sess.run(train,feed_dict = {x:xT,y:yT})
	if it%200 == 0:
		c = sess.run(cost,feed_dict = {x:xValid,y:yValid})
		acc = sess.run(accuracy,feed_dict={x:xTest,y:yTest})/100.0
		print("At step {}".format(it))
		print("Cost {}, acc {}".format(c,acc))


c = sess.run(cost,feed_dict={x:xTest,y:yTest})
t = sess.run(accuracy,feed_dict={x:xTest,y:yTest})/100.0
print("Cost {} , acc {}".format(c,t))
exit()
indices = np.random.choice(np.arange(numTrain),batch_size)
xT = xTrain[indices]
yT = yTest[indices]


