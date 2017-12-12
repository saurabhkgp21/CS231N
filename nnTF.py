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
xTrain = xInput[:numTrain]
yTrain = yInput[:numTrain]
xValid = xInput[numTrain:]
yValid = yInput[numTrain:]
xTest = np.reshape(xTest,(xTest.shape[0],-1))
xTest = (xTest - np.mean(xTest,axis = 0))/np.std(xTest,axis = 0)
nInput = xTrain.shape[1]
nHidden1 = 200
nHidden2 = 10
nOutput = 10
learningRate = 0.0001
epochs = 150
batch_size = 200
prob = 0.4
trainAcc = []
trainCost = []
validAcc = []
validCost = []
W = {
	'W1' : tf.Variable(tf.random_normal([nInput,nHidden1],stddev = 1/sqrt(nInput/2))),
	'W2' : tf.Variable(tf.random_normal([nHidden1,nHidden2],stddev = 1/sqrt(nHidden1/2))),
	'W3' : tf.Variable(tf.random_normal([nHidden2,nOutput],stddev = 1/sqrt(nHidden2/2)))
}
B = {
	'b1' : tf.Variable(tf.zeros([nHidden1])),
	'b2' : tf.Variable(tf.zeros([nHidden2])),
	'b3' : tf.Variable(tf.zeros([nOutput]))
}

def Output(x,isTraining):
	a1 = tf.add(tf.matmul(x,W['W1']), B['b1'])
	z1 = tf.nn.relu(a1)
	z1 = tf.layers.dropout(z1,rate = prob,training = isTraining)
	a2 = tf.add(tf.matmul(z1,W['W2']), B['b2'])
	return a2
	z2 = tf.nn.relu(a2)
	z2 = tf.layers.dropout(z2,rate = prob,training = isTraining)
	a3 = tf.add(tf.matmul(z2,W['W3']), B['b3'])
	return a3


x = tf.placeholder(tf.float32,[None,nInput])
y = tf.placeholder(tf.int64,[None,])
isTraining = tf.placeholder(tf.bool)
logits = Output(x,isTraining)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
train = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


predict = tf.argmax(tf.nn.softmax(logits),1)
accuracy = 100.0*tf.reduce_mean(tf.cast(tf.equal(predict,y),tf.float32))

for it in xrange(epochs):
	indices = np.random.choice(np.arange(numTrain),batch_size)
	xT = xTrain[indices]
	yT = yTrain[indices]
	sess.run(train,feed_dict = {x:xT,y:yT,isTraining:True})
	if it%200 == 0 or 1:
		print("At step {}".format(it+1))
		c,acc = sess.run([cost,accuracy],feed_dict = {x:xTrain,y:yTrain,isTraining:False})
		trainAcc += [acc]
		trainCost += [c]
		print("Training cost {}, accuracy {}".format(c,acc))
		c,acc = sess.run([cost,accuracy],feed_dict = {x:xValid,y:yValid,isTraining:False})
		validAcc += [acc]
		validCost += [c]
		print("Validation cost {}, accuracy {}".format(c,acc))

c = sess.run(cost,feed_dict={x:xTest,y:yTest,isTraining:False})
t = sess.run(accuracy,feed_dict={x:xTest,y:yTest,isTraining:False})
print("Cost {} , acc {}".format(c,t))
plt.figure(1)
plt.subplot(121)
plt.plot(trainAcc)
plt.plot(validAcc)
plt.subplot(122)
plt.plot(trainCost)
plt.plot(validCost)
plt.show()
exit()