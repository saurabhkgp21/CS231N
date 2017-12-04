import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import *
# import assignment1
from assignment1.cs231n.data_utils import load_CIFAR10

address = './assignment1/cs231n/datasets/cifar-10-batches-py/'

xTrain,yTrain,xTest,yTest = load_CIFAR10(address)
# xTrain = np.array(xTrain[:2000])
# yTrain = np.array(yTrain[:2000])
# print(xTrain.shape)
xTrain = np.reshape(xTrain,(xTrain.shape[0],-1))
xTest = np.reshape(xTest,(xTest.shape[0],-1))
# print(xTrain.shape)
# print(xTest.shape)
xForPreProcess = xTrain
np.vstack((xForPreProcess,xTest))
m = np.mean(xForPreProcess,axis = 0)
s = np.std(xForPreProcess,axis = 0)
# xTrain = (xTrain - m)/s
# xTest = (xTest - m)/s
xTrain = (xTrain - np.mean(xTrain,axis = 0))/np.std(xTrain,axis = 0)
xTest = (xTest - np.mean(xTest,axis = 0))/np.std(xTest,axis = 0)


numTrain = xTrain.shape[0]
numFeatures = xTrain.shape[1]

def convertLabel(y):
	Y = np.zeros((y.shape[0],10))
	Y[np.arange(y.shape[0]),y] = 1
	return Y

# yTrain = convertLabel(yTrain)
# yTest = convertLabel(yTest)
x = tf.placeholder(tf.float32,[None,numFeatures])
y = tf.placeholder(tf.int64,[None])

W = tf.Variable(tf.random_normal([numFeatures,10],stddev = 1/sqrt(numTrain)))
z = tf.matmul(x,W)
predict = tf.nn.softmax(z)
predictLabel = tf.cast(tf.argmax(predict,1),tf.int32)
# actualLabel = tf.cast(tf.argmax(y,1),tf.int32)
# actualLabel = y
# acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.argmax(y,1)),tf.int32))
acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict,1),tf.cast(y,tf.int64)),tf.int32))
# accFloat = tf.cast(acc,tf.float32)
accuracy = (10000*acc)/tf.shape(predict)[0]
# softmaxcrossEntropyWithLogits , first calculate softmax of logits 
# then cross entropy using lables*log(softmax)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z ,labels = y))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits =z , labels = y) )
train = tf.train.GradientDescentOptimizer(1e-3).minimize(cost)
# s = tf.nn.softmax(logits = l)
# a = np.array([[10,10,10,20,20]])
# b = np.array([[0,0,0,1,0]])


sess= tf.InteractiveSession()
tf.global_variables_initializer().run()

num_iters = 10000
batch_size = 200
plottingAcc = []
plottingCost = []
for it in xrange(num_iters+1):
	indices = np.random.choice(np.arange(numTrain),batch_size)
	xT = xTrain[indices]
	yT = yTrain[indices]
	sess.run(train,feed_dict={x:xT,y:yT})

	if it%200 == 0:
		c = sess.run(cost,feed_dict={x:xT,y:yT})
		print("At step {}, cost {}".format(it,c))
		t = sess.run(accuracy,feed_dict={x:xTest,y:yTest})
		t = float(t)/100
		plottingAcc.append(t)
		# plottingCost.append(c)
		# print("Train Accuracy = {}".format(sess.run(acc,feed_dict={x:xT,y:yT})))
		# print("Test Cost = {}".format(t))
		print("Test Accuracy = {}".format(t))
		# print("Lable = {}".format(sess.run(predictLabel,feed_dict={x:xTest})))
		# print("Actual Label = {}".format(sess.run(actualLabel,feed_dict={y:yTest})))
# plt.figure(1)
# plt.subplot(121)
plt.plot(plottingAcc)
# plt.subplot(122)
# plt.plot(plottingCost)
plt.show()