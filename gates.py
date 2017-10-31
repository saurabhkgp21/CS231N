import tensorflow as tf
import numpy as np

trainX = np.array([[0,0],[1,0],[0,1],[1,1]])

# Change Outpute According Gate --- (0-->[1,0] 1--->[0,1])
trainY = np.array([[0,1],[1,0],[1,0],[0,1]])

x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,2])


# For and or Gate
# W = tf.Variable(tf.random_normal([2,2]))
# b = tf.Variable(tf.random_normal([2]))
# z = tf.nn.softmax(tf.matmul(x,W) + b)

# For Xor and XNOR Gate
W1 = tf.Variable(tf.random_normal([2,4]))
b1 = tf.Variable(tf.random_normal([4]))
W2 = tf.Variable(tf.random_normal([4,2]))
b2 = tf.Variable(tf.random_normal([2]))

a1 = tf.sigmoid(tf.matmul(x,W1) + b1)
z = tf.nn.softmax(tf.matmul(a1,W2) + b2)


cost = -tf.reduce_mean(y*tf.log(z))		# Note negative Sign

train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in xrange(5000):
		sess.run(train,feed_dict={x:trainX,y:trainY})

	print(sess.run(z,feed_dict={x:trainX,y:trainY}))