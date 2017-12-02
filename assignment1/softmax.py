import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import softmax
from cs231n import gradient_check
from cs231n.classifiers import linear_classifier
address = './cs231n/datasets/cifar-10-batches-py/'

xTrain,yTrain,xTest,yTest = load_CIFAR10(address)
lengthTrain = 5000
lengthTest = 100

xTrain = np.reshape(xTrain[:lengthTrain],(lengthTrain,-1))
# yTrain = np.reshape(yTrain[:lengthTrain])
yTrain = yTrain[:lengthTrain]
xTest = np.reshape(xTest[:lengthTest],(lengthTest,-1))
# yTest = np.reshape(yTest[:lengthTest])
yTest = yTest[:lengthTest]
xTrain = (xTrain - np.mean(xTrain,axis = 0))/(np.std(xTrain,axis = 0))

W = np.random.randn(xTrain.shape[1],10)*0.001
loss,grad = softmax.softmax_loss_naive(W,xTrain,yTrain,100)
# exit()

f = lambda w: softmax.softmax_loss_naive(w,xTrain,yTrain,0)[0]
# grad_numerical = gradient_check.grad_check_sparse(f,W,grad,10)

loss_naive,grad_naive = softmax.softmax_loss_naive(W,xTrain,yTrain,0)
loss_vectorized,grad_vectorized = softmax.softmax_loss_vectorized(W,xTrain,yTrain,0)


grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print(loss_naive)
print(loss_vectorized)
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

# exit()

learning_rate = 3e-4
regStrengths = xrange(0,200,50)
cvFold = 5
bestAccuracy = -1
bestReg = -1
for reg in regStrengths:
	acc = np.array([])
	X = np.array(np.split(xTrain,cvFold))
	Y = np.array(np.split(yTrain,cvFold))
	for k in xrange(cvFold):
		xValid = X[k]
		yValid = Y[k]
		xCurrent = X[np.arange(cvFold)!=k]
		yCurrent = Y[np.arange(cvFold)!=k]
		xCurrent = np.reshape(xCurrent,(xTrain.shape[0] - xTrain.shape[0]/cvFold,xTrain.shape[1]))
		yCurrent = np.reshape(yCurrent,(yTrain.shape[0] - yTrain.shape[0]/cvFold,))
		softmax = linear_classifier.Softmax()
		softmax.train(xCurrent,yCurrent,reg = k, verbose = True,num_iters = 1500,learning_rate = learning_rate)

		yPredict = softmax.predict(xValid)
		correct = np.sum(yPredict==yValid)
		acc = np.append(acc,float(correct)/yValid.shape[0])

	print("Reg = {}, acc = {}".format(reg,np.mean(acc)))
	if np.mean(acc) > bestAccuracy:
		bestReg = reg
		bestAccuracy = np.mean(acc)