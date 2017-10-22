import numpy as np
import time
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import linear_svm
from cs231n.classifiers import linear_classifier
from cs231n import gradient_check
address = './cs231n/datasets/cifar-10-batches-py/'

xTrain,yTrain,xTest,yTest = load_CIFAR10(address)
lengthTrain = 5000
lengthTest = 100

xTrain = np.reshape(xTrain,(xTrain.shape[0],-1))
xTest = np.reshape(xTest,(xTest.shape[0],-1))

xTrain = xTrain[:lengthTrain]
yTrain = yTrain[:lengthTrain]
xTest = xTest[:lengthTest]
yTest = yTest[:lengthTest]

# print(xTrain.shape)
# print(np.mean(xTrain,axis = 0).shape)
# print(np.std(xTrain,axis = 0).shape)

xTrain = (xTrain - np.mean(xTrain,axis = 0))/np.std(xTrain,axis = 0)
# print(np.mean(xTrain,axis = 0))
# print(np.std(xTrain,axis = 0))


W = 2*np.random.random_sample((xTrain.shape[1],10)) - 1
reg = 10


start = time.time()
loss,dw = linear_svm.svm_loss_naive(W, xTrain, yTrain, 0)
print("Time for Naive ",time.time() - start)


start = time.time()
lossVector,dwVector = linear_svm.svm_loss_vectorized(W,xTrain,yTrain,0)
print("Time for Vectorised approach ",time.time() - start)


# def f(w):
	# return linear_svm.svm_loss_naive(w,xTrain,yTrain,0)[0]
# gradient_check.grad_check_sparse(f, W, dw, num_checks=10, h=1e-5)

cvFold = 5
learningRates = [1e-3]
regStrengths = [0,100 ,200,400, 500]

xTrainCV = np.array(np.split(xTrain,cvFold))
yTrainCV = np.array(np.split(yTrain,cvFold))

accuracy = -1
alphaBest = -1
regBest = -1
for alpha in learningRates:
	for reg in regStrengths:
		acc = np.array([])
		for i in xrange(cvFold):
			xValid = xTrainCV[i]
			yValid = yTrainCV[i]
			xTrainTemp = xTrainCV[np.arange(cvFold)!=i]
			yTrainTemp = yTrainCV[np.arange(cvFold)!=i]
			xTrainTemp = np.reshape(xTrainTemp,(xValid.shape[0]*(cvFold-1),xValid.shape[1]))
			yTrainTemp = np.reshape(yTrainTemp,(yValid.shape[0]*(cvFold-1),))
			svm = linear_classifier.LinearSVM()
			if i==0:
				svm.train(xTrainTemp,yTrainTemp,learning_rate = alpha, reg = reg,verbose = True, num_iters = 500, batch_size = 500)
			else:
				svm.train(xTrainTemp,yTrainTemp,learning_rate = alpha, reg = reg,verbose = True, num_iters = 500, batch_size = 500)
			yValidPredict = svm.predict(xValid)
			correct = np.sum(yValidPredict==yValid)
			acc = np.append(acc,[float(correct)/xValid.shape[0]])

		print("Alpha = {}, Reg = {} , Acc = {}".format(alpha,reg,np.mean(acc)))
		if np.mean(acc) > accuracy:
			accuracy = np.mean(acc)
			alphaBest = alpha
			regBest = reg


print("Best Alpha = {}, Best Reg = {}, Accuracy = {}".format(alphaBest,regBest,accuracy))
# exit()
svm = linear_classifier.LinearSVM()
svm.train(xTrain,yTrain, batch_size = 2000, verbose = True, num_iters = 1000)
yPredict = svm.predict(xTest)
correct = np.sum(yPredict==yTest)
print("Without cross validation accuracy {}".format(float(correct)/yTest.shape[0]))
svm.train(xTrain,yTrain, batch_size = 2000, verbose = True, num_iters = 1000,learning_rate = alphaBest,reg = regBest)
yPredict = svm.predict(xTest)
correct  = np.sum(yPredict==yTest)
print("With cross validation accuracy = {}".format(float(correct)/yPredict.shape[0]))