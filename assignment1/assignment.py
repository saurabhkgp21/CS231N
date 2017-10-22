import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor


cifar10 = './cs231n/datasets/cifar-10-batches-py'
xTrain,yTrain,xTest,yTest = load_CIFAR10(cifar10)

lengthTrain = 5000
lengthTest = 500
xTrain = xTrain[:lengthTrain]
yTrain = yTrain[:lengthTrain]
xTrainOrgnl = xTrain
yTrainOrgnl = yTrain
xTest = xTest[:lengthTest]
yTest = yTest[:lengthTest]
xTrain = np.reshape(xTrain,(lengthTrain,xTrain.shape[1]*xTrain.shape[2]*xTrain.shape[3]))
xTest = np.reshape(xTest,(lengthTest,xTest.shape[1]*xTest.shape[2]*xTest.shape[3]))

clsfr = KNearestNeighbor()
cvFold = 5
# kValue = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
kValue = np.random.random_integers(1,100,10)
kValue = kValue[np.argsort(kValue)]
kValue = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
xTrain = np.array(np.split(xTrain,cvFold))
yTrain = np.array(np.split(yTrain,cvFold))


kAccuracies = []
# print(xTrain)
for ptr,k in enumerate(kValue):
	kValueAcc = []
	for i in xrange(0,cvFold):
		xValid = xTrain[i]
		yValid = yTrain[i]
		xTrainCV = xTrain[np.arange(cvFold)!=i]
		yTrainCV = yTrain[np.arange(cvFold)!=i]
		xTrainCV = np.reshape(xTrainCV,(lengthTrain - lengthTrain/cvFold,xTrainCV.shape[2]))
		yTrainCV = np.reshape(yTrainCV,(lengthTrain - lengthTrain/cvFold,))
		clsfr.train(xTrainCV,yTrainCV)
		yPredict = clsfr.predict(xValid,k=k)
		acc = np.sum(yPredict==yValid)
		print(k,acc)
		# print(k)
		# print(yPredict)
		# print(yValid)
		kValueAcc.append([float(acc)/(lengthTrain/cvFold)])
		# exit()
	kAccuracies.append(kValueAcc)

print([np.mean(i) for i in kAccuracies])
plt.figure()
x = np.array(kValue)
y = np.array([np.mean(i) for i in kAccuracies])
print(x.shape)
print(y.shape)
plt.errorbar(np.array(kValue),np.array([np.mean(i) for i in kAccuracies]), yerr = np.array([np.std(i) for i in kAccuracies]))
plt.show()



clsfr.train(xTrain,yTrain)
yPredict = clsfr.predict(xTest)
# yPredict = clsfr.predict_labels(dists,k=1)
# plt.imshow(dists, interpolation='none')
# plt.show()
acc = np.sum(yPredict==yTest)
print("Accuracy using k=1",float(acc)/lengthTest)
