import numpy as np
import matplotlib.pyplot as plt
import math

def loadDataSet(path):
    dataMat = []
    labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split(",")
        dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    maxCycle = 500
    alpha = 0.001
    weights = np.ones((n,1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix*weights)
        error = labelMat - h
        weights = weights+alpha*dataMatrix.T*error
    return weights


def drawScatter(Input,LabelMat):
    m,n = np.shape(Input)
    print(m,n)
    data = np.array(Input)
    for i in range(m):
        if LabelMat[i] == 1:
            plt.scatter(data[i,1],data[i,2],c='blue',marker='o')
        else:
            plt.scatter(data[i, 1], data[i, 2], c='red', marker='s')


dataArr,labelMat = loadDataSet("testSet.txt")
drawScatter(dataArr,labelMat)
w = gradAscent(dataArr,labelMat)
X = np.linspace(-3,3,100)
Y = -(float(w[0])+float(w[1])*X)/float(w[2])
plt.plot(X,Y)
plt.show()
print(w)