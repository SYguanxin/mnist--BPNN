import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt


# 特征缩放类(归一法
class Feature_scaling:
    def __init__(self, train):
        self.min = np.min(train)
        self.ptp = np.ptp(train)

    def __str__(self):
        return f"avr = {self.avr}, ptp = {self.ptp}"

    def initial_feature(self, feature):
        m = feature.shape[0]
        a0 = np.ones(m).reshape(m, 1)
        return np.hstack((a0, np.divide(np.subtract(feature, self.min), self.ptp)))

    def new_data_updating(self, test):
        m = test.shape[0]
        a0 = np.ones(m).reshape(m, 1)
        return np.hstack((a0, np.divide(np.subtract(test, self.min), self.ptp)))


class DATA:
    trainX = None
    trainLabelOrin = None
    trainLabel = None
    trainM = None
    PixelN = None
    testX = None
    testLabelOrin = None
    testLabel = None
    testM = None
    Fs = None
    def read_data(self, path, split=False):
        train = pd.read_csv(path)
        if split:
            train,test = self.train_test_split(train, proportion=0.8)
        self.trainX = np.asarray(train.drop('label', axis=1))
        self.trainLabelOrin = np.array(train.loc[:, 'label'])
        self.trainLabel = np.eye(10, dtype=np.uint8)[self.trainLabelOrin]  # 获取label的onehot编码矩阵
        self.trainM, self.PixelN = self.trainX.shape
        print("样本个数和像素数为:",self.trainM,self.PixelN)
        self.Fs = Feature_scaling(self.trainX)
        self.trainX = self.Fs.initial_feature(self.trainX)
        if split:
            self.testX = np.asarray(test.drop('label', axis=1))
            self.testLabelOrin = np.array(test.loc[:, 'label'])
            self.testLabel = np.eye(10, dtype=np.uint8)[self.testLabelOrin]
            self.testX = self.Fs.new_data_updating(np.asarray(self.testX))
            self.testM = self.testX.shape[0]

    def train_test_split(self, data, proportion=0.7, random_seed=None):
        np.random.seed(random_seed)
        test_size = int((1-proportion) * len(data))
        index = np.random.choice(np.arange(len(data)), size=test_size, replace=False)
        train = data.drop(index=list(index),axis=0)
        test = data.loc[index]
        return train, test


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        out = np.reciprocal((1 + np.exp(-z)))
        return out

    def backward(self, a):
        dx = (1.0 - a) * a
        return dx


class LeakyRelu:
    def __init__(self,alpha):
        self.alpha = alpha
    def forward(self, x):
        mask = (x <= 0)
        out = x.copy()
        out[mask] = x[mask] * self.alpha
        return out
    def backward(self, a):
        dx = np.where(a >= 0, 1, self.alpha)
        return dx


class BPNN:
    def __init__(self, layerDev, batchSize, epsilon=1, learingRate=0.08, regParam=0.01):
        self.layerL = len(layerDev)
        self.layerDev = [0] + layerDev
        self.thetaDev = [0, 0] + [np.subtract(np.multiply( \
            np.random.rand(layerDev[l + 1], layerDev[l] + 1), \
            2 * epsilon), epsilon) \
            for l in range(self.layerL - 1)]
        self.dwDev = [0, 0] + [np.zeros((layerDev[l + 1], layerDev[l] + 1)) \
                               for l in range(self.layerL - 1)]
        self.AF = LeakyRelu(0.1)
        # self.AF = Sigmoid()
        self.learingRate = learingRate
        self.regParam = regParam
        self.batchSize = batchSize

    def computing_a(self, X):
        a, m = X, X.shape[0]
        a = a.reshape(len(a), 1)
        aDev = [0, a]
        for l in range(1, self.layerL-1):
            z = np.dot(self.thetaDev[l + 1],a)
            a = self.AF.forward(z)
            if l != self.layerL - 1:
                a = np.vstack((1, a.reshape(len(a),1)))
            a = a.reshape(len(a),1)
            aDev.append(a.copy())
        # 最后一层用Softmax
        z = np.dot(self.thetaDev[self.layerL],a)
        z = np.exp(z-np.max(z))
        aDev.append(z/np.sum(z))
        return aDev

    def costFunc(self, X, label, thetaDevOrin=None):
        if thetaDevOrin is None:
            thetaDev = self.thetaDev
        else:
            thetaDev = [0, 0]
            for l in range(2, self.layerL+1):
                thetaDev.append(thetaDevOrin[:self.layerDev[l]*(self.layerDev[l-1]+1)] \
                                .reshape(self.layerDev[l],self.layerDev[l-1]+1))
                thetaDevOrin = np.delete(thetaDevOrin,slice(0,self.layerDev[l]*(self.layerDev[l-1]+1)))
        m = X.shape[0]
        a = []
        for i in range(m):
            a.append(self.computing_a(X[i])[-1].reshape(self.layerDev[-1]))
        a = np.array(a)
        # sqrErrors = -np.matmul(label.T, np.log(a)) - np.matmul((1 - label).T,np.log(1 - a))
        a[(a<=0)] = 0.00001
        sqrErrors = -np.matmul(label.T,np.log(a))
        J = np.sum(sqrErrors/m)
        reg = 0
        for l in range(2,self.layerL+1):
            reg += np.sum(thetaDev[l]**2) * (self.regParam /2)
        J += reg
        return J

    def backpropagation(self, X, label):
        aDev = self.computing_a(X)
        deltaDev = list("0" * (self.layerL + 1))
        deltaDev[-1] = aDev[-1] - label
        self.dwDev[-1] += np.matmul(deltaDev[-1],  aDev[self.layerL - 1].T)/self.batchSize
        for l in range(self.layerL - 1, 1, -1):
            delta = deltaDev[l + 1]
            # 除了最后一层 计算时偏置节点不传播误差
            if l < self.layerL - 1:
                delta = delta[1:]
            deltaDev[l] = np.dot(self.thetaDev[l + 1].T, delta) * np.vstack(([1], self.AF.backward(aDev[l][1:])))
            self.dwDev[l] += np.matmul(deltaDev[l][1:], aDev[l - 1].T)

    def update(self):
        for l in range(2, self.layerL + 1):
            self.dwDev[l][:,1:] += self.thetaDev[l][:,1:] * self.regParam
            self.thetaDev[l] -= self.learingRate * self.dwDev[l]

    def gradientCheck(self, X, label, Epsilon):
        thetaDev = self.thetaDev.copy()
        dwDev = self.dwDev.copy()
        wDev = []
        for l in range(2,self.layerL+1):
            thetaDev[l] = np.ravel(thetaDev[l])
            dwDev[l] = np.ravel(dwDev[l])
        thetaDev = np.concatenate(thetaDev[2:])
        dwDev = np.concatenate(dwDev[2:])
        for i,x in enumerate(thetaDev):
            print(f"\r{i}",end='')
            thetaDevPE = thetaDev.copy()
            thetaDevPE[i] += Epsilon
            JPE = self.costFunc(X,label,thetaDevPE)
            thetaDevPE[i] -= 2*Epsilon
            JSE = self.costFunc(X,label,thetaDevPE)
            wDev.append((JPE-JSE)/2*Epsilon)
        print("\n",np.max(dwDev),np.argmax(dwDev), np.min(dwDev))
        maxindex = np.argmax(dwDev-np.array(wDev))
        print((dwDev-np.array(wDev))[maxindex],maxindex,dwDev[maxindex],wDev[maxindex])

    def train(self, X, label, gradientCheck=False):
        m, pixelN = X.shape
        for i in np.random.randint(m,size=(self.batchSize,)):
        # for i in range(0,2):
            self.backpropagation(X[i], label[i].reshape(10,1))
        self.update()
        if gradientCheck:
            self.gradientCheck(X[0:2],label[0:2],Epsilon=0.00001)
        self.dwDev = [0, 0] + [np.zeros((self.layerDev[l + 2], self.layerDev[l + 1] + 1)) \
                               for l in range(self.layerL - 1)]


def accuracy(X,label,m,answer=False):
    resultList = []
    accuracy = 0
    for i in range(m):
        res = np.argmax(BPNN.computing_a(X[i])[-1])
        resultList.append(res)
        if res == label[i]:
            accuracy += 1
    if answer:
        ans = pd.DataFrame(resultList,columns=['Label'],index=range(1,len(resultList)+1))
        ans.to_csv('./datafile/answer.csv',index_label=['ImageId'])
    return accuracy/m


timeStart = time()
Data = DATA()
Data.read_data('./datafile/train.csv',True)
BPNN = BPNN(layerDev=[Data.PixelN,200,10], learingRate=0.08, batchSize=64)

epochs = 10000
times = 100
acc = []
for i in range(1,epochs+1):
    process = int((i%(epochs//times))//(epochs/times/100))
    print(f"\r{i: >5} {'='*process}{'-'*(100-process)}",end='')
    # print("代价函数为:", BPNN.costFunc(Data.trainX, Data.trainLabel))
    BPNN.train(Data.trainX,Data.trainLabel,False)
    # BPNN.learingRate *= epochs/100
    # if i == 100:
    #     BPNN.learingRate = 0.03
    BPNN.learingRate *= 0.99995
    if not i%(epochs//times):
        temp = accuracy(Data.testX,Data.testLabelOrin,Data.testM)
        print(f"\repoch={i: <4}  costFunc=",
              f"{BPNN.costFunc(Data.trainX, Data.trainLabel): <18}",
              "  accuracy=",temp)
        # if temp - acc < 0.001:
        #     acc = temp
        #     break
        acc.append(temp)

print("\n样本准确率为:", accuracy(Data.trainX,Data.trainLabelOrin,Data.trainM))
print("验证准确率为:", accuracy(Data.testX,Data.testLabelOrin,Data.testM))
testLabel = pd.read_csv('./datafile/testlabel.csv')
testLabel = np.asarray(testLabel.drop(columns='ImageId'))
testX = np.hstack((np.ones((len(testLabel),1)),np.asarray(pd.read_csv('./datafile/test.csv'))))
print("提交准确率为:",accuracy(testX,testLabel,len(testLabel),True))
timeEnd = time()
print("用时",timeEnd-timeStart)
plt.plot(np.arange(times),acc)
plt.show()