import numpy as np
import matplotlib.pyplot as plt

class Softmax:
    def __init__(self,input_layer,output_layer,learning_rate):
        self.input = input_layer
        self.output = output_layer
        self.eta = learning_rate
        self.w = np.ones((self.input,self.output))
        self.b = np.ones((1,self.output))

    def predict(self,X):
        P = self.softmax(np.dot(X,self.w) + self.b)
        return P

    def train(self,X,T):
        self.w = self.w - self.eta * self.grad_cross_entropy(X,T)[0]
        self.b = self.b - self.eta * self.grad_cross_entropy(X,T)[1]

    def softmax(self,Z):
        return np.exp(Z)/np.sum(np.exp(Z), axis=1)[:, np.newaxis]

    def grad_cross_entropy(self,X,T):
        P = self.softmax(np.dot(X, self.w) + self.b)
        n = float(len(X))
        dw = np.dot(X.T, P-T)/n
        db = np.dot(np.ones([1,n]),P-T)/n
        return dw,db

def zscore(x):
    xmean = x.mean(keepdims=True)
    xstd  = np.std(x, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

train_data = np.loadtxt("data/mnist_train.csv",delimiter = ",", skiprows=0)
test_data = np.loadtxt("data/mnist_test.csv",delimiter = ",", skiprows=0)

def main(train_data,test_data):
    input_layer = 784
    output_layer = 10
    learning_rate = 0.25
    batch_size = 900
    s = Softmax(input_layer,output_layer,learning_rate)
    #data
    train_label = train_data[:,0]
    train_feature = zscore(train_data[:,1:])
    test_label = test_data[:,0]
    test_feature = zscore(test_data[:,1:])

    batch = int(np.floor(len(train_data)/batch_size))
    for k in range(1):
        for j in range(batch):
            if j == batch-1:
                train_feature_batch = train_feature[batch_size*j:,:]
                train_label_batch = train_label[batch_size*j:]
            else:
                train_feature_batch = train_feature[batch_size*j:batch_size*(j+1), :]
                train_label_batch = train_label[batch_size*j:batch_size*(j+1)]
            targets = np.zeros((len(train_label_batch),output_layer))
            for i in range(len(train_label_batch)):
                targets[i][int(train_label_batch[i])] = 1
            s.train(train_feature_batch,targets)
    counter = 0
    predict = s.predict(test_feature)
    for i in range(len(predict)):
        predicted_label = np.argmax(predict[i])
        if predicted_label == test_label[i]:
            counter = counter + 1
    print("accuracy rate = {}".format(float(counter)/float(len(test_data))))
    return float(counter)/float(len(test_data))

main(train_data,test_data)
