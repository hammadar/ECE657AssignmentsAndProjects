import numpy as np
import pandas as pd

class MLP():

    def __init__(self, inputs, hiddenNeurons, outputs, learning_rate):
        self.params={}
        self.learning_rate = learning_rate
        self.error = []
        self.layers = [inputs, hiddenNeurons, outputs]
        self.losses = []


    def set_weights(self):

        np.random.seed(10)
        self.params["Wi"] = np.random.random((self.layers[0], self.layers[1]))
        self.params["bi"] = np.random.random((self.layers[1],))
        self.params["Wh"] = np.random.random((self.layers[1], self.layers[2]))
        self.params["bh"] = np.random.random((self.layers[2],))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1-x)

    def loss(self, outputs, targets):
        n = len(outputs) #might need to change this to shape[0]
        v1 = np.log(targets)
        v2 = np.multiply(v1, outputs.T)
        v3 = np.log(1-targets)
        v4 = 1-outputs.T
        v5 = np.multiply(v4,v3)
        v6 = np.sum(v2+v5)


        loss = -1/n * v6
        return loss

    def softmax(self, x):
        e = np.exp(x) / np.exp(x).sum()#[:, None]
        return e

    def dsoftmax(self, x, y):
        '''X = self.softmax(x)
        y = y.argmax()
        m = 1
        grad = self.softmax(X)
        grad[range(m), y] -= 1
        grad = grad / m'''
        s = x.reshape(-1,1)
        w = np.multiply(s, 1-s) + sum (-s * np.roll(s, i, axis=1) for i in range(s.shape[1]))
        return w

    def propogate(self, input, y):
        hidden_layer = input.dot(self.params["Wi"]) + self.params['bi']
        Ah = self.sigmoid(hidden_layer)
        output_layer = Ah.dot(self.params["Wh"]) + self.params["bh"]
        outputs = self.sigmoid(output_layer)
        error = self.loss(outputs, y)

        self.params["hidden_layer"] = hidden_layer
        self.params["output_layer"] = output_layer
        self.params["Ah"] = Ah

        return outputs, error

    def back_propogate(self, outputs, y, input):

        dE_doutputs = -(np.divide(y, outputs) - np.divide((1-y), (1-outputs)))
        dE_dsoftmax = self.dsigmoid(outputs)
        dE_doutputLayer = dE_doutputs * dE_dsoftmax

        dE_dAh = dE_doutputLayer.dot(self.params["Wh"].T)
        dE_dWh = self.params["Ah"].reshape(-1,1).dot(dE_doutputLayer.reshape(-1,1).T)
        dE_dbh = np.sum(dE_doutputLayer, axis=0)

        dE_dhiddenLayer = dE_dAh * self.dsigmoid(self.params["hidden_layer"])
        dE_dWi = input.T.dot(dE_dhiddenLayer)
        dE_dbi = np.sum(dE_dWi, axis=0)

        #outputSignal = self.loss(outputs, y) * self.dsoftmax(outputs, y)
        #dHiddenWeights = self.learning_rate * np.dot()

        self.params["Wi"] -= self.learning_rate * dE_dWi
        self.params["Wh"] -= self.learning_rate * dE_dWh
        self.params["bi"] -= self.learning_rate * dE_dbi
        self.params["bh"] -= self.learning_rate * dE_dbh


    def train(self, X,y):
        self.set_weights()

        for i in range(X.shape[0]):
            targets = y[i,:]
            inputs = X[i,:]
            outputs, loss = self.propogate(inputs, targets)
            self.back_propogate(outputs, targets, inputs)
            self.losses.append(loss)

    def predict(self, X):
        hidden_layer = X.dot(self.params["Wi"]) + self.params["bi"]
        Ah = self.sigmoid(hidden_layer)
        output_layer = Ah.dot(self.params["Wh"]) + self.params["bh"]
        pred = self.softmax(output_layer)
        pred[np.where(pred == np.max(pred))] = 1
        pred[np.where(pred != 1)] = 0
        return pred


pddata = pd.read_csv("train_data.csv")
pdtrain = pd.read_csv("train_labels.csv")
pdcombined = pd.concat([pddata, pdtrain], axis=1)
combined = pdcombined.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
train = combined[:10, :]
test = combined[18566:, :]

mlp = MLP(784, 784, 4, 0.05)
mlp.train(train[:,:-4], train[:,-4:])
