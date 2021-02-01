import numpy as np
import pandas as pd
import warnings

class MLP:
    def set_weights(self):
        self.weights_hidden = np.random.rand(self.layers[0], self.layers[1])
        self.bias_hidden = np.random.randn(self.layers[1])
        self.weights_output = np.random.rand(self.layers[1], self.layers[2])
        self.bias_output = np.random.randn(self.layers[2])

    def __init__(self, inputs, hiddenNeurons, outputs, learning_rate):
        self.params={}
        self.learning_rate = learning_rate
        self.error = []
        self.layers = [inputs, hiddenNeurons, outputs]
        self.losses = []
        self.set_weights()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        #x[x==0] = 1e-9
        expX = np.exp(x - np.max(x))
        sum = expX.sum(axis=0, keepdims=True)
        sum[sum==0] = 1e-9

        return expX/sum



    def train(self, inputs, targets):

                zh = np.dot(inputs, self.weights_hidden) + self.bias_hidden
                ah = self.softmax(zh)

                zo = np.dot(ah, self.weights_output) + self.bias_output
                ao = self.softmax(zo)

                dcost_dzo = ao - targets
                dzo_dwo = ah

                dcost_wo = np.dot(dzo_dwo.reshape(-1,1), dcost_dzo.reshape(-1,1).T)
                dcost_bo = dcost_dzo


                dzo_dah = self.weights_output
                dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
                dah_dzh = self.sigmoid_derivative(zh)
                dzh_dwh = inputs
                dcost_wh = np.dot(dzh_dwh.reshape(-1,1), (dah_dzh * dcost_dah).reshape(-1,1).T)
                dcost_bh = dcost_dah * dah_dzh


                self.weights_hidden -= self.learning_rate * dcost_wh
                self.bias_hidden -= self.learning_rate * dcost_bh.sum(axis=0)

                self.weights_output -= self.learning_rate * dcost_wo
                self.bias_output -= self.learning_rate * dcost_bo.sum(axis=0)

                return self




    def predict(self, inputs):
        zh = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        ah = self.softmax(zh)

        zo = np.dot(ah, self.weights_output) + self.bias_output
        ao = self.softmax(zo)
        output = (ao == ao.max(axis=1, keepdims=1)).astype(int)
        return output

    def getWeights_Bias(self):
        return self.weights_hidden, self.bias_hidden, self.weights_output, self.bias_output

    def getTrainingError(self, inputs, targets):
        outputs = self.predict(inputs)
        error = targets - outputs
        error[error != 0] = 1
        error_condensed = error.sum(axis=1)
        n = inputs.shape[0]
        error_condensed[error_condensed > 0] = -1
        error_condensed += 1
        sum = error_condensed.sum()
        return 1 - sum/n

    def checkNan(self):
        if np.isnan(self.weights_hidden).any() or np.isnan(self.bias_hidden).any() or np.isnan(self.weights_output).any() or np.isnan(self.bias_output).any():
            return True
        return False




class MLPPredefined:
    def __init__(self, weights_hidden, bias_hidden, weights_output, bias_output):
        self.weights_hidden = weights_hidden
        self.bias_hidden = bias_hidden
        self.weights_output = weights_output
        self.bias_output = bias_output

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        expX = np.exp(x)
        return expX/expX.sum(axis=0, keepdims=True)

    def predict(self, inputs):
        zh = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        ah = self.sigmoid(zh)

        zo = np.dot(ah, self.weights_output) + self.bias_output
        ao = self.softmax(zo)
        output = ao#(ao == ao.max(axis=1, keepdims=1)).astype(int)
        return output








'''pddata = pd.read_csv("wheat-seeds.csv")
combined = pddata.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
data = combined[:,:-1]
outputs = combined[:,-1:].astype(int)
outputs -= 1
nvalues = np.max(outputs)+1
targets = np.eye(nvalues)[outputs.reshape(-1)]'''

'''pddata = pd.read_csv("train_data.csv")
pdtrain = pd.read_csv("train_labels.csv")
pdcombined = pd.concat([pddata, pdtrain], axis=1)
combined = pdcombined.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
train = combined[:18565, :]
test = combined[18566:, :]
train_data = train[:,:-4]
train_targets = train[:,-4:]
#warnings.simplefilter("ignore")'''

pddata = pd.read_csv("wheat-seeds.csv")
combined = pddata.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
train_data = combined[:,:-1]
outputs = combined[:,-1:].astype(int)
nvalues = np.max(outputs)+1
train_targets = np.eye(nvalues)[outputs.reshape(-1)]




network = MLP(train_data.shape[1], 20, 4, 0.5)
old_error = 1.0

for j in range (1000):
    for i in range(train_data.shape[0]):
        network = network.train(train_data[i], train_targets[i])
        '''new_error = new_network.getTrainingError(train_data, train_targets)
        if (not new_network.checkNan()):
            network = new_network
            old_error = new_error
            #print("{},{},{}".format(old_error, i, j))'''
    print("{}, epoch {}".format(network.getTrainingError(train_data, train_targets), j))



