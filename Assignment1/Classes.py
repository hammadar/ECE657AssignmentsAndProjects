import math
import numpy as np
import pandas as pd

np.random.seed(1)


class Network:
    def __init__(self, inputs, targets, hiddenNeurons, learning):
        self.inputs = inputs
        input_addition = np.full((self.inputs.shape[0], 1), 1)
        self.inputs =  np.concatenate((self.inputs, input_addition), axis=1)
        self.inputWeights = np.random.rand(self.inputs.shape[1], hiddenNeurons)
        self.hiddenWeights = np.random.rand(hiddenNeurons, 4)
        self.hiddenWeights_2 = np.concatenate((np.copy(self.hiddenWeights), np.random.rand(1,4)), axis=0)
        '''np.random.random((hiddenNeurons, 4))'''
        '''np.full((hiddenNeurons, 4), 0.5)'''
        '''np.random.rand(hiddenNeurons, 4)'''
        '''np.random.rand(self.inputs.shape[1], hiddenNeurons)'''
        self.targets = targets
        self.learning = learning

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def softmax_derivative(self, outputs, targets, hiddenLayerOutputs):
        error = outputs - targets
        error = error.reshape(-1,1)


        return error


    def softmax(self, x):
        # x[x==0] = 1e-9
        expX = np.exp(x - np.max(x))
        sum = expX.sum(axis=0, keepdims=True)
        sum[sum == 0] = 1e-9

        return expX / sum #np.divide?


    def lossFunction(self):
        return -0.5 * (self.error) * self.sigmoid_derivative(self.output) #np.multiply?

    def lossFunctionSoftMax(self):

        return -self.softmax_derivative(self.output, self.target, self.sMhidden_layer)


    def propogate(self, input, target):
        self.input = input
        input_addition = np.full((1, ), 1)
        self.input = np.concatenate((self.input, input_addition), axis=0)
        self.hiddenLayer = np.dot(self.input, self.inputWeights)
        hiddenLayer_addition = np.full((1,), 1)
        self.hiddenLayer_2 = np.concatenate((self.hiddenLayer, hiddenLayer_addition), axis=0)
        self.sMhidden_layer = self.softmax(self.hiddenLayer_2)
        self.outputLayer = np.dot(self.sMhidden_layer, self.hiddenWeights_2)
        self.output = self.softmax(self.outputLayer)
        self.target = target
        self.error = self.target - self.output

    def backPropogate(self):
        outputSignal = self.lossFunctionSoftMax()
        dHiddenWeights = self.learning * np.dot(self.softmax(self.hiddenLayer_2.reshape(-1,1)), outputSignal.T)
        hiddenSignal = np.dot(outputSignal.T, self.hiddenWeights.T) * self.sigmoid_derivative(self.hiddenLayer)
        dInputWeights = self.learning * np.dot(self.input.reshape(-1,1), hiddenSignal)
        self.inputWeights += dInputWeights
        self.hiddenWeights += dHiddenWeights[:-1,:]
        self.hiddenWeights_2 += dHiddenWeights

    def predict(self, inputs):
        input_addition = np.full((inputs.shape[0], 1), -1)
        inputs = np.concatenate((inputs, input_addition), axis=1)
        hiddenLayer = np.dot(inputs, self.inputWeights)
        hiddenLayer_addition = np.full((hiddenLayer.shape[0], 1), -1)
        hiddenLayer = np.concatenate((hiddenLayer, hiddenLayer_addition), axis=1)
        outputLayer = np.dot(self.softmax(hiddenLayer), self.hiddenWeights_2)
        outputs = self.softmax(outputLayer)
        #outputs = (outputs == outputs.max(axis=1, keepdims=1)).astype(float)
        return np.round(outputs)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

pddata = pd.read_csv("train_data.csv")
pdtrain = pd.read_csv("train_labels.csv")
pdcombined = pd.concat([pddata, pdtrain], axis=1)
combined = pdcombined.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
train = combined[:18565, :]
test = combined[18566:, :]
data = train[:, :-4]
targets = train[:, -4:]

'''pddata = pd.read_csv("wheat-seeds.csv")
combined = pddata.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
data = combined[:,:-1]
outputs = combined[:,-1:].astype(int)
nvalues = np.max(outputs)+1
targets = np.eye(nvalues)[outputs.reshape(-1)]'''




network = Network(data, targets, 75, 0.6)

while True:

    for i in range(data.shape[0]):
        network.propogate(data[i,:], targets[i,:])
        network.backPropogate()

    x = network.predict(data)
    error = rmse(x, targets)
    print("Error is : {}".format(error))
    if error <= 0.1:
        break


'''while rmse(network.outputs, network.targets) > 0.1:
    print(rmse(network.outputs, network.targets))
    network.propogate()
    network.backPropogate()'''
