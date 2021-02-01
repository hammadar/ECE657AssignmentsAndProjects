import numpy as np
import pandas as pd
import pickle as pkl

class MLP:
    def __init__(self, layers_neurons):
        self.layers_neurons = layers_neurons
        self.params = {}
        self.L = len(self.layers_neurons) #normally 2, but allows for more layers for future use
        self.n = 0
        self.outputs = None


    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def softmax(self, x):
        expX = np.exp(x - np.max(x))
        sum = expX.sum(axis=0, keepdims=True)
        sum[sum == 0] = 1e-9 #necessary to remove divide by 0, which introduces nan in weights

        return expX / sum  

    def set_params(self):
        np.random.seed(10)

        for i in range(1, self.L+1):
            self.params["W"+str(i)] = np.random.randn(self.layers_neurons[i], self.layers_neurons[i-1]) / np.sqrt(self.layers_neurons[i-1]) #set input and hidden weights for training
            self.params["b"+str(i)] = np.zeros((self.layers_neurons[i],1)) #set input and hidden biases to 0

    def propogate(self, X):
        values = {}

        A = X.T #for dot product use below

        for i in range(self.L - 1): #propogate forward for all hidden layers, just one in this case
            Z = self.params["W"+str(i+1)].dot(A) + self.params["b"+str(i+1)] #value of each neuron in hidden layer
            A = self.softmax(Z) #output of each neuron in hidden layer
            values["A" + str(i+1)] = A
            values["W" + str(i+1)] = self.params["W"+str(i+1)]
            values["Z"+str(i+1)] = Z # for use in backpropogation

        Z = self.params["W"+str(self.L)].dot(A) + self.params["b"+str(self.L)] #value of output layer
        A = self.softmax(Z) #output of output layer
        values["A" + str(self.L)] = A
        values["W" + str(self.L)] = self.params["W" + str(self.L)]
        values["Z" + str(self.L)] = Z #for use in backpropogation

        return A, values

    def sigmoid_derivative(self, X):
        s = self.sigmoid(X)
        return s * (1-s)

    def loss(self, outputs, Y):
        return -np.mean(Y * np.log(outputs.T + 1e-9)) #cross entropy loss function to determine error

    def backpropogate(self, X, Y, values):
        derivatives = {}
        values["A0"] = X.T
        A = values["A" + str(self.L)] #output of final layer
        dL_dZ = A - Y.T #cross entropy loss with softmax
        dL_dW = dL_dZ.dot(values["A"+str(self.L-1)].T)/self.n #delta for weights = loss signal * output of hidden layer, normalised
        dL_db = np.sum(dL_dZ, axis=1, keepdims = True) / self.n #delta for bias is loss function
        dZ_dAPrev = values["W" + str(self.L)].T.dot(dL_dZ) #for use in hidden layer loss signals

        derivatives["W" + str(self.L)] = dL_dW
        derivatives["b"+str(self.L)] = dL_db #store for use in parameter updates

        for i in range(self.L - 1, 0, -1):
            dL_dZ = dZ_dAPrev * self.sigmoid_derivative(values["Z"+str(i)]) #even though activation function is softmax, derivations found online indicate that sigmoid derivative is used for hidden layer back prop
            dL_dW = 1. / self.n * self.n * dL_dZ.dot(values["A"+str(i-1)].T)
            dL_db = 1. / self.n * np.sum(dL_dZ, axis=1, keepdims=True)
            if i > 1: #no need for further signal backpropogation once you reach input layer
                dZ_dAPrev = values["W"+str(i)].T.dot(dL_dZ)
            derivatives["W"+str(i)] = dL_dW
            derivatives["b"+str(i)] = dL_db

        return derivatives

    def train(self, X, Y, learning_rate, epochs):
        np.random.seed(10)

        if not bool(self.params): #allows multiple training commands with different learning rates
            self.n = X.shape[0]
            self.layers_neurons.insert(0, X.shape[1])
            self.set_params()



        for epoch in range(epochs):
            self.outputs = None
            for i in range(0,X.shape[0],2): #trained in pairs of two inputs to increase speed a little
                x = X[i:i+2,:]
                y = Y[i:i+2,:]
                A, values = self.propogate(x)
                if self.outputs is None:
                    self.outputs = A
                else:
                    self.outputs = np.concatenate((self.outputs, A), axis=1) #necessary to keep track of all outputs at each epoch to determine loss
                derivatives = self.backpropogate(x, y, values)

                for i in range(1, self.L + 1): #updates weights and biases
                    self.params["W"+str(i)] = self.params["W"+str(i)] - learning_rate * derivatives["W"+str(i)]
                    self.params["b"+str(i)] = self.params["b"+str(i)] - learning_rate * derivatives["b"+str(i)]
            loss = self.loss(self.outputs,Y) #cross entropy loss to help determine when to stop training
            print("Loss: ", loss)

        return self

    def predict(self, X):
        pred = self.propogate(X)[0]
        pred = pred.T
        pred1 = np.zeros_like(pred)
        pred1[np.arange(len(pred)), pred.argmax(1)] = 1 #replace max probability value with 1, rest with 0
        #accuracy = 1 - self.loss(X,Y)
        return pred1

'''pddata = pd.read_csv("train_data.csv")
pdtrain = pd.read_csv("train_labels.csv")
pdcombined = pd.concat([pddata, pdtrain], axis=1)
combined = pdcombined.to_numpy()
np.take(combined, np.random.permutation(combined.shape[0]), axis=0, out=combined)
train = combined[:18565, :]
test = combined[18566:, :]
data = train[:, :-4]
targets = train[:, -4:]
test_data = test[:,:-4]
test_labels = test[:,-4:]

layers = [45,4]
network = MLP(layers)

network = network.train(data, targets, 0.6,400)'''



