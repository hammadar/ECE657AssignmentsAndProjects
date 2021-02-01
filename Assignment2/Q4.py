import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import exp

'''class SOMNN:
    def __init__(self, layers, seed):
        self.colours = np.array([
            [0,0,255],
            [51,51,255],
            [51,153,255],
            [102,178,255],
            [153,204,255],
            [0,153,76],
            [0,204,102],
            [0,255,128],
            [51,255,153],
            [51,255,51],
            [153,0,0],
            [204,0,0],
            [255,0,0],
            [255,51,51],
            [102,0,0],
            [204,204,0],
            [255,255,0],
            [255,255,51],
            [0,153,153],
            [0,102,102],
            [0,204,204],
            [255,204,255],
            [255,153,255],
            [255,102,255]
        ])

        self.colours_norm = self.colours/255
        self.layers = layers
        self.seed = seed

    def setup(self):
        np.random.seed(self.seed)
        self.weights = np.random.uniform(0,1,(3,self.layers[0], self.layers[1]))#np.random.rand(3,self.layers[0],self.layers[1])

    def train(self, epochs, sigma0):
        for epoch in range(epochs):
            distances = np.empty((3,self.layers[0], self.layers[1]))
            euc_distances = np.empty((self.layers[0], self.layers[1]))
            for k in range(self.colours_norm.shape[0]):
                for i in range(self.layers[0]):
                    for j in range(self.layers[1]):
                        weight = self.weights[:,i,j]
                        distances[:,i,j] = self.distance(self.colours_norm[k], weight)
                        euc_distances[i,j] = self.euc_distance(self.colours_norm[k], weight)**2

                min_dist = euc_distances.min()
                min_index = np.where(euc_distances == min_dist)
                min_i = min_index[0][0]
                min_j = min_index[1][0]
                d_neighbourhood = np.empty((self.layers[0], self.layers[1]))
                #winning_distance =  np.empty((self.layers[0], self.layers[1]))
                min_weight = self.weights[:, min_i, min_j]
                #weight_distance = self.colours_norm[k] - self.weights[min_index[0]][min_index[1]]
                for i in range(self.layers[0]):
                    for j in range(self.layers[1]):
                        learning_rate = self.learning_rate(epoch, epochs)
                        neighbourhood = self.neighbourhood(epoch, np.array([min_i,min_j]), np.array([i,j]),sigma0, epochs)
                        #neighbourhood = self.neighbourhood(epoch, min_weight, self.weights[:,i,j], sigma0, epochs)
                        dw_ddist= learning_rate * neighbourhood
                        d_neighbourhood[i,j]=dw_ddist
                        #winning_distance[i,j] = self.distance(self.colours_norm[k], self.)
                #winning_distance = self.distance(self.colours_norm[k], min_weight)
                dw = d_neighbourhood * (distances) #* winning_distance
                self.weights[:,:,:] += dw
            if epoch in [0, 20, 40, 100, 999]:
                self.plot(sigma0, epoch)



    def distance(self, input, weight):
        distance = (input-weight)
        return distance

    def euc_distance(self,input,weight):
        distance = np.linalg.norm(input-weight, 2)
        return distance


    def learning_rate(self, epoch, total_epochs):
        return 0.8 * exp(-epoch/total_epochs)

    def neighbourhood(self, epoch, i, j, sigma0, total_epochs):
        neighbourhood = exp(-self.euc_distance(i,j)**2/(2*self.sigma(epoch, sigma0, total_epochs)**2))
        return neighbourhood

    def sigma(self, epoch, sigma0, total_epochs):
        return sigma0*exp(-epoch/total_epochs)

    def plot(self, sigma, epoch):
        plt.xlabel("Neurons D1")
        plt.ylabel("Neurons D2")
        plt.title("Sigma - {}, Epoch - {}".format(sigma, epoch))
        plt.imshow(self.weights.T)
        plt.show()


somnn = SOMNN([100,100], 25)

for sigma0 in [1,10,30,50,70]:
    somnn.setup()
    somnn.train(1000, sigma0)'''


class SOMNN:
    def __init__(self, layers, seed):
        self.colours = np.array([ #random bunch of colours in the range specified
            [0, 0, 255],
            [51, 51, 255],
            [51, 153, 255],
            [102, 178, 255],
            [153, 204, 255],
            [0, 153, 76],
            [0, 204, 102],
            [0, 255, 128],
            [51, 255, 153],
            [51, 255, 51],
            [153, 0, 0],
            [204, 0, 0],
            [255, 0, 0],
            [255, 51, 51],
            [102, 0, 0],
            [204, 204, 0],
            [255, 255, 0],
            [255, 255, 51],
            [0, 153, 153],
            [0, 102, 102],
            [0, 204, 204],
            [255, 204, 255],
            [255, 153, 255],
            [255, 102, 255]
        ])

        self.colours_norm = self.colours / 255
        self.layers = layers
        self.seed = seed

    def setup(self): #re-initialise weights (done for every new sigma)
        np.random.seed(self.seed)
        self.weights = np.random.uniform(0, 1, (
        3, self.layers[0], self.layers[1]))  # np.random.rand(3,self.layers[0],self.layers[1])

    def train(self, epochs, sigma0): #method to train
        for epoch in range(epochs): #for one epoch
            distances = np.empty((3, self.layers[0], self.layers[1])) #empty array to hold vector of differences between input and all weights
            euc_distances = np.empty((self.layers[0], self.layers[1])) #empty array to hold euclidean distances between input and all weights -> for use in neighbourhood function
            if epoch in [0, 20, 40, 100]:
                self.plot(sigma0, epoch)
            for k in range(self.colours_norm.shape[0]): #for each input
                distances = self.distance(self.colours_norm[k].reshape((3, 1, 1)), self.weights)
                euc_distances = self.euc_distance(self.colours_norm[k].reshape((3, 1, 1)), self.weights)
                min_dist = euc_distances.min() #get min distance, ie winning node
                min_index = np.where(euc_distances == min_dist) #get index of winning node
                min_i = min_index[0][0]
                min_j = min_index[1][0]
                min_array = np.array((min_i, min_j)).reshape(2, 1, 1) #reshape array for use in operations below
                coordinate_array = np.indices((100, 100)) #array to hold 2D coordinates of all nodes for use in neighbourhood calculation
                learning_rate = self.learning_rate(epoch, epochs) #scalar learning rate
                neighbourhood = self.neighbourhood(epoch, min_array, coordinate_array, sigma0, epochs) #get value of neighbourhood function
                d_neighbourhood = learning_rate * neighbourhood
                dw = d_neighbourhood * (distances)
                self.weights += dw
            if epoch == 999:
                self.plot(sigma0, epoch)

    def distance(self, input, weight):
        distance = (input - weight)
        return distance

    def euc_distance(self, input, weight):
        temp = input - weight
        distance = np.linalg.norm(temp, 2, axis=0)
        return distance

    def learning_rate(self, epoch, total_epochs):
        return 0.8 * exp(-epoch / total_epochs)

    def neighbourhood(self, epoch, i, j, sigma0, total_epochs):
        neighbourhood = np.exp(-self.euc_distance(i, j) ** 2 / (2 * self.sigma(epoch, sigma0, total_epochs) ** 2))
        return neighbourhood

    def sigma(self, epoch, sigma0, total_epochs):
        return sigma0 * exp(-epoch / total_epochs)

    def plot(self, sigma, epoch):
        plt.xlabel("Neurons D1")
        plt.ylabel("Neurons D2")
        plt.title("Sigma - {}, Epoch - {}".format(sigma, epoch))
        plt.imshow(self.weights.T)
        plt.savefig("Sigma-{}, Epoch-{}.png".format(sigma, epoch))
        plt.show()




somnn = SOMNN([100,100], 25)

for sigma0 in [1,10,30,50,70]: #change values in sigma here to see different values
    somnn.setup()
    somnn.train(1000, sigma0)