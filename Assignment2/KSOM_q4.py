import numpy as np
import sys
from numpy import random
from math import exp
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

def calibrateColorCode(colorCode):
    return float(colorCode/255.0)

colour_inputs = np.array([[calibrateColorCode(255.0),   0,   0], # red
                    [calibrateColorCode(245.0),   0,   0],
                    [calibrateColorCode(235.0),   0,   0],
                    [calibrateColorCode(225.0),   0,   0],
                    [calibrateColorCode(215.0),   0,   0],
                    [calibrateColorCode(190.0),   0,   0],
                    [  0, calibrateColorCode(255.0),   0], # green
                    [  0, calibrateColorCode(245.0),   0],
                    [  0, calibrateColorCode(235.0),   0],
                    [  0, calibrateColorCode(225.0),   0],
                    [  0, calibrateColorCode(215.0),   0],
                    [  0, calibrateColorCode(190.0),   0],  
                    [  0,   0, calibrateColorCode(255.0)], # index 2: blue
                    [  0,   0, calibrateColorCode(245.0)],
                    [  0,   0, calibrateColorCode(235.0)],
                    [  0,   0, calibrateColorCode(225.0)],
                    [  0,   0, calibrateColorCode(215.0)],
                    [  0,   0, calibrateColorCode(190.0)],
                    [calibrateColorCode(255.0), calibrateColorCode(255.0),   0], # yellow
                    [calibrateColorCode(200.0), calibrateColorCode(200.0),   0], # yellow
                    [calibrateColorCode(255.0), calibrateColorCode(100.0), calibrateColorCode(153.0)], # pink
                    [calibrateColorCode(249.0), calibrateColorCode(130.0), calibrateColorCode(217.0)], # pink
                    [calibrateColorCode(102.0), calibrateColorCode(255.0), calibrateColorCode(255.0)], # teal
                    
                    ], dtype=np.float)

def calculate_learning_rate(epoch,initial_learning_rate,total_epochs):
    sigma_k= initial_learning_rate * (exp(-epoch/total_epochs))
    return sigma_k

# def getNeighborhoodTopology(winningWeight,inputColor,sigma_k):
#     distance = np.linalg.norm(inputColor-winningWeight)
#     N_c= exp(- (distance ** 2)/(2* (sigma_k ** 2)))
#     return N_c

def initialize_weights():
    np.random.seed(25)
    neuron_map = np.random.uniform(0, 1, (3,100,100))
    return neuron_map
    # print(weights)

# def calculateEuclideanDistance(weight,inputColor):
#     euclidean_distances=np.empty([100,100])
#     for row in range(100):
#         for col in range(100):
#             # print("weight[row][col]=",weight[row][col])
#             # print("inputColor=",inputColor)
#             euclidean_distances[row][col]=(np.linalg.norm(inputColor-weight[row][col]))

#     # temp = inputColor-weight
#     # euclidean_distances = np.linalg.norm(temp,2,axis=0)
#     return euclidean_distances

def calculateEuclideanDistance(weight,inputColor):
    difference=inputColor-weight
    euclidean_distances = np.linalg.norm(difference,2,axis=0)
    return euclidean_distances

def findMinimumEuclideanDistance(euclidean_distances):
    minDistance=euclidean_distances.min()
    return minDistance

def findWinningWeightIndex(euclidean_distances,minDistance):
    index=np.where(euclidean_distances==minDistance)
    return index

def getNeighborhoodTopology(winningWeightIndex,learning_rate):
    winning_index_i=winningWeightIndex[0][0]
    winning_index_j=winningWeightIndex[1][0]
    neighborhood_array = np.array((winning_index_i,winning_index_j)).reshape(2,1,1)
    map_index_array=np.indices((100,100))
    N_c= np.exp((-calculateEuclideanDistance(neighborhood_array,map_index_array)**2)/(2* (learning_rate ** 2)))
    return N_c

def getNeighborhoodRadius(color_input,weights):
    radius = color_input - weights
    return radius

def ksom(total_epochs,initial_learning_rate):


    weights_map=initialize_weights()

    # print("Init map:",weights_map)

    # plt.imshow(weights_map, extent=[0, 16000, 0, 1], aspect='auto')
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

        
        
    plt.imshow(weights_map.T, aspect='auto')
    plt.show()
    plt.close()

    

    for epoch in range(total_epochs):
        # print("epoch =",epoch)
        neighborhood_radius=np.empty([3,100,100])
        euclidean_distances=np.empty([100,100])
        for i in range(len(colour_inputs)):
            #Find winning neuron
            # euclidean_distances=calculateEuclideanDistance(weights_map,colour_inputs[i])
            euclidean_distances=calculateEuclideanDistance(weights_map,colour_inputs[i].reshape((3,1,1)))
            # print("colour_inputs[i]= ",colour_inputs[i])
            # print(euclidean_distances)
            minDistance=findMinimumEuclideanDistance(euclidean_distances)
            winningWeightIndex=findWinningWeightIndex(euclidean_distances,minDistance)
            

            # print("MinDistance=",minDistance)
            # print("index=",winningWeightIndex)
            
            # print(weights_map.shape)
            # print("HERE:",winningWeightIndex[0])
            # print(winningWeightIndex[1])

            # print("HERE: 1",weights_map[98][42][0])
            # print("HERE: 2",weights_map[index_i][index_j][0])
            # print(weights_map[winningWeightIndex[0]][winningWeightIndex[1]][0])
            # winningWeight=weights_map[index_i][index_j]


            learning_rate=calculate_learning_rate(epoch,initial_learning_rate,total_epochs)
            N_c = getNeighborhoodTopology(winningWeightIndex,learning_rate)
            neighborhood_radius = getNeighborhoodRadius(colour_inputs[i].reshape(3,1,1),weights_map)
            weight_update= (N_c * learning_rate) * neighborhood_radius
            weights_map = weights_map + weight_update


           
        # print("MinDistance=",minDistance)
        # print("index=",winningWeightIndex)
        # print("learning_rate=",learning_rate)
        # print("neighborhood_radius=",neighborhood_radius)
        # print("N_c=",N_c)
        # print("updated map:",weights_map)

        if(epoch in [19,39,99,999]):
            # plt.imshow((weights_map.T * 255).astype('uint8'))
            plt.imshow(weights_map.T, aspect='auto')
            plt.show()
            plt.close()

def main():  

    initial_learning_rate= 10

    total_epochs = 1000
    ksom(total_epochs,initial_learning_rate)
    # indices = np.random.randint(0, len(colours), size=(100, 100))


    # colour_map= np.array(colours[indices], dtype=float)

    # print(colour_map[1][1])
    # print(colours)
    # img = 
    # plt.imshow(colour_map, extent=[0, 16000, 0, 1], aspect='auto')
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()
    # sys.exit(0)

if __name__ == '__main__':
    main()