import numpy as np
import pandas as pd
import pickle as pkl
import MLPTake3 as MLP

STUDENT_NAME = 'HAMMAD AHMED REHMAN'
STUDENT_ID = '20293340'

def test_mlp(data_file):
    # Load the test set
    # START
    data_file = np.genfromtxt(data_file, delimiter=",")
    # END


    # Load your network
    # START
    layers_neurons = [45, 4]
    network = MLP.MLP(layers_neurons)
    file = open("params.pkl", "rb")
    network.params = pkl.load(file)

    # END


    # Predict test set - one-hot encoded

    y_pred = network.predict(data_file)
    return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100

HR - Report on Q4 in scanned pdf
'''

