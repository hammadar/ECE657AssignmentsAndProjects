# import required packages
import pandas as pd
import numpy as np
import keras
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import utils


def loadData(csv): #preparation of training and testing data
    df_temp = pd.read_csv(csv)
    df = pd.DataFrame(pd.np.empty((1259, 14)))
    df.columns = ["Date", "Volume -1", "Open -1", "High -1", "Low -1",
                  "Volume -2", "Open -2", "High -2", "Low -2",
                  "Volume -3", "Open -3", "High -3", "Low -3", "Target"]
    df["Date"] = df_temp["Date"]
    # df1 = df_temp.iloc["Close/Last", "Open", "High", "Low"]
    for i in range(1, len(df_temp) - 3):
        df.loc[df.index[i], "Volume -1"] = df_temp.loc[df_temp.index[i + 1], " Volume"]
        df.loc[df.index[i], "Open -1"] = df_temp.loc[df_temp.index[i + 1], " Open"]
        df.loc[df.index[i], "High -1"] = df_temp.loc[df_temp.index[i + 1], " High"]
        df.loc[df.index[i], "Low -1"] = df_temp.loc[df_temp.index[i + 1], " Low"]
        df.loc[df.index[i], "Volume -2"] = df_temp.loc[df_temp.index[i + 2], " Volume"]
        df.loc[df.index[i], "Open -2"] = df_temp.loc[df_temp.index[i + 2], " Open"]
        df.loc[df.index[i], "High -2"] = df_temp.loc[df_temp.index[i + 2], " High"]
        df.loc[df.index[i], "Low -2"] = df_temp.loc[df_temp.index[i + 2], " Low"]
        df.loc[df.index[i], "Volume -3"] = df_temp.loc[df_temp.index[i + 3], " Volume"]
        df.loc[df.index[i], "Open -3"] = df_temp.loc[df_temp.index[i + 3], " Open"]
        df.loc[df.index[i], "High -3"] = df_temp.loc[df_temp.index[i + 3], " High"]
        df.loc[df.index[i], "Low -3"] = df_temp.loc[df_temp.index[i + 3], " Low"]
        df.loc[df.index[i], "Target"] = df_temp.loc[df_temp.index[i], " Open"]

    df = df.iloc[1:-3, :]
    train, test = train_test_split(df, test_size=0.3, shuffle=True, random_state=10)
    return train, test

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__": 
	# 1. load your training data

    train = "data/train_data_RNN.csv"
    test = "data/test_data_RNN.csv"
    rnn = utils.RNN()
    rnn.train()

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss


	# 3. Save your model

