import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import re
import pickle




class MLP:

    def __init__(self, sizes):
        self.sizes = sizes #contains layer info
        self.train_X, self.train_y, self.test_X, self.test_y = self.loadData()
        self.train_X, self.test_X = self.normalise(self.train_X, self.test_X)
        self.setParams()


    def loadData(self):
        (train_X, train_y), (test_X, test_y) = keras.datasets.cifar10.load_data()
        train_y = keras.utils.to_categorical(train_y)
        test_y = keras.utils.to_categorical(test_y)
        indices = np.random.choice(np.arange(train_X.shape[0]), 10000, replace=False) #for random 20% selection
        train_X = train_X[indices]
        train_y = train_y[indices]
        return train_X, train_y, test_X, test_y

    def normalise(self, train, test):
        train = train.astype("float32")/255.0
        test = test.astype("float32")/255.0
        return train, test

    def setParams(self):
        model = keras.Sequential()
        for i in range(len(self.sizes)):
            if i == 0: #first layer
                model.add(keras.layers.Flatten(input_shape = (32,32,3))) #flatten for dense layers
                model.add(keras.layers.Dense(self.sizes[i], activation="sigmoid"))

            elif i == (len(self.sizes)-1):
                model.add(keras.layers.Dense(self.sizes[i], activation="softmax")) #final layer
            else:
                model.add(keras.layers.Dense(self.sizes[i], activation="sigmoid")) #intermediate layers
        opt = keras.optimizers.Adam(learning_rate=0.00015) #low learning rate required
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"]) #loss and metrics ideal for multi category prediction
        self.model = model

    def train(self):
        fitted = self.model.fit(self.train_X, self.train_y, epochs=5, batch_size=32, validation_data=(self.test_X, self.test_y)) #fit and store
        self.plot(fitted)


    def plot(self,history): #to plot graphs used in report
        plt.plot(history.history["categorical_accuracy"])
        plt.plot(history.history["val_categorical_accuracy"])
        plt.title("Testing v Training Accuracy {}".format(self.sizes))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["training", "validation"], loc = "upper left")
        plt.show()





class CNN:
    def __init__(self, question):
        self.train_X, self.train_y, self.test_X, self.test_y = self.loadData()
        self.train_X, self.test_X = self.normalise(self.train_X, self.test_X)
        self.setParams(question)
        self.question = question



    def loadData(self):
        (train_X, train_y), (test_X, test_y) = keras.datasets.cifar10.load_data()
        train_y = keras.utils.to_categorical(train_y)
        test_y = keras.utils.to_categorical(test_y)
        indices = np.random.choice(np.arange(train_X.shape[0]), 10000, replace=False)
        train_X = train_X[indices]
        train_y = train_y[indices]
        return train_X, train_y, test_X, test_y

    def normalise(self, train, test):
        train = train.astype("float32")/255.0
        test = test.astype("float32")/255.0
        return train, test

    def setParams(self, question):
        model = keras.Sequential()
        if question == 1: #first CNN model
            model.add(keras.layers.Conv2D(64,(3,3),input_shape=(32,32,3), activation="relu")) #for first layer
            model.add(keras.layers.Conv2D(64,(3,3),activation="relu"))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512, activation="sigmoid"))
            model.add(keras.layers.Dense(512, activation="sigmoid"))
            model.add(keras.layers.Dense(10, activation="softmax"))
            opt = keras.optimizers.Adam(learning_rate=0.00015)
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
            self.model = model
        elif question == 2: #for second model
            model.add(keras.layers.Conv2D(64,(3,3), input_shape=(32,32,3), activation="relu"))
            model.add(keras.layers.MaxPooling2D(2,2))
            model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
            model.add(keras.layers.MaxPooling2D(2, 2))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(512, activation="sigmoid"))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(512, activation="sigmoid"))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(10, activation="softmax"))
            opt = keras.optimizers.Adam(learning_rate=0.00015)
            model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
            self.model = model

    def train(self):
        fitted = self.model.fit(self.train_X, self.train_y, epochs=5, batch_size=32, validation_data=(self.test_X, self.test_y))
        self.plot(fitted)


    def plot(self, history):
        plt.plot(history.history["categorical_accuracy"])
        plt.plot(history.history["val_categorical_accuracy"])
        plt.title("Testing v Training Accuracy for CNN Q{}".format(self.question))
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["training", "validation"], loc="upper left")
        plt.show()

class RNN:
    def __init__(self, train=True):
        if train:
            self.train_X, self.train_y, self.scaler = self.loadTrainData("data/train_data_RNN.csv")
            self.setParams()
        else:
            self.test_X, self.test_y, self.scaler, self.test_ynorm = self.loadTestData("data/test_data_RNN.csv")
            self.loadModel("models/harehman_RNN_model.h5")
            self.test()


    def loadTrainData(self, trainCSV):
        train = pd.read_csv(trainCSV)
        scaler = MinMaxScaler()
        train =scaler.fit_transform(train.iloc[:,1:])

        train_X = train[:, :-1]

        train_y = train[:, -1]

        train_X = train_X.reshape(-1,3,4) #3D data for LSTM layer
        train_y = train_y.reshape(-1,1)

        return train_X, train_y, scaler

    def loadTestData(self, testCSV):
        test = pd.read_csv(testCSV)

        test_X = test.iloc[:,1:-1].to_numpy()

        test_y = test.iloc[:,-1].to_numpy()

        test_y = test_y.reshape(-1,1)
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler().fit(test_y)

        test_X = scaler_X.fit_transform(test_X)
        test_ynorm = scaler_y.transform(test_y)
        test_X = test_X.reshape(-1, 3, 4)

        return test_X, test_y, scaler_y, test_ynorm

    def loadModel(self, modelPath):
        self.model = keras.models.load_model(modelPath)

    def setParams(self):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(12,input_shape=(3,4), activation="sigmoid"))
        model.add(keras.layers.Dense(1, activation = "relu"))
        opt = keras.optimizers.Adam(learning_rate=0.00005)
        model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mean_squared_error"])
        self.model = model

    def train(self, plot=False):
        fitted = self.model.fit(self.train_X, self.train_y, validation_split=0.3, epochs=200, batch_size=10)
        if plot:
            self.plot(fitted)
        else:
            print("Final Mean Squared Error Loss: {}\n".format(fitted.history["mean_squared_error"][-1]))
        self.model.save("models/harehman_RNN_model.h5")

    def test(self):
        loss,_ = self.model.evaluate(self.test_X, self.test_ynorm)
        predicted = self.model.predict(self.test_X)
        predicted = self.scaler.inverse_transform(predicted)
        print("Mean Squared Error Loss is: {}\n".format(loss))
        plt.plot(predicted)
        plt.plot(self.test_y)
        plt.title("Predicted vs Actual")
        plt.ylabel("$")
        plt.xlabel("Datapoint")
        plt.legend(["predicted", "actual"], loc="upper right")
        plt.show()


    def plot(self, history):
        plt.plot(history.history["mean_squared_error"])
        plt.plot(history.history["val_mean_squared_error"])
        plt.title("Testing v Training")
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Epoch")
        plt.legend(["training", "validation"], loc="upper right")
        plt.show()

class NLP:
    def __init__(self, train): #maxlen is maximum length of each input to cut/pad to (for constant input length)
        maxlen = 200
        if train==True:
            self.train_X, self.train_y = self.loadData(True)
            self.tokenise(train,maxlen)
            self.maxlen = maxlen
            self.setParams()
            self.train(train)
        else:
            self.test_X, self.test_y = self.loadData(False)
            self.tokenise(train,maxlen)
            self.maxlen = maxlen
            self.model = keras.models.load_model("models/harehman_NLP_model.h5")
            self.test()


    def loadData(self, train): #load data for training
        if train==True:
            extension = "train"
        else:
            extension = "test"
        dir = "data/aclImdb/" + extension
        negdir = dir+"/neg/"
        posdir = dir+"/pos/"
        neg_data = pd.DataFrame(columns=["Text", "Positive", "Negative"])
        pos_data = pd.DataFrame(columns=["Text", "Positive", "Negative"])

        for file in os.listdir(negdir): # make DF for negative reviews
            if file.endswith(".txt"):
                neg_data = neg_data.append({"Text":self.preprocess(open(negdir+file, "r").read()), "Positive":0, "Negative":1}, ignore_index=True)

        for file in os.listdir(posdir): #make DF for postiive reviews
            if file.endswith(".txt"):
                pos_data = pos_data.append({"Text":self.preprocess(open(posdir+file, "r").read()), "Positive":1, "Negative":0},ignore_index=True)

        combined = pd.concat([neg_data, pos_data])
        combined = combined.sample(frac=1).reset_index(drop=True)
        X = combined.iloc[:,0].to_numpy()
        y = combined.iloc[:, 1:].to_numpy(dtype=np.float32)


        return X,y




    def setParams(self): #setup model
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocabulary, 210, input_length=self.maxlen)) #setup feature embedding
        model.add(keras.layers.LSTM(130, activation="sigmoid"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2, activation="softmax")) #2 outputs -> 1 hot encoded
        opt = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["acc"])
        self.model = model



    def preprocess(self, para): #check further processing
        para = re.sub("[^a-zA-z]", " ", para) #remvoe punctuation and numbers
        para = re.sub(r"\s+[a-zA-Z]\s+", " ", para) #remove single characters
        para = re.sub(r"\s+", " ", para) #remove extra spaces
        para = re.compile(r'<[^>]+>').sub("", para) #remove tags
        return para

    def tokenise(self, train=True, maxlen=200): #to convert each word to a sequence of ints
        tokens = keras.preprocessing.text.Tokenizer(num_words=10000)  #randomly chose 10k max words, should be sufficient for this case
        if train==True:
            tokens.fit_on_texts(self.train_X) #fit tokeniser
            self.train_X = tokens.texts_to_sequences(self.train_X) #change training data to tokenised d ata
            self.vocabulary = len(tokens.word_index)+1 #max vocab based off fitted token model
            self.train_X = keras.preprocessing.sequence.pad_sequences(self.train_X, maxlen=maxlen).astype("float32") #set constant length for each input
            with open("models/tokeniser.pickle", "wb") as handle: #needed for testing
                pickle.dump(tokens, handle)
        else:
            with open("models/tokeniser.pickle", "rb") as handle:
                tokens = pickle.load(handle)
            self.test_X = tokens.texts_to_sequences(self.test_X)
            self.vocabulary = len(tokens.word_index) + 1
            self.test_X = keras.preprocessing.sequence.pad_sequences(self.test_X, maxlen=maxlen).astype("float32")

    def train(self, plot=False): #train as above
        fitted = self.model.fit(self.train_X, self.train_y, validation_split=0.3, epochs=3, batch_size=20)
        if plot:
            self.plot(fitted)
        self.model.save("models/harehman_NLP_model.h5")

    def test(self):
        _, acc = self.model.evaluate(self.test_X, self.test_y)
        print("Accuracy for this NLP model is {}".format(acc*100))


    def plot(self, history):
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.title("Testing v Training")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["training", "validation"], loc="upper right")
        plt.show()









