import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
import time

NAME = f"Zero-vs-One-cnn-16x16-{int(time.time())}"

X = np.load("featureSet.npy") #Load in the numpy array of features
Y = np.load("labels.npy") #Load in the array of labels

X = X/255.0 #Normalizes data, many cases should use "keras.utils.normalize()"

model = Sequential() #Sets the model type
model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:])) #Sets the window size of the layer and defines input shape of the data
model.add(Activation("relu")) #Sets the activation function to use for the layer
model.add(MaxPooling2D(pool_size=(2, 2))) #Specifies poool size

model.add(Conv2D(64, (3, 3))) #Sets the window size of the layer
model.add(Activation("relu")) #Sets the activation function to use for the layer
model.add(MaxPooling2D(pool_size=(2, 2))) #Specifies poool size

model.add(Flatten()) #Flattens layer data into one dimensonal array

model.add(Dense(64)) #Creates new layer with 64 neurons
model.add(Activation("relu")) #Apply activation function to layer

model.add(Dense(1)) #Creates the output layer
model.add(Activation("sigmoid")) #An activation fucntion to be applied to the output layer, removes raw data

model.compile( #Compiles layers into finished neural network
    loss="binary_crossentropy", #Used when it is true/false, one and zero sort of prediciton, else you would probably use categorical
    optimizer="adam", 
    metrics=["accuracy"])

model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.1) #Train the neural network

model.save("Zero_Vs_One") #Saves the neural network