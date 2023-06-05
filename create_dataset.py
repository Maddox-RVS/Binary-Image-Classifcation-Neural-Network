import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from random import shuffle

DATASET_DIRECTORY = os.getcwd() + "\\training data" #Gets working directory
CATEGORIES = ["Zero", "One"] # Establishes the two number categories
IMAGE_SIZE = (16, 16)

trainingData = []

def createTrainingData():
    for category in CATEGORIES: #Loops through each category folder in training data
        path = os.path.join(DATASET_DIRECTORY, category) #Joins the working directory with the folder name for "Zero" and "One"
        classNumber = CATEGORIES.index(category) #Gets the numerical value of the category

        for image in os.listdir(path): #Loops through all images in folder
            try:
                imageArray = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) #Loads each image in gray scale to variable imageArray
                imageArray = cv2.resize(imageArray, IMAGE_SIZE) #Resize image
                # plt.imshow(imageArray, cmap="gray") #Plots image
                # plt.show() #Shows image plot
                trainingData.append([imageArray, classNumber]) #Adds data to the training dataset
            except Exception as e:
                print(f"Could not append image {os.path.join(path, image)} to the dataset.\n\nError: {e}")

createTrainingData() #Creates training data
print(f"Training data length: {len(trainingData)}")

shuffle(trainingData) #Shuffles training data

featureSet_X = []
labels_Y = []

for features, labels in trainingData: #Populates the featureX and labelY dataset
    featureSet_X.append(features)
    labels_Y.append(labels)

featureSet_X = np.array(featureSet_X).reshape(-1, IMAGE_SIZE.__getitem__(0), IMAGE_SIZE.__getitem__(1), 1) #Convert to numpy array so that it can be passed to neural network
np.save("featureSet", featureSet_X) #Saves the numpy array
np.save("labels", labels_Y) #Saves the labels array