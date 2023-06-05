import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
import cv2
import matplotlib as plt
import numpy as np

IMAGE_SIZE = (16, 16)
CATEGORIES = ["Zero", "One"]
NEW_DATA_PATH = os.path.join(os.getcwd(), "new data") #Gets the file path of the new data folder
MODEL_PATH = os.path.join(os.getcwd(), "Zero_Vs_One") #Gets the neural network models file path
model = load_model(MODEL_PATH) #Loads the neural network model

print("\n---------------------\nClass Identification\n---------------------")
for image in os.listdir(NEW_DATA_PATH): #Loops through all images in folder
    try:
        imageArray = cv2.imread(os.path.join(NEW_DATA_PATH, image), cv2.IMREAD_GRAYSCALE) #Loads each image in gray scale to variable imageArray
        imageArray = cv2.resize(imageArray, IMAGE_SIZE) #Resize image

        # plt.imshow(imageArray, cmap="gray") #Plots image
        # plt.show() #Shows image plot

        newData = []
        newData.append(imageArray)
        newData = np.array(newData).reshape(-1, IMAGE_SIZE.__getitem__(0), IMAGE_SIZE.__getitem__(1), 1) #Loads each image in gray scale to variable imageArray

        prediction = model.predict(newData) #Makes a prediction using the neural network model

        if prediction[0] >= 0.5: prediction[0] = 1 #Turns porbability based prediction into binary numbersfor the two categorical options of either "Zero" or "One"
        else: prediction[0] = 0

        print(f"The image [{image}] at \"{os.path.join(NEW_DATA_PATH, image)}\" was classified as a {CATEGORIES[int(prediction[0])]}")
    except Exception as e:
        print(f"Could not classify image {os.path.join(NEW_DATA_PATH, image)}.\nError: {e}\n")