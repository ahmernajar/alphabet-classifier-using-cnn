#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: ahmernajar
"""
#Importing the Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense ,BatchNormalization,Dropout
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam

#Building the CNN
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(64, 5, 5, input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(Convolution2D(64, 5, 5, activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.4))


# Adding a second convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(BatchNormalization())
#classifier.add(Dropout(0.4))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
#classifier.add(Dropout(0.4))


classifier.add(Dense(output_dim = 26, activation = 'softmax'))
#classifier.add(Dropout(0.4))


# Compiling the CNN

classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Image Data generator
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_test)

#OneHot Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()


classifier.fit_generator(datagen.flow(X_train, y_train , batch_size=86),
                         steps_per_epoch= 30,validation_data = (X_test,y_test),
                         epochs=30)



