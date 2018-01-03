#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolutional Neural Network :- For Image recognition and classification of 
    classes of images from the dataset and to decide that the image is of 
    which class !!
            Here we are using example of example of dog and cat classes!!!!
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from timeit import default_timer as timer
start = timer()

# Initialising the CNN
classifier = Sequential()

#Convolution
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#For better performance more deep the network is better the efficiency !!!
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Flattening
classifier.add(Flatten())
# Full connection
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the CNN to the images
"""For this part Refer keras documentation and in that image preprocessing 
    I have imported this code from there for image preprocessing and fitting CNN
    to the dataset !!!!!!
"""

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 63)
# elapsed time
end = timer()
print(end - start)
import os
os.system('say "your program has finished"')


