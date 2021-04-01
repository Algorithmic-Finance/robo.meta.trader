##### Deep Learning Tradebot
##### Chart-based Robo Trading / Version CNN 1
#####
##### (c)2016-2021 Ronald Hochreiter <ronald@algorithmic.finance>

### Packages

import glob 
import cv2 # opencv-python

import pandas as pd
import numpy as np

import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

### Parameter

path_img = "/path/img/"
file_labels = "/path/labels.csv"

input_shape = (200, 200, 3)
num_classes = 2
num_test = 200

### Process Images

# Get images from data_path
list_img = glob.glob(path_img + "*.png")

# Convert images into numpy arrays
imgs = np.array([cv2.imread(f) for f in filelist])

# Read Labels
labels = pd.read_csv(file_labels)

# Train test split and normalize RGB values
x_train = np.multiply(imgs[:num_test], 1/255) 
x_test  = np.multiply(imgs[num_test:], 1/255)

y_train = imgs[:num_test] 
y_test  = imgs[num_test:]

### Deep Learning

# Parameter
batch_size = 20
epochs = 500

# Network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
