# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 01:42:01 2020

@author: udayk
"""
import pandas as pd
import numpy as np
import keras
import os
from keras import Sequential
import matplotlib as plt
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten
from keras.layers import Dense
from keras.applications.vgg16 import preprocess_input
model = VGG16()
model.summary()
model = VGG16(weights='imagenet',include_top=False,input_shape=[224,224,3])
from keras.models import Model
from keras.applications import preprocess
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


object=['buildings','mountain','glacier','forest','sea','street']
DIR='C:\\Users\\udayk\\Downloads\\intel-image-classification\\seg_train\\seg_train\\'
images=[]
label=[]
for x in object:
    nd=DIR+x+'\\'
    print(nd)
    
    for y in os.listdir(nd):
        print(y)
        image=nd+y
        images.append(image)
        label.append(x)
        
        
        
df=pd.DataFrame({'Path':images,'Category':label})
df['Category'].value_counts()


df = shuffle(df)
for x,y in zip(df['Path'][:5],df['Category'][:5]):
    img=mpimg.imread(x)
    print(y)
    imgplot = plt.imshow(img)
    plt.show()


from keras.preprocessing.image import ImageDataGenerator

WIDTH = 224
HEIGHT = 224
BATCH_SIZE = 32

    
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_generator=ImageDataGenerator(preprocessing_function=preprocess_input)

    
x = model.output    
flat1 = Flatten()(x)
class1 = Dense(1024, activation='relu')(flat1)
class1=Dropout(0.4)(class1)
output = Dense(6, activation='softmax')(class1)
model = Model(inputs=model.inputs, outputs=output)

df['Category']= encoder.fit_transform(df['Category'])


from sklearn.model_selection import train_test_split

dftrain,dfval = train_test_split(df, test_size = 0.2, random_state = 0)

train_generator = train_datagen.flow_from_dataframe(dftrain,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='categorical', x_col='Path', y_col='Category')

validation_generator = validation_generator.flow_from_dataframe(dfval,target_size=(HEIGHT, WIDTH),batch_size=BATCH_SIZE,class_mode='categorical', x_col='Path', y_col='Category')

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


EPOCHS = 5
BATCH_SIZE = 32
STEPS_PER_EPOCH = 320
VALIDATION_STEPS = 64

MODEL_FILE = 'filename.model'

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)