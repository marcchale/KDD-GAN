# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:00:42 2020

@author: mchale
"""

#%% Import modules
import pandas as pd
import numpy as np
import time
import os
import math
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam


import matplotlib.pyplot as plt
#from ann_visualizer.visualize import ann_viz
import graphviz
import networkx as nx

#%% Import Data
kdd_train=pd.read_csv("C:/Users/mchale/OneDrive/AFIT/Summer 2020/CSCE 823 Advanced ML/Project/Datasets/kdd_train_GANS.csv", header = None)
kdd_test=pd.read_csv("C:/Users/mchale/OneDrive/AFIT/Summer 2020/CSCE 823 Advanced ML/Project/Datasets/kdd_test_GANS.csv", header = None)

x_cols=kdd_train.shape[1]
x_cols=kdd_test.shape[1]

y_train=kdd_train[0]
y_test=kdd_test[0]
x_train=kdd_train.iloc[:,1:(x_cols+1)]
x_test=kdd_test.iloc[:,1:(x_cols+1)]

#RAM Management
del kdd_train, kdd_test

#%% Explore the variables
print('Dimensions:')
print('Train X:', x_train.shape, 'Train Y:',y_train.shape)
print('Test X:', x_test.shape,'Train X:', y_test.shape)


#%% Vanilla GAN

#Define Stand Alone Discriminator
def define_discriminator(in_shape=95):
    
    # Callbacks & hyperparameters
    LRdec= pow(10, -10)
    opt = K.optimizers.Nadam(learning_rate= 0.0001, decay= LRdec, beta_1=.6, beta_2=0.7, epsilon=1e-7)
    width = 95
#    batch_sz =16
#    num_epochs = 100
    act = "relu"
    #act = "relu"

    model = K.models.Sequential([K.layers.Input(shape=95),
                             K.layers.Dense(width, activation = act)])
    
    #Add dropout hidden layers and dense output    
    model.add(Dropout(0.2, input_shape=(95,)))
    model.add(Dropout(0.2, input_shape=(95,)))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    #dropout?
    # Compile & Train

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model #our function returns the GAN discriminator we just made
    #Cant perform validation testing? consider sequestering some, or otherwise use test
    #loss_history = GAN.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size = batch_sz, verbose=2, epochs=num_epochs,callbacks=[earlystopping_cb])


#Define Stand Alone Discriminator
def define_generator(latent_dim):
    
    model = K.models.Sequential([K.layers.Input(shape=latent_dim),
                             K.layers.Dense(latent_dim, activation = 'relu')]) #input layer (dense, latent_dim nodes)
    
    model.add(Dense(45, activation='sigmoid')) #1st hidden layer (dense, 45 nodes)
    model.add(Dropout(0.2, input_shape=(95,))) #2nd hidden layer (dropout, 45 nodes)
    model.add(Dense(95, activation='relu'))    #output layer     (dense, 95 nodes ie output dim)

    model.summary()

    #we intentionally withold compiling inside the function


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model








