# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 19:00:42 2020

methodology adapted from Jason Brownlee
https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/


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
from tensorflow.keras.optimizers import Adam
from numpy.random import randint


import matplotlib.pyplot as plt
#from ann_visualizer.visualize import ann_viz
import graphviz
import networkx as nx


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


#Define Stand Alone Discriminator  #Possible issue: not returning model into gal env (12/26/2020)
def define_generator(latent_dim):
    
    model = K.models.Sequential([K.layers.Input(shape=latent_dim),
                             K.layers.Dense(latent_dim, activation = 'relu')]) #input layer (dense, latent_dim nodes)
    
    model.add(Dense(45, activation='sigmoid')) #1st hidden layer (dense, 45 nodes)
    model.add(Dropout(0.2, input_shape=(95,))) #2nd hidden layer (dropout, 45 nodes)
    model.add(Dense(95, activation='relu'))    #output layer     (dense, 95 nodes ie output dim)

    model.summary()

    return model
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


#Load real examples
def load_real_samples():
    """assumes local file path. Returns df with x_train"""
    kdd_train=pd.read_csv("C:/Users/mchale/OneDrive/AFIT/Summer 2020/CSCE 823 Advanced ML/Project/Datasets/kdd_train_GANS.csv", header = None)
    #kdd_test=pd.read_csv("C:/Users/mchale/OneDrive/AFIT/Summer 2020/CSCE 823 Advanced ML/Project/Datasets/kdd_test_GANS.csv", header = None)

    #x_cols=kdd_test.shape[1]

    #y_train=kdd_train[0]
    #y_test=kdd_test[0]
    x_train=kdd_train.iloc[:,1:(kdd_train.shape[1]+1)] #extract the x set, drop the y set
    #x_test=kdd_test.iloc[:,1:(x_cols+1)]
    x_cols=x_train.shape[1] #record number of columns x, use later for ANN input size

    return x_train


#n_samples=int(np.floor(x_train.shape[0]))

# Select subsets of real samples
def generate_real_samples(dataset, n_samples):
    #chose random instances
    ix = randint(0, int(np.floor(dataset.shape[0])), n_samples) #produces some repeats
	# generate class labels
    X = dataset.iloc[ix]

    #label the real examples with 1
    y = np.ones((n_samples, 1))   #label= indicating these are REAL data

    return X, y


# generate points in latent space as input for the generator
#how do I select size of latent space? same as columns in x_train?
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


#Input random pts from latent space (plus labels) into generator
#generate n fake examples with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)#we are calling a model 'generator' to predict
	# label fake eamples with 0
	y = np.zeros((n_samples, 1)) 
	return X, y



#Train the generator and discriminator

def train(g_model, d_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epoch= int(dataset.shape[0]/n_batch)#internally calc num rows/epochs at correct pace
    half_batch= int(n_batch/2) #some examples of each type used per train epoch
    #manually enumerate epochs
    for i in range(n_epochs):
        #enumerate the batches over the training set
        for j in range(bat_per_epoch): #bat per epoch is the totoal num examples each epoch 
            X_real, y_real =generate_real_samples(dataset, half_batch) #recall that y real = 1
            #update discriminator model weights
            d_loss_1, _ = d_model.train_on_batch(X_real, y_real)
            #generate fake examples with our user defined function
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            #update discriminator model weights
            d_loss_2, _ = d_model.train_on_batch(X_fake, y_fake)
            #generate random pts in latent space
            X_gan= generate_latent_points(latent_dim, n_batch)
            #assign inverted labels for fake samples
            y_gan= np.ones((n_batch,1))
            #update generator weights with error from discriminator predictions
            
            #should this use model "g_model" instead of fan_model(corrected)
            g_loss = g_model.train_on_batch(X_gan, y_gan)
            #print out summary of loss on current batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epoch, d_loss_1, d_loss_2, g_loss))
           	# save the generator model for fakw data cretion!
            g_model.save('generator.h5')
            
            
#Enabling definitions
# size of the latent space
latent_dim = 45
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()
# train model
train(generator, discriminator, gan_model, dataset, latent_dim)




