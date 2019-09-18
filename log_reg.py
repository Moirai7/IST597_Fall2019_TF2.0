""" 
author:-aam35
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import utils
tf.executing_eagerly()
# Define paramaters for the model
learning_rate = 0.001
batch_size = 100
n_epochs = None
n_train = None
n_test = None

# Step 1: Read in data
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#Create dataset load function [Refer fashion mnist github page for util function]
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = train_images.batch(batch_size)
train_label = train_labels.batch(batch_size)

# create testing Dataset and batch it
test_data = test_images.batch(batch_size)
test_label = test_labels.batch(batch_size)

#############################
########## TO DO ############
#############################


# create one iterator and initialize it with different datasets
'''
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data
'''

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w, b = tf.Variable(initializer = tf.initializers.RandomUniform(0, 0.01), shape=None), None
#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
def model(batch_x):
    """
    We will define the learned variables, the weights and biases,
    within the method ``model()`` which also constructs the neural network.
    The variables named ``hn``, where ``n`` is an integer, hold the learned weight variables. 
    The variables named ``bn``, where ``n`` is an integer, hold the learned bias variables.
    """
 
    b1 = tf.get_variable("b1", [config.n_hidden1], initializer = tf.zeros_initializer())
    h1 = tf.get_variable("h1", [config.n_input, config.n_hidden1],
                         initializer = tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.add(tf.matmul(batch_x,h1),b1))
 
    b2 = tf.get_variable("b2", [config.n_hidden2], initializer = tf.zeros_initializer())
    h2 = tf.get_variable("h2", [config.n_hidden1, config.n_hidden2],
                         initializer = tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,h2),b2))
 
    b3 = tf.get_variable("b3", [config.n_class], initializer = tf.zeros_initializer())
    h3 = tf.get_variable("h3", [config.n_hidden2, config.n_class],
                         initializer = tf.contrib.layers.xavier_initializer())
 
    layer3 = tf.add(tf.matmul(layer2,h3),b3)
 
    return layer3
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def cal_loss(logits, actual):
    total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,labels = actual)
    avg_loss = tf.reduce_mean(total_loss)
    return avg_loss
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
def cal_acc(logits, label):
   preds = tf.nn.softmax(logits)
   correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
   accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
   return accuracy

#Step 8: train the model for n_epochs times
for i in range(n_epochs):
	total_loss = 0
	n_batches = 0
	#Optimize the loss function
	print("Train and Validation accuracy")
	################################
	###TO DO#####
	############
	
#Step 9: Get the Final test accuracy

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set 
images = test_data[0:9]

# Get the true classes for those images.
y = test_class[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights 

def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = None
    #TO DO## obtains these value from W
    w_max = None

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

