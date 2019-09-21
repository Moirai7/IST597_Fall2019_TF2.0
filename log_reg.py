""" 
author:Lan Zhang
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

assert tf.executing_eagerly()
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Define paramaters for the model
learning_rate = 0.001
batch_size = 1000
buffer_size = 10000
n_epochs = 1000
dropout_prob = False
n_train = None
n_test = None

# Step 1: Read in data
#fmnist_folder = 'None'
#Create dataset load function [Refer fashion mnist github page for util function]
#Create train,validation,test split
#train, val, test = utils.read_fmnist(fmnist_folder, flatten=True)
train_data, test_data = tf.keras.datasets.fashion_mnist.load_data()

# Step 2: Create datasets and iterator
# create training Dataset and batch it
# create testing Dataset and batch it
train_images, train_labels = train_data
train_images = tf.expand_dims(tf.cast(train_images, dtype=tf.float32),-1)
train_labels = tf.cast(train_labels, dtype=tf.int64)
train_labels = tf.reshape(train_labels, shape=[train_labels.shape[0],1])
train_images /= 255.

test_images, test_labels = test_data
test_images = tf.expand_dims(tf.cast(test_images, dtype=tf.float32),-1)
test_images /= 255.
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

# create one iterator and initialize it with different datasets
features, label = iter(train_dataset).next()

#img_test, label_test = next(iter(test_dataset))
# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
#w, b = tf.Variable(initializer = tf.initializers.RandomUniform(0, 0.01), shape=None), None
#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer

##vgg16 
#vgg16 = {"conv1_1":[3,3,1,64], "conv1_2":[3,3,64,64], "pool1":[1,2,2,1],"conv2_1":[3,3,64,128],"conv2_2":[3,3,128,128],"pool2":[1,2,2,1],"conv3_1":[3,3,128,256],"conv3_2":[3,3,256,256],"conv3_3":[3,3,256,256],"pool3":[1,2,2,1], "conv4_1":[3,3,256,512],"conv4_2":[3,3,512,512],"conv4_3":[3,3,512,512],"pool4":[1,2,2,1],"conv5_1":[3,3,512,512],"conv5_2":[3,3,512,512],"conv5_3":[3,3,512,512],"pool5":[1,2,2,1],"fc6":[512],"fc7":[10],"fc8":[10]}
vgg16 = {"conv1_1":[3,3,1,32], "conv1_2":[3,3,32,32], "pool1":[1,2,2,1],"conv2_1":[3,3,32,16],"conv2_2":[3,3,16,16],"pool2":[1,2,2,1],"conv3_1":[3,3,16,8],"conv3_2":[3,3,8,8],"conv3_3":[3,3,8,8],"pool3":[1,2,2,1], "conv4_1":[3,3,8,4],"conv4_2":[3,3,4,4],"conv4_3":[3,3,4,4],"pool4":[1,2,2,1],"conv5_1":[3,3,4,4],"conv5_2":[3,3,4,4],"conv5_3":[3,3,4,4],"pool5":[1,2,2,1],"fc6":[4],"fc7":[10],"fc8":[10]}

class Vgg16(tf.Module):        
  def __init__(self):
    super(Vgg16, self).__init__()
    self.trainable = {}
    for name in vgg16.keys():
      if "conv" in name:
        self.trainable[name]=[]
        self.trainable[name].append(tf.Variable(trainable=True,initial_value=tf.random.truncated_normal(vgg16[name], dtype=tf.float32, stddev=0.05), name=name+"/filter"))
        self.trainable[name].append(tf.Variable(initial_value=tf.constant(0., shape=vgg16[name][-1], dtype=tf.float32), trainable=True, name=name+"/biases"))
      if "fc" in name:
        self.trainable[name]=[]
        self.trainable[name].append(tf.Variable(trainable=True,initial_value=tf.random.truncated_normal([vgg16[name][0],10], dtype=tf.float32, stddev=0.05), name=name+"/weights"))
        self.trainable[name].append(tf.Variable(initial_value=tf.constant(0., shape=[10], dtype=tf.float32), trainable=True, name=name+"/biases"))

  def conv_layer(self,bottom, name):
    with tf.name_scope(name):
        filters = self.trainable[name][0]
        conv = tf.nn.conv2d(bottom, filters, [1, 1, 1, 1], padding='SAME')
        conv_biases = self.trainable[name][1]
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)
        return relu

  def max_pool(self,bottom, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(bottom, ksize=vgg16[name], strides=vgg16[name], padding='SAME', name=name)

  def fc_layer(self,bottom, name):
    with tf.name_scope(name):
        shape = bottom.get_shape().as_list()
        k = 1
        for d in shape[1:]:
          k *= d
        x = tf.reshape(bottom, [-1, k])
        weight = self.trainable[name][0]
        bias = self.trainable[name][1]
        return tf.nn.bias_add(tf.matmul(x, weight), bias)

  def __call__(self, batch_x):
    conv1_1 = self.conv_layer(batch_x, "conv1_1")
    conv1_2 = self.conv_layer(conv1_1, "conv1_2")
    pool1 = self.max_pool(conv1_2, "pool1")
    
    conv2_1 = self.conv_layer(pool1, "conv2_1")
    conv2_2 = self.conv_layer(conv2_1, "conv2_2")
    pool2 = self.max_pool(conv2_2, "pool2")

    conv3_1 = self.conv_layer(pool2, "conv3_1")
    conv3_2 = self.conv_layer(conv3_1, "conv3_2")
    conv3_3 = self.conv_layer(conv3_2, "conv3_3")
    pool3 = self.max_pool(conv3_3, "pool3")

    conv4_1 = self.conv_layer(pool3, "conv4_1")
    conv4_2 = self.conv_layer(conv4_1, "conv4_2")
    conv4_3 = self.conv_layer(conv4_2, "conv4_3")
    pool4 = self.max_pool(conv4_3, "pool4")

    conv5_1 = self.conv_layer(pool4, "conv5_1")
    conv5_2 = self.conv_layer(conv5_1, "conv5_2")
    conv5_3 = self.conv_layer(conv5_2, "conv5_3")
    pool5 = self.max_pool(conv5_3, "pool5")

    fc6 = self.fc_layer(pool5, "fc6")
    relu6 = tf.nn.relu(fc6)

    fc7 = self.fc_layer(relu6, "fc7")
    relu7 = tf.nn.relu(fc7)

    fc8 = self.fc_layer(relu7, "fc8")
    return fc8

class CNNs(tf.Module):
  def __init__(self):
        self.conv1 = tf.layers.Conv2D(32, 3,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.maxpool = tf.layers.MaxPooling2D((2, 2),
                                              strides=(2, 2),
                                              padding='same')
        self.conv2 = tf.layers.Conv2D(64, 3,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.Conv2D(128, 3,
                                      activation=tf.nn.relu)
        self.dense1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.dense2 = tf.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.layers.Dropout(0.5)
        self.dense3 = tf.layers.Dense(10)

  def __call__(self, batch_x, training=True):
        x = self.conv1(batch_x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = tf.layers.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training)
        x = self.dense2(x)
        x = self.dropout(x, training)
        x = self.dense3(x)
        return x


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def cal_loss(logits, actual):
    total_loss = tf.losses.sparse_softmax_cross_entropy(logits = logits,labels = actual)
    return total_loss


# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


# Step 7: calculate accuracy with test set
from sklearn.metrics import accuracy_score
def cal_acc(logits, label):
   preds = tf.argmax(tf.nn.softmax(logits),1)
   print(logits.shape)
   print(preds)
   correct_preds = tf.equal(preds, label)
   accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
   return accuracy_score(preds, label)
   return accuracy

#Step 8: train the model for n_epochs times
mymodel = CNNs()#Vgg16()
for i in range(n_epochs):
   total_loss = 0
   n_batches = 0
   for (_x, _y) in tfe.Iterator(train_dataset):
     with tf.GradientTape() as tape:
       logits = mymodel(_x)
       loss = cal_loss(logits, _y)
     print(loss,cal_acc(logits, _y))
     gradients = tape.gradient(loss, mymodel.variables)
     optimizer.apply_gradients(zip(gradients, mymodel.trainable_variables))

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

