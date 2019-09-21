""" 
author:Lan Zhang
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
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
train_images /= 255.

test_images, test_labels = test_data
test_images = tf.expand_dims(tf.cast(test_images, dtype=tf.float32),-1)
test_labels = tf.cast(test_labels, dtype=tf.int64)
test_images /= 255.
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

# create one iterator and initialize it with different datasets
#features, label = iter(train_dataset).next()

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
vgg16 = {"conv1_1":[3,3,1,32], "conv1_2":[3,3,32,32], "pool1":[1,2,2,1],"conv2_1":[3,3,32,64],"conv2_2":[3,3,64,64],"pool2":[1,2,2,1],"conv3_1":[3,3,64,128],"conv3_2":[3,3,128,128],"conv3_3":[3,3,128,128],"pool3":[1,2,2,1], "conv4_1":[3,3,128,256],"conv4_2":[3,3,256,256],"conv4_3":[3,3,256,256],"pool4":[1,2,2,1],"conv5_1":[3,3,256,256],"conv5_2":[3,3,256,256],"conv5_3":[3,3,256,256],"pool5":[1,2,2,1],"fc6":[256,128],"fc7":[128,64],"fc8":[64,10]}

class Vgg16(tf.Module):        
  def __init__(self):
    super(Vgg16, self).__init__()
    ##tensorflow 2.0 discourage name-based variable. use python objects to track variables
    ##fc output size + conv+pool+relu or conv+relu+pool
    self.trainable = {}
    for name in vgg16.keys():
      if "conv" in name:
        self.trainable[name]=[]
        self.trainable[name].append(tf.Variable(trainable=True,initial_value=tf.random.truncated_normal(vgg16[name], dtype=tf.float32, stddev=0.05), name=name+"/filter"))
        self.trainable[name].append(tf.Variable(initial_value=tf.constant(0., shape=vgg16[name][-1], dtype=tf.float32), trainable=True, name=name+"/biases"))
      if "fc" in name:
        self.trainable[name]=[]
        self.trainable[name].append(tf.Variable(trainable=True,initial_value=tf.random.truncated_normal(vgg16[name], dtype=tf.float32, stddev=0.05), name=name+"/weights"))
        self.trainable[name].append(tf.Variable(initial_value=tf.constant(0.05, shape=vgg16[name][-1], dtype=tf.float32), trainable=True, name=name+"/biases"))

  def conv_layer(self,bottom, name):
    with tf.name_scope(name):
        filters = self.trainable[name][0]
        conv = tf.nn.conv2d(bottom, filters, [1, 1, 1, 1], padding='SAME')
        conv_biases = self.trainable[name][1]
        bias = tf.nn.bias_add(conv, conv_biases)
        return bias

  def max_pool(self,bottom, name):
    with tf.name_scope(name):
        layer = tf.nn.max_pool2d(bottom, ksize=vgg16[name], strides=vgg16[name], padding='SAME', name=name)
        relu = tf.nn.relu(layer)
        return relu

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

  @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size, 28, 28, 1], dtype=tf.float32)])
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
    super(CNNs, self).__init__()
    conv1_filter_size = 3
    conv2_filter_size = 3
    conv3_filter_size = 3
    fc_layer_size = 128
    num_filters1 = 32
    num_filters2 = 64
    num_filters3 = 128
    features = 2048
    self.weight1 = tf.Variable(tf.random.truncated_normal([conv1_filter_size,conv1_filter_size,1,num_filters1], stddev=0.05), trainable=True)
    self.biases1 = tf.Variable(tf.constant(.05, shape=[num_filters1]), trainable=True)
    self.beta1 = tf.Variable(tf.constant(0.0, shape=[num_filters1]),name='beta', trainable=True)
    self.gamma1 = tf.Variable(tf.constant(1.0, shape=[num_filters1]),name='gamma', trainable=True)
    self.weight2 = tf.Variable(tf.random.truncated_normal([conv2_filter_size,conv2_filter_size,num_filters1,num_filters2], stddev=0.05), trainable=True)
    self.biases2 = tf.Variable(tf.constant(.05, shape=[num_filters2]), trainable=True)
    self.beta2 = tf.Variable(tf.constant(0.0, shape=[num_filters2]),name='beta', trainable=True)
    self.gamma2 = tf.Variable(tf.constant(1.0, shape=[num_filters2]),name='gamma', trainable=True)
    self.weight3 = tf.Variable(tf.random.truncated_normal([conv3_filter_size,conv3_filter_size,num_filters2,num_filters3], stddev=0.05), trainable=True)
    self.biases3 = tf.Variable(tf.constant(.05, shape=[num_filters3]), trainable=True)
    self.beta3 = tf.Variable(tf.constant(0.0, shape=[num_filters3]),name='beta', trainable=True)
    self.gamma3 = tf.Variable(tf.constant(1.0, shape=[num_filters3]),name='gamma', trainable=True)
    self.weight4 = tf.Variable(tf.random.truncated_normal([features,fc_layer_size]), trainable=True)
    self.biases4 = tf.Variable(tf.constant(.05, shape=[fc_layer_size]), trainable=True)
    self.weight5 = tf.Variable(tf.random.truncated_normal([fc_layer_size,10]), trainable=True)
    self.biases5 = tf.Variable(tf.constant(.05, shape=[10]), trainable=True)

  def __call__(self, batch_x):
    layer = self.conv_layer(batch_x, self.weight1, self.biases1,self.beta1, self.gamma1)

    layer = self.conv_layer(layer, self.weight2, self.biases2,self.beta2, self.gamma2)

    layer = self.conv_layer(layer, self.weight3, self.biases3,self.beta3, self.gamma3)

    layer_flat = self.create_flatten_layer(layer)

    layer = tf.matmul(layer_flat, self.weight4) + self.biases4
    layer = tf.nn.relu(layer)

    layer = tf.matmul(layer, self.weight5) + self.biases5
    return layer

  def create_flatten_layer(self, layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

  def conv_layer(self,batch_x,weight,biases,beta,gamma):
    layer = tf.nn.conv2d(batch_x, weight, strides=[1,1,1,1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool2d(layer,[1,2,2,1],[1,2,2,1],padding='SAME')
    batch_mean2, batch_var2 = tf.nn.moments(layer,[0])
    layer = tf.nn.batch_normalization(layer,batch_mean2, batch_var2, beta, gamma, 1e-3)
    layer = tf.nn.relu(layer)
    if dropout_prob:
       layer = tf.nn.dropout(layer,dropout_prob)
    return layer



# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
def cal_loss(logits, actual):
    total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = actual)
    return total_loss


# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)


# Step 7: calculate accuracy with test set
from sklearn.metrics import accuracy_score
def cal_acc(logits, label):
   preds = tf.argmax(tf.nn.softmax(logits),1)
   return accuracy_score(preds, label)


def test(mymodel):
   _accs = 0
   for ids,(x,y) in test_dataset.enumerate():
      logits = mymodel(x)
      _accs+= cal_acc(logits, y)*batch_size
   return _accs/buffer_size

#Step 8: train the model for n_epochs times
def train():
  train_loss_results = []
  train_accuracy_results = []
  mymodel = Vgg16()
  best = 0.0
  for i in range(n_epochs):
   total_loss = 0
   n_batches = 0
   acc = tf.metrics.Accuracy()
   loss_avg = tf.metrics.Mean()
   for idx, (_x, _y) in train_dataset.enumerate():
     with tf.GradientTape() as tape:
       logits = mymodel(_x)
       loss = cal_loss(logits, _y)
     print(idx,tf.reduce_mean(loss),cal_acc(logits, _y))
     gradients = tape.gradient(loss, mymodel.trainable_variables)
     optimizer.apply_gradients(zip(gradients, mymodel.trainable_variables))
     loss_avg(loss)
     acc(tf.argmax(tf.nn.softmax(logits),1), _y)
     ##test
     if (idx % 9 == 0):
        a = test(mymodel)
        if (a>=best):
           tf.saved_model.save(mymodel, 'model/vgg16/')
           best = a
        print(a)

   train_loss_results.append(loss_avg.result())
   train_accuracy_results.append(acc.result())
   print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(i,loss_avg.result(),acc.result()))

#Step 9: Get the Final test accuracy
def test_all():
   mymodel = tf.saved_model.load('model/vgg16/')
   a = test(mymodel)
   print(a)

test_all()

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

