"""
author:Lan Zhang
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

tf.enable_eager_execution()

assert tf.executing_eagerly()

learning_rate = 0.001
batch_size = 1000
buffer_size = 10000
n_epochs = 1000
dropout_prob = False

train_data, test_data = tf.keras.datasets.fashion_mnist.load_data()
train_images, train_labels = train_data
train_images = tf.expand_dims(tf.cast(train_images, dtype=tf.float32),-1)
train_labels = tf.cast(train_labels, dtype=tf.int32)
#train_labels = tf.reshape(train_labels, shape=[train_labels.shape[0],1])
train_images /= 255.

test_images, test_labels = test_data
test_images = tf.expand_dims(tf.cast(test_images, dtype=tf.float32),-1)
test_images /= 255.

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)

features, label = iter(train_dataset).next()
print("Example feature:", features[0])
print("Example label:", label[0])

class FMNISTModel(tf.keras.Model):
    
    def __init__(self):
        super(FMNISTModel, self).__init__()
        self._input_shape = [-1, 28, 28, 1]
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
        
    def __call__(self, inputs, training=False):
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x)
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

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets, training):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=training)
    return tape.gradient(loss_value, model.variables)

model = FMNISTModel()

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

from tqdm import tqdm

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 4

for epoch in tqdm(range(num_epochs)):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in tfe.Iterator(train_dataset):
        # Optimize the model
        grads = grad(model, x, y, training=True)

        optimizer.apply_gradients(zip(grads, model.variables),
                                 global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y, training=True))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

