# -*- coding: utf-8 -*-
'''
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
'''

import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
## Permuted MNIST

# Generate the tasks specifications as a list of random permutations of the input pixels.
task_permutation = []
for task in range(num_tasks_to_run):
	task_permutation.append( np.random.permutation(784) )
  
num_tasks_to_run = 10

num_epochs_per_task = 20

learning_rate = 0.001
batch_size = 100
image_sz = 28
size_input = image_sz * image_sz
size_hidden = 128
size_output = 10

#Based on tutorial provided create your MLP model for above problem
#For TF2.0 users Keras can be used for loading trainable variables and dataset.
#You might need google collab to run large scale experiments

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = tf.reshape(tf.cast(train_images, dtype=tf.float32),[-1, size_input])
train_labels = tf.cast(train_labels, dtype=tf.int64)
train_images /= 255.

test_images = tf.reshape(tf.cast(test_images, dtype=tf.float32),[-1, size_input])
test_labels = tf.cast(test_labels, dtype=tf.int64)
test_images /= 255.
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size,drop_remainder=True)


# Define class to build mlp model
class MLP(tf.Module):
  def __init__(self):
    self.device = 'gpu'
    self.W1 = tf.Variable(tf.random.truncated_normal([size_input, size_hidden], stddev=0.05))
    self.b1 = tf.Variable(tf.random.normal([1, size_hidden]))
    self.W2 = tf.Variable(tf.random.truncated_normal([size_hidden, size_output], stddev=0.05))
    self.b2 = tf.Variable(tf.random.normal([1, size_output]))
    self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    self._variables = [self.W1, self.W2, self.b1, self.b2]

  def forward(self, X):
    if self.device is not None:
      with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
        self.y = self.compute_output(X)
    else:
      self.y = self.compute_output(X)

    return self.y

  def loss(self, logits, y_true):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = y_true)
  
  def pred(self, logits, y_true):
    return tf.argmax(tf.nn.softmax(logits),1)

  @tf.function(input_signature=[tf.TensorSpec(shape=[batch_size, size_input], dtype=tf.float32), tf.TensorSpec(shape=[batch_size], dtype=tf.int64),tf.TensorSpec(shape=None,dtype=tf.float32)])
  def __call__(self, X_train, y_train, dropout):
    self.dropout = dropout
    
    with tf.GradientTape() as tape:
      logits = self.forward(X_train)
      current_loss = self.loss(logits, y_train)
      pred = self.pred(logits, y_train)
    grads = tape.gradient(current_loss, self._variables)
    self.optimizer.apply_gradients(zip(grads, self._variables))
    return logits, current_loss, pred

  def compute_output(self, X):
    # Compute values in hidden layer
    what = tf.matmul(X, self.W1) + self.b1
    hhat = tf.nn.relu(what)
    # Compute output
    output = tf.matmul(hhat, self.W2) + self.b2
    return output


# Initialize model using CPU
mlp_on_cpu = MLP()

time_start = time.time()
for epoch in range(num_epochs_per_task):
  train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000, seed=epoch*(2612)).batch(batch_size,drop_remainder=True)
  acc_avg = tf.metrics.Accuracy()
  loss_avg = tf.metrics.Mean()
  for inputs, outputs in train_ds:
    logits,loss,pred = mlp_on_cpu(inputs, outputs, 0.)
    loss_avg(loss)
    acc_avg(pred, outputs)
  print('Number of Epoch = {} - Average MSE:= {:.10f} - ACC:={:.4f}'.format(epoch + 1, loss_avg.result() / train_images.shape[0], acc_avg.result()))
time_taken = time.time() - time_start

print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
#For per epoch_time = Total_Time / Number_of_epochs
