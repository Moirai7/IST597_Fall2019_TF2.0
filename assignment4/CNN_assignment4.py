# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bu-bMxcsM2j0P2ttB0SpQ70389wwGlfC
"""

# -*- coding: utf-8 -*-
"""CNN_week9.ipynb

IST597 :- Implementing CNN from scratch
Week 9 Tutorial

Author:- aam35
"""
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import time
#import tensorflow.contrib.eager as tfe
import tensorflow_addons as tfa
#tf.enable_eager_execution()
#tf.executing_eagerly()
seed = 2612
tf.random.set_seed(seed=seed)
np.random.seed(seed)


#from tensorflow.examples.tutorials.mnist import input_data
import input_data
data = input_data.read_data_sets('data/fashion', one_hot=True, reshape=False)

batch_size = 100
h_size = 28
w_size = 28
hidden_size = 100
learning_rate = 0.01
output_size = 10
eps = 1e-5
channel = 30
filter_h, filter_w, filter_c , filter_n = 5 ,5 ,1 ,30
momentum = 0.9

num_epochs = 5
train_x =  tf.convert_to_tensor(data.train.images)
train_y = tf.convert_to_tensor(data.train.labels)
time_start = time.time()
num_train = 55000
z= 0

def accuracy_function(yhat,true_y):
  correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

class CNN(object):
  def __init__(self,hidden_size,output_size,device=None, batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False, compare_BN=False, compare_LN=False, compare_WN=False):
      self.batch_norm1 = batch_norm1
      self.batch_norm2 = batch_norm2
      self.layer_norm1 = layer_norm1
      self.layer_norm2 = layer_norm2
      self.weight_norm = weight_norm
      self.compare_BN = compare_BN
      self.compare_LN = compare_LN
      self.compare_WN = compare_WN
      #self.g1 = tf.Variable(tf.random_normal([filter_n], stddev=0.1)
      self.g1 = tf.Variable(tf.random.normal([1], stddev=0.1))
      self.v1 = tf.Variable(tf.random.normal([filter_h, filter_w, filter_c, filter_n], stddev=0.1))
      self.W1 = tf.Variable(tf.random.normal([filter_h, filter_w, filter_c, filter_n], stddev=0.1))
      self.b1 = tf.Variable(tf.zeros([filter_n]),dtype=tf.float32)
      self.W2 = tf.Variable(tf.random.normal([14*14*filter_n, hidden_size], stddev=0.1))
      self.b2 = tf.Variable(tf.zeros([hidden_size]),dtype=tf.float32)
      self.W3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
      self.b3 = tf.Variable(tf.zeros([output_size]),dtype=tf.float32)
      
      self.gamma1 = tf.Variable(tf.ones( shape=[1,1,1,filter_n]))
      self.beta1  = tf.Variable(tf.zeros([1,1,1,filter_n]))
      self.avg_mean1 = tf.Variable(tf.zeros([1,1,1,filter_n]),dtype=tf.float32)
      self.avg_var1 = tf.Variable(tf.zeros([1,1,1,filter_n]),dtype=tf.float32)
      self.gamma2 = tf.Variable(tf.ones( shape=[1,hidden_size]))
      self.beta2  = tf.Variable(tf.zeros([1,hidden_size]))
      self.avg_mean2 = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
      self.avg_var2 = tf.Variable(tf.zeros([1,hidden_size]),dtype=tf.float32)
      
      self.gamma3 = tf.Variable(tf.ones(shape=[batch_size,1,1,1]))
      self.beta3  = tf.Variable(tf.zeros(shape=[batch_size,1,1,1]))
      self.gamma4 = tf.Variable(tf.ones(shape=[hidden_size,1]))
      self.beta4  = tf.Variable(tf.zeros(shape=[hidden_size,1]))
      
      self.eps = tf.constant(eps)
      self.momentum = tf.constant(momentum)
      self.tf_BN = tf.keras.layers.BatchNormalization(axis=3,momentum=momentum, center=True , scale=True, epsilon=eps, beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='zeros',trainable=True)
      self.tf_LN = tf.keras.layers.LayerNormalization(axis=1, center=True , scale=True, epsilon=eps, beta_initializer='zeros',gamma_initializer='ones',trainable=True)
      #self.tf_WN = tfa.layers.wrappers.WeightNormalization(data_init=True, input_shape=(batch_size,h_size,w_size,filter_n))
      self.variables = [self.W1,self.b1, self.W2, self.b2, self.W3, self.b3, self.gamma1, self.beta1, self.gamma2, self.beta2, self.gamma3, self.beta3, self.gamma4, self.beta4, self.g1, self.v1]
      self.device = device
      self.size_output = output_size
  
  def flatten(self,X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
    
      X_padded = tf.pad(X, [[0,0], [padding, padding], [padding, padding], [0,0]])

      windows = []
      for y in range(out_h):
          for x in range(out_w):
              window = tf.slice(X_padded, [0, y*stride, x*stride, 0], [-1, window_h, window_w, -1])
              windows.append(window)
      stacked = tf.stack(windows) # shape : [out_h, out_w, n, filter_h, filter_w, c]

      return tf.reshape(stacked, [-1, window_c*window_w*window_h])
  
  def convolution(self,X, W, b, padding, stride):
      n, h, w, c = map(lambda d: d, X.get_shape())
      filter_h, filter_w, filter_c, filter_n = [d for d in W.get_shape()]
    
      out_h = (h + 2*padding - filter_h)//stride + 1
      out_w = (w + 2*padding - filter_w)//stride + 1

      X_flat = self.flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
      W_flat = tf.reshape(W, [filter_h*filter_w*filter_c, filter_n])
    
      z = tf.matmul(X_flat, W_flat) + b     # b: 1 X filter_n
    
      return tf.transpose(tf.reshape(z, [out_h, out_w, n, filter_n]), [2, 0, 1, 3])
    
 
    
  def relu(self,X):
      return tf.maximum(X, tf.zeros_like(X))
    
  def max_pool(self,X, pool_h, pool_w, padding, stride):
      n, h, w, c = [d for d in X.get_shape()]
    
      out_h = (h + 2*padding - pool_h)//stride + 1
      out_w = (w + 2*padding - pool_w)//stride + 1

      X_flat = self.flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)

      pool = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h*pool_w, c]), axis=3)
      return tf.transpose(pool, [2, 0, 1, 3])

    
  def affine(self,X, W, b):
      n = X.get_shape()[0] # number of samples
      X_flat = tf.reshape(X, [n, -1])
      return tf.matmul(X_flat, W) + b 
    
  def softmax(self,X):
      X_centered = X - tf.reduce_max(X) # to avoid overflow
      X_exp = tf.exp(X_centered)
      exp_sum = tf.reduce_sum(X_exp, axis=1)
      return tf.transpose(tf.transpose(X_exp) / exp_sum) 
    
  
  def cross_entropy_error(self,yhat, y):
      return -tf.reduce_mean(tf.log(tf.reduce_sum(yhat * y, axis=1)))
    
  
  def forward(self,X, training=True):
      if self.device is not None:
        with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
          self.y = self.compute_output(X, training)
      else:
        self.y = self.compute_output(X, training)
      
      return self.y
    
    
  def loss(self, y_pred, y_true):
      '''
      y_pred - Tensor of shape (batch_size, size_output)
      y_true - Tensor of shape (batch_size, size_output)
      '''
      y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
      y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
      return tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_true_tf))
    
    
  def backward(self, X_train, y_train):
      """
      backward pass
      """
      # optimizer
      # Test with SGD,Adam, RMSProp
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
      #predicted = self.forward(X_train)
      #current_loss = self.loss(predicted, y_train)
      #optimizer.minimize(current_loss, self.variables)

      #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      with tf.GradientTape() as tape:
          predicted = self.forward(X_train, True)
          current_loss = self.loss(predicted, y_train)
      #print(predicted)
      #print(current_loss)
      #current_loss_tf = tf.cast(current_loss, dtype=tf.float32)
      self._variables = self.variables
      if self.compare_BN and self.batch_norm1 == False:
        self._variables.extend([self.tf_BN.gamma, self.tf_BN.beta])
      if self.compare_LN and self.layer_norm1 == False:
        self._variables.extend([self.tf_LN.gamma, self.tf_LN.beta])
      if self.compare_WN and self.weight_norm == False:
        self._variables.extend([self.tf_WN.gamma, self.tf_WN.beta])
      self.grads = tape.gradient(current_loss, self._variables)
      optimizer.apply_gradients(zip(self.grads, self._variables),
                              global_step=tf.compat.v1.train.get_or_create_global_step())
      return predicted
      

  def BatchNorm1(self, input_x, gamma, beta, training):
      #print(input_x.shape) #(64, 28, 28, 30)
      if training:
        mean = tf.math.reduce_mean(input_x, [0,1,2], keepdims=True)
        var = tf.math.reduce_variance(input_x, [0,1,2], keepdims=True)
        self.avg_mean1 = self.momentum * self.avg_mean1 + (1 - self.momentum) * mean
        self.avg_var1 = self.momentum * self.avg_var1 + (1 - self.momentum) * var
        x_hat = (input_x - mean) / tf.sqrt(var + self.eps)
        out = gamma * x_hat + beta
      else:
        x_hat = (input_x - self.avg_mean1) / tf.sqrt(self.avg_var1 + self.eps)
        out = gamma * x_hat + beta
      return out

  def BatchNorm2(self, input_x, gamma, beta, training):
      #print(input_x.shape) #(64, 100)
      if training:
        mean = tf.math.reduce_mean(input_x, [0], keepdims=True)
        var = tf.math.reduce_variance(input_x, [0], keepdims=True)
        self.avg_mean2 = self.momentum * self.avg_mean2 + (1 - self.momentum) * mean
        self.avg_var2 = self.momentum * self.avg_var2 + (1 - self.momentum) * var
        x_hat = (input_x - mean) / tf.sqrt(var + self.eps)
        out = gamma * x_hat + beta
      else:
        x_hat = (input_x - self.avg_mean2) / tf.sqrt(self.avg_var2 + self.eps)
        out = gamma * x_hat + beta
      return out
    
  def LayerNorm1(self, input_x, gamma, beta, training):
      mean = tf.math.reduce_mean(input_x, [1,2,3], keepdims=True)
      var = tf.math.reduce_variance(input_x, [1,2,3], keepdims=True)
      x_hat = (input_x - mean) / tf.sqrt(var + self.eps)
      out = gamma * x_hat + beta
      return out

  def LayerNorm2(self, input_x, gamma, beta, training):
      mean = tf.math.reduce_mean(input_x, [1], keepdims=True)
      var = tf.math.reduce_variance(input_x, [1], keepdims=True)
      x_hat = (input_x - mean) / tf.sqrt(var + self.eps)
      out = gamma * x_hat + beta
      return out
    
  def compute_output(self,X, training):
      if self.weight_norm:
        self.W1 = self.g1 / tf.sqrt(tf.reduce_sum(tf.square(self.v1),[0])) * self.v1
      if self.compare_WN and self.weight_norm == False:
        self.W1 = self.tf_WN(X)
      #self.W1 = tf.nn.l2_normalize(self.W1, axis=list(range(self.W1.shape.ndims - 1))) * self.g1
      conv_layer1 = self.convolution(X, self.W1, self.b1, padding=2, stride=1)
      if self.batch_norm1:
        conv_layer1 = self.BatchNorm1(conv_layer1, self.gamma1, self.beta1, training)
      if self.compare_BN and self.batch_norm1==False:
        conv_layer1 = self.tf_BN(conv_layer1, training=training)
      if self.layer_norm1:
        conv_layer1 = self.LayerNorm1(conv_layer1, self.gamma3, self.beta3, training)
      if self.compare_LN  and self.layer_norm1==False:
        conv_layer1 = self.tf_LN(conv_layer1, training=training)
      conv_activation = self.relu(conv_layer1)
      conv_pool = self.max_pool(conv_activation, pool_h=2, pool_w=2, padding=0, stride=2)
      conv_affine =self.affine(conv_pool, self.W2,self.b2)
      if self.batch_norm2:
        conv_affine = self.BatchNorm2(conv_affine, self.gamma2, self.beta2, training)
      if self.layer_norm2:
        conv_affine = self.LayerNorm2(conv_affine, self.gamma4, self.beta4, training)
      conv_affine_activation = self.relu(conv_affine)
      
      conv_affine_1 = self.affine(conv_affine_activation, self.W3, self.b3)
      return conv_affine_1

test_dataset = tf.data.Dataset.from_tensor_slices((data.test.images, data.test.labels)).map(lambda x, y: (x, tf.cast(y, tf.float32)))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=batch_size,drop_remainder=True)
def test(mymodel):
   _accs = 0
   logitsall = []
   yall=[]
   for ids,(_x,_y) in test_dataset.enumerate():
      logits = mymodel.compute_output(_x, False)
      logitsall.extend(logits)
      yall.extend(_y)
   res = accuracy_function(logitsall, yall)
   print ("Test Accuracy = {:.3%}".format(res))
   return res

def compare(m1, m2):
  res1 = []
  res2 = []
  gamma = []
  beta = []
  start = time.time()
  inx = 0
  for epoch in range(num_epochs):
        train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels)).map(lambda x, y: (x, tf.cast(y, tf.float32)))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=batch_size,drop_remainder=True)
        acc1 = tf.keras.metrics.Accuracy()
        acc2 = tf.keras.metrics.Accuracy()
        for inputs, outputs in train_ds:
            logits2 = m2.backward(inputs, outputs)
            acc2(tf.argmax(tf.nn.softmax(logits2),1), tf.argmax(outputs,1))
            logits1 = m1.backward(inputs, outputs)
            acc1(tf.argmax(tf.nn.softmax(logits1),1), tf.argmax(outputs,1))

            '''
            if m1.compare_BN == True:
                gamma.append(tf.reduce_mean(m2.tf_BN.gamma-m1.gamma1))
                beta.append(tf.reduce_mean(m2.tf_BN.beta-m1.beta1))
            elif m1.compare_LN == True:
                gamma.append(tf.reduce_mean(m2.tf_LN.gamma-m1.gamma3))
                beta.append(tf.reduce_mean(m2.tf_LN.beta-m1.beta3))
                print(gamma[-1], beta[-1])
            if len(gamma) == 15:
                return res1, res2, gamma, beta
            '''

            inx += batch_size
            if (inx % (batch_size*100) == 0):
              print('mine...')
              res1.append(test(m1).numpy())
              print('tf...')
              res2.append(test(m2).numpy())

        print ("Training Accuracy = {:.3%}, {:.3%}".format(acc1.result(),acc2.result()))
  end = time.time()
  print(end-start)
  return res1, res2, gamma, beta

def train(mlp_on_cpu):
  res = []
  start = time.time()
  inx = 0
  for epoch in range(num_epochs):
        train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels)).map(lambda x, y: (x, tf.cast(y, tf.float32)))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=batch_size,drop_remainder=True)
        loss_total = tf.Variable(0, dtype=tf.float32)
        accuracy_total = tf.Variable(0, dtype=tf.float32)
        acc = tf.keras.metrics.Accuracy()
        for inputs, outputs in train_ds:
            logits = mlp_on_cpu.backward(inputs, outputs)
            preds = tf.argmax(tf.nn.softmax(logits),1)
            loss_total = loss_total + mlp_on_cpu.loss(logits, outputs)
            acc(preds, tf.argmax(outputs,1))
            inx += batch_size
            if (inx % (batch_size*100) == 0):
              res.append(test(mlp_on_cpu).numpy())    
        print('Number of Epoch = {} - loss:= {:.4f}'.format(epoch + 1, loss_total.numpy() / num_train))
        print ("Training Accuracy = {:.3%}".format(acc.result()))
  end = time.time()
  print(end-start)
  return res

import matplotlib.pyplot as plt
def plot(name, acc, acc_BN, norm):
  fig, ax = plt.subplots()
  ax.plot(range(0,len(acc),1),acc, label='Without '+norm)
  ax.plot(range(0,len(acc),1),acc_BN, label='With '+norm)
  ax.set_xlabel('Training steps')
  ax.set_ylabel('Accuracy')
  ax.set_xticks(range(1,len(acc)+1,1)*batch_size)
  ax.legend(loc=4)
  plt.savefig(name+'.pdf', dpi=600)

def com_plot(name, acc, acc_BN, norm):
  fig, ax = plt.subplots()
  ax.plot(range(0,len(acc),1),acc, label='With tf  '+norm)
  ax.plot(range(0,len(acc),1),acc_BN, label='With my  '+norm)
  ax.set_xlabel('Training steps')
  ax.set_ylabel('Accuracy')
  ax.set_xticks(range(1,len(acc)+1,1)*batch_size)
  ax.legend(loc=4)
  plt.savefig(name+'.pdf', dpi=600)

def com_plot_param(name, acc, acc_BN):
  fig, ax = plt.subplots()
  ax.plot(range(0,len(acc),1),acc, label='Gamma')
  ax.plot(range(0,len(acc),1),acc_BN, label='Beta')
  ax.set_xlabel('Batch')
  ax.set_ylabel('Accuracy')
  ax.set_xticks(range(1,len(acc)+1,1)*batch_size)
  ax.legend(loc=4)
  plt.savefig(name+'.pdf', dpi=600)
'''
#experiment 1 [0.8434, 0.8886, 0.9067, 0.9138, 0.9279, 0.9285, 0.9351, 0.9379, 0.9436, 0.9465, 0.9475, 0.9514, 0.9519, 0.954, 0.9535, 0.9556, 0.9587, 0.9582, 0.9621, 0.9629, 0.962, 0.9638, 0.9647, 0.9636, 0.9649, 0.9661, 0.9681]


mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=True, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False)
acc_BN1 = train(mlp_on_cpu)
print(acc_BN1)

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False)
acc = train(mlp_on_cpu)
print(acc)
plot('bn1', acc, acc_BN1, 'BN')


#experiment 2 [0.8416, 0.8916, 0.9101, 0.9165, 0.9235, 0.9284, 0.9358, 0.9372, 0.9396, 0.9448, 0.9452, 0.9512, 0.9532, 0.9533, 0.9511, 0.9521, 0.9556, 0.9484, 0.9583, 0.9606, 0.9602, 0.9601, 0.9638, 0.9628, 0.9634, 0.9642, 0.9647]

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=True, layer_norm2=False, weight_norm=False)
acc_LN1 = train(mlp_on_cpu)
print(acc_LN1)

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False)
acc = train(mlp_on_cpu)
print(acc)
plot('ln1', acc, acc_LN1, 'LN')


'''
#experiment 3 [0.673, 0.8548, 0.886, 0.9017, 0.9106, 0.9175, 0.9255, 0.9261, 0.9351, 0.9371, 0.9398, 0.9436, 0.9456, 0.9482, 0.9495, 0.9487, 0.9536, 0.9529, 0.9564, 0.9566, 0.9549, 0.9588, 0.9613, 0.9625, 0.963, 0.9622, 0.9648]

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=True)  
acc_WN1 = train(mlp_on_cpu)                                                                         
print(acc_WN1)                                                                                                                                         
mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False) 
acc = train(mlp_on_cpu)                                                                             
print(acc)                                                                             
plot('wn1', acc, acc_WN1, 'WN')    

'''

#experiment 5
my_mlp_on_cpu = CNN(hidden_size, output_size, device='cpu', batch_norm1=True, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False, compare_BN=True, compare_LN=False, compare_WN=False)  
tf_mlp_on_cpu = CNN(hidden_size, output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False, compare_BN=True, compare_LN=False, compare_WN=False) 
acc_my_BN, acc_tf_BN, gamma, beta = compare(my_mlp_on_cpu,tf_mlp_on_cpu)                                                                             
com_plot('com_BN1', acc_tf_BN, acc_my_BN, 'BN') 
com_plot_param('com_param', gamma, beta) 

#experiment 6
my_mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=True, layer_norm2=False, weight_norm=False, compare_BN=False, compare_LN=True, compare_WN=False)
tf_mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False, compare_BN=False, compare_LN=True, compare_WN=False)
acc_my_BN, acc_tf_BN, gamma, beta = compare(my_mlp_on_cpu, tf_mlp_on_cpu)                             
com_plot('com_LN1', acc_tf_BN, acc_my_BN, 'LN')
com_plot_param('com_param', gamma, beta) 


my_mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=True, compare_BN=False, compare_LN=False, compare_WN=True)
tf_mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=False, layer_norm1=False, layer_norm2=False, weight_norm=False, compare_BN=False, compare_LN=False, compare_WN=True)
acc_my_BN, acc_tf_BN, gamma, beta = compare(my_mlp_on_cpu, tf_mlp_on_cpu)                             
com_plot('com_WN1', acc_tf_BN, acc_my_BN, 'WN')
com_plot_param('com_param_WN', gamma, beta) 

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=False, batch_norm2=True, layer_norm1=False, layer_norm2=False, weight_norm=False)
acc_BN2 = train(mlp_on_cpu)
print(acc_BN2)
plot('bn2', acc, acc_BN2)

mlp_on_cpu = CNN(hidden_size,output_size, device='cpu', batch_norm1=True, batch_norm2=True, layer_norm1=False, layer_norm2=False, weight_norm=False)
acc_BN12 = train(mlp_on_cpu)
print(acc_BN12)
plot('bn12', acc, acc_BN12)
'''
