"""
author:Lan Zhang
"""
import time

import tensorflow as tf
print(tf.__version__)
#import tensorflow.contrib.eager as tfe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Create data
NUM_EXAMPLES = 10000

#define inputs and outputs with some noise 
X = tf.random.normal([NUM_EXAMPLES], seed=2612)  #inputs 
#noise = tf.random.normal([NUM_EXAMPLES], seed=2612) #noise 
#noise = tf.random.gamma([NUM_EXAMPLES],1, seed=2612) #noise 
noise = tf.random.normal([NUM_EXAMPLES], seed=2612) #noise 
y = X * 3 + 2 + noise  #true output
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Create variables.
W = tf.Variable(0.)
b = tf.Variable(0.)

batch_size = 1000
train_steps = 1000
learning_rate = 0.001

# Define the linear predictor.
def prediction(x):
  return x * W + b

# Define loss functions of the form: L(y, y_predicted)
def squared_loss(y, y_predicted):
  #return tf.square(y - y_predicted)
  return tf.reduce_mean(tf.square(y - y_predicted))

def huber_loss(y, y_predicted, m=1.0):
  """Huber loss."""
  error = y - y_predicted
  abs_error = tf.abs(error)
  quadratic = tf.math.minimum(abs_error, m)
  linear = abs_error - quadratic
  losses = tf.math.add(
        tf.multiply(
            tf.convert_to_tensor(0.5, dtype=quadratic.dtype),
            tf.multiply(quadratic, quadratic)),
        tf.multiply(m, linear))
  #return losses
  return tf.reduce_mean(losses)

def hybrid_loss(y, y_predicted):
  return huber_loss(y, y_predicted) + tf.reduce_mean(tf.abs(y - y_predicted))

def train(loss_func, lr = learning_rate):
  W.assign(0.)
  b.assign(0.)
  _loss = tf.Variable([0.])
  for i in range(train_steps):
    db = dataset.batch(batch_size)
    for idx, (_x, _y) in db.enumerate():
      with tf.GradientTape() as tape:
        yhat = prediction(_x)
        ###loss
        loss = loss_func(_y, yhat)
        lr = learning_rate/2 if loss == _loss else learning_rate
        #t = tf.cast(tf.equal(loss, _loss),dtype=tf.float32)
        #lr = learning_rate if tf.argmin(t) == 0 else learning_rate/2
        _loss = loss

      dW, db = tape.gradient(loss, [W, b])
      #update the paramters using Gradient Descent
      W.assign_sub(dW * lr)
      b.assign_sub(db* lr)

#%matplotlib inline
def plotdata(func_name):
  plt.plot(X, y, 'bo',label='org')
  plt.plot(X, X * W.numpy() + b.numpy(), 'r',
           label=func_name + " regression")
  plt.legend()
  plt.show()

def savefig(func_name):
  plt.rcParams['savefig.dpi'] = 600
  plt.rcParams['figure.dpi'] = 600
  plt.clf()
  plt.plot(X, y, 'bo',label='org')
  plt.plot(X, X * W.numpy() + b.numpy(), 'r',
           label=func_name + " regression")
  plt.legend()
  plt.savefig(func_name+'.pdf', dpi=600)

print(W.numpy())
print(b.numpy())
train(squared_loss)
savefig('squared')
#plotdata('squared')

print(W.numpy())
print(b.numpy())
train(huber_loss)
savefig('huber')
#plotdata('huber')

print(W.numpy())
print(b.numpy())
train(hybrid_loss)
savefig('hybrid')
print(W.numpy())
print(b.numpy())


#train(hybrid_loss, lr = 0.003)
