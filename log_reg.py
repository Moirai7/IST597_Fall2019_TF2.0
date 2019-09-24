import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

assert tf.executing_eagerly()

import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

batch_size = 1000
buffer_size = 15000
n_epochs = 100
'''define the images information'''
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

train_data, test_data = tf.keras.datasets.fashion_mnist.load_data()
train_images, train_labels = train_data
test_images, test_labels = test_data

train_labels = tf.cast(train_labels, dtype=tf.int64)
test_labels = tf.cast(test_labels, dtype=tf.int64)
train_images = tf.reshape(tf.cast(train_images/255., tf.float32),[-1,img_size_flat])
test_images = tf.reshape(tf.cast(test_images/255., tf.float32),[-1,img_size_flat])

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)


def plot_images(images, cls_true, cls_pred = None):
    #assert len(images) == len(cls_true) == 9  # only show 9 images
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().reshape(img_shape), cmap="binary")  # binary means black_white image
        # show the true and pred values
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0},Pred: {1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])  # remove the ticks
        ax.set_yticks([])
    plt.show()

#images = test_images[0:9]
#cls_true = test_labels[0:9]
#plot_images(images, cls_true)

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]),trainable=True)  # img_size_flat*num_classes
biases = tf.Variable(tf.zeros([num_classes]),trainable=True)

def model(X):
  logits = tf.matmul(X,weights) + biases 
  return logits

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
def cal_loss(logits, y_true):
  y_pred = tf.nn.softmax(logits)
  y_pred_cls = tf.argmax(y_pred, dimension=1)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, 
                                                       logits=logits)
  cost = tf.reduce_mean(cross_entropy)
  return cost


def optimize():
    for i in range(n_epochs):
        for (_x, _y) in tfe.Iterator(train_dataset):
          with tf.GradientTape() as tape:
            logits = model(_x)
            loss = cal_loss(logits, _y)
          gradients = tape.gradient(loss, [weights,biases])
          optimizer.apply_gradients(zip(gradients, [weights,biases]))      

'''define a function to print the accuracy'''    
def print_accuracy():
    logits = model(test_images)
    y_pred = tf.nn.softmax(logits)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, test_labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy on test-set:{0:.1%}".format(acc))
    
def plot_weights():
    w = weights.numpy()
    w_min = np.min(w)
    w_max = np.max(w)
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(0.3, 0.3)
    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[:,i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))
            ax.imshow(image, vmin=w_min,vmax=w_max,cmap="seismic")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

optimize()
print_accuracy()
plot_weights()
#print_confusion_martix()
