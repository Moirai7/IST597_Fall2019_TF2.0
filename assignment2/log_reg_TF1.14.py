import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

assert tf.executing_eagerly()

import numpy as np
from sklearn.metrics import confusion_matrix

print(tf.__version__)

batch_size = 1000
buffer_size = 15000
n_epochs = 100
sz = 28
sz_flat = sz * sz
img_shape = (sz, sz)
num_classes = 10

train_data, test_data = tf.keras.datasets.fashion_mnist.load_data()
train_images, train_labels = train_data
test_images, test_labels = test_data

train_labels = tf.cast(train_labels, dtype=tf.int64)
test_labels = tf.cast(test_labels, dtype=tf.int64)
train_images = tf.reshape(tf.cast(train_images/255., tf.float32),[-1,sz_flat])
test_images = tf.reshape(tf.cast(test_images/255., tf.float32),[-1,sz_flat])

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(buffer_size).batch(batch_size)


def plot_images(images, cls_true, cls_pred = None):
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().reshape(img_shape), cmap="binary")
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0},Pred: {1}".format(cls_true[i],cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

weights = tf.Variable(tf.zeros([sz_flat, num_classes]),trainable=True)
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

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
optimize()
print_accuracy()
plot_weights()


from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects
def scatter(x, label):
    num_classes = len(np.unique(label))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    axs = plt.subplot(aspect='equal')
    _ = axs.scatter(x[:,0], x[:,1], lw=0, s=10, c=palette[label.astype(np.int)])
    axs.axis('off')
    axs.axis('tight')
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    txts = []

    for i in range(num_classes):
        xtext, ytext = np.median(x[label == i, :], axis=0)
        txt = axs.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)


def clustering():
  w = weights.numpy().T
  label = np.concatenate(np.array([ np.ones(sz_flat)*i for i in range(10)]))
  out = TSNE(n_components=2).fit_transform(np.concatenate(w).reshape(-1,1))
  print(out)
  scatter(out,label)
     

          
          
clustering()
