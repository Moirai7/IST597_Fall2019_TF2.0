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
n_epochs = 15
dropout_prob = False
n_train = None
n_test = None

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape( [60000,784])
test_images = test_images.reshape( [10000,784])

import sklearn
'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=30,random_state=0)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
'''

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

yhat = clf.fit(train_images, train_labels).predict(test_images)  
from sklearn.metrics import accuracy_score
def cal_acc(yhat, label):
   return accuracy_score(yhat, label)

print(cal_acc(yhat, test_labels))
