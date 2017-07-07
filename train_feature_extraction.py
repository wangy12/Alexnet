import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import pandas as pd
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from scipy import ndimage
import time

from scipy.misc import imread

# Load traffic signs data.
training_file = './train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']

# Analyse the original data
# Number of training examples
n_train = X_train.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# How many unique classes/labels there are in the dataset.
filename = './signnames.csv'
n_classes = sum(1 for line in open(filename)) - 1

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# number of training examples = 39209

# Plotting the count of each sign
# Making a histogram for the distribution

# training data
# plt.hist(y_train, bins = n_classes, color = 'lime', histtype='step', fill=True, label='Train')
# plt.show()

# augment data

pics_in_class = np.bincount(y_train)

min_desired = min([int(np.mean(pics_in_class)), 600])
print('Generating new data.')

# Angles to be used to rotate images in additional data made
angles = [-10, 10, -15, 15, -5, 5, -2, 2]

# Iterate through each class
for i in range(len(pics_in_class)):
    # Check if less data than the mean
    if pics_in_class[i] < min_desired:
        # Count how many additional pictures we want
        new_wanted = min_desired - pics_in_class[i]
        picture = np.where(y_train == i)
        more_X = []
        more_y = []
        
        # Make the number of additional pictures needed to arrive at the mean
        for num in range(new_wanted):
            # Rotate images and append new ones to more_X, append the class to
            more_X.append(ndimage.rotate(X_train[picture][random.randint(0,pics_in_class[i] - 1)],random.choice(angles), reshape=False))
            more_y.append(i)
                
        # Append the pictures generated for each class back to the original da
        X_train = np.append(X_train, np.array(more_X), axis=0)
        y_train = np.append(y_train, np.array(more_y), axis=0)
                
print('Additional data generated. Any classes lacking data now have', min_desired, 'pictures')


n_train = X_train.shape[0]

print("Number of training examples =", n_train)


# use 1/3 of that as validation set

# Split data into training and validation sets.
# Splitting the training dataset into training and validation data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Shuffle the data prior to splitting
X_train, y_train = shuffle(X_train, y_train)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify = y_train, test_size=0.3, random_state=23)

print('Dataset successfully split for training and validation.')

n_valid = X_valid.shape[0]

print("Number of validation examples =", n_valid)


# Define placeholders and resize operation.

sign_names = pd.read_csv(filename)
nb_classes = 43

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
mu = 0
sigma = 0.1
    
fc8W = tf.Variable( tf.truncated_normal(shape, mean = mu, stddev = sigma) )
fc8b = tf.Variable(tf.zeros(nb_classes))
    
# logits = tf.matmul(fc7, fc8W) + fc8b
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)


# Define loss, training, accuracy operations.
rate = 0.002
EPOCHS = 10
BATCH_SIZE = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


# Train and evaluate the feature extraction model.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

    
t_start = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './alexnet_traffic_sign')
    print("Model saved")
    
    
duration = time.time() - t_start

print(duration)


