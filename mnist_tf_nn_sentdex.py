'''
https://www.youtube.com/watch?v=BhpvH5DuVu8&index=3&list=PLSPWNkAMSvv5DKeSVDbEbUKSsK4Z-GgiP
'''

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST_DATA_PATH = 'data/mnist'
mnist = input_data.read_data_sets(MNIST_DATA_PATH, one_hot=True)

# using 3 hidden layers, each layer has 500 nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# images will be classified into 10 classes/categories: 0 to 9
n_classes = 10
# feed 100 images into network each time
batch_size = 100

x = tf.placeholder('float',[ None, 28 * 28])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weidhts': tf.Variable(tf.random_normal([784, n_nodes_hl1]))
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl1)}
    
    hidden_2_layer = {'weidhts': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl2)}

    hidden_3_layer = {'weidhts': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl3]))
                      'biases': tf.Variable(tf.random_normal(n_nodes_hl3)}