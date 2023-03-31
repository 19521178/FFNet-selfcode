import tensorflow as tf 
import numpy as np
from static_var import *

class FCLayer(object):
    def __init__(self, dim_input, dim_output, activation = TypesActivation.LINEAR, regularizer = None):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.activation = activation
        self.is_regular = regularizer is not None
        # self.regularizer = regularizer
        
        if self.activation == TypesActivation.RELU:
            # Kaiming initialization for special - ReLU
            self.W = tf.Variable(tf.random_normal([self.dim_input,self.dim_output],0.0,np.sqrt(2.0/float(self.dim_input))))
            self.b = tf.Variable(tf.zeros([self.dim_output]))
        else:
            # self.W = tf.Variable(tf.random_normal([self.dim_input,self.dim_output],0.0,np.sqrt(1.0/float(self.dim_input))))
            # Xavier initialization for none ReLU
            self.W = tf.Variable(tf.random_normal([self.dim_input,self.dim_output],0.0,np.sqrt(6.0/float(self.dim_input+self.dim_output))))
            self.b = tf.Variable(tf.zeros([self.dim_output]))

        if self.is_regular:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer(self.W))
        
    def forward(self, x):
        if self.activation == TypesActivation.RELU:
            return tf.nn.relu(tf.matmul(x, self.W) + self.b)
        elif self.activation == TypesActivation.LEAKYRELU:
            return tf.nn.leaky_relu(tf.matmul(x, self.W) + self.b)
        elif self.activation == TypesActivation.TANH:
            return tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        elif self.activation == TypesActivation.SIGMOID:
            return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)
        elif self.activation == TypesActivation.LINEAR:
            return tf.matmul(x, self.W) + self.b
        elif self.activation == TypesActivation.SOFTMAX:
            return tf.nn.softmax(tf.matmul(x, self.W) + self.b)

def create_FC_layers(dims, is_regular = False, regular_params = {}):
    if is_regular:
        if regular_params.type == TypesRegularizer.L1:
            regularizer = tf.contrib.layers.l1_regularizer(regular_params.scale)
        elif regular_params.type == TypesRegularizer.L2:
            regularizer = tf.contrib.layers.l2_regularizer(regular_params.scale)
        elif regular_params.type == TypesRegularizer.L1_L2:
            regularizer = tf.contrib.layers.l1_l2_regularizer(regular_params.scale)
    else:
        regularizer = None
    
    layers = []
    for i in range(1, len(dims)-1):
        layers.append(FCLayer(dims[i-1], dims[i], TypesActivation.RELU, regularizer))
    layers.append(FCLayer(dims[-2], dims[-1], TypesActivation.LINEAR, regularizer))
    return layers



