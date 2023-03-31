import tensorflow as tf
from static_var import *
from utils.DictNamespace import DictNamespace as dict
import numpy as np

class NeuralNetwork(object):
    def __init__(self, opt_params, model_params, layers):
        self.opt_params = opt_params
        self.model_params = model_params
        self.layers = layers
        
        self.create_flow()
        self.build()
        
        self.saver = self.get_saver()
        
    def init_info(self):
        raise NotImplementedError()

    def create_flow(self):
        raise NotImplementedError()
    
    def build(self):
        #############################################
        # loss functions
        #############################################
        if self.opt_params.loss==TypesLoss.MEAN_SQUARE:
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - self.y), reduction_indices=[1]))
        elif self.opt_params.loss==TypesLoss.CROSS_ENTROPY:
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1)), tf.float32))

        #############################################
        # optimization method
        #############################################
        if self.opt_params.optimizer==TypesOptimizer.SGD:
            self.train_step = tf.train.GradientDescentOptimizer(self.opt_params.lr).minimize(self.loss)
        elif self.opt_params.optimizer==TypesOptimizer.RMSPROP:
            self.train_step = tf.train.RMSPropOptimizer(self.opt_params.lr).minimize(self.loss)
        elif self.opt_params.optimizer==TypesOptimizer.Adam:
            self.train_step = tf.train.AdamOptimizer(self.opt_params.lr)

        self.init = tf.global_variables_initializer()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        
        self.sess.run(self.init)
        
    def get_saver_mapping(self):
        saver_mapping = {}
        for nl in range(len(self.layers)):
            saver_mapping['W_'+str(nl)] = self.layers[nl].W
            saver_mapping['b_'+str(nl)] = self.layers[nl].b
        return saver_mapping
    
    def get_saver(self):
        saver_mapping = self.get_saver_mapping()
        return tf.train.Saver(saver_mapping, max_to_keep=10000)
        
    def train(self, previous_info_batch, x_batch, y_batch):
        raise NotImplementedError()
    
    def forward(self, previous_info_batch, x_batch):
        raise NotImplementedError()
    
    def save_model(self, path, file_name):
        save_path = self.saver.save(self.sess, path + file_name)
    
    def load_model(self, path, file_name):
        self.saver.restore(self.sess, path + file_name)
        
        
class FCNet(NeuralNetwork):
    def create_flow(self):
        self.x = tf.placeholder(tf.float32, [None, self.model_params.dim_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.model_params.dim_output])
        
        self.y = self.x
        for layer in self.layers:
            self.y = layer.forward(self.y)
            
    def init_info(self):
        return dict()
    
    def get_saver_mapping(self):
        saver_mapping = NeuralNetwork.get_saver_mapping(self)
        return saver_mapping
    
    def train(self, previous_info_batch, x_batch, y_batch):
        self.sess.run(self.train_step, feed_dict={self.x: x_batch, self.y_: y_batch})
    
    def forward_batch(self, previous_info_batch, x_batch):
        a_value = self.sess.run(self.y, feed_dict={self.x: x_batch})
        return dict(), a_value
    
    def forward(self, previous_info, x):
        a_value = self.sess.run(self.y, feed_dict={self.x: [x]})
        return dict(), a_value[0]
    
class GRUNet(NeuralNetwork):
    def create_flow(self):
        self.pre_h = tf.placeholder(tf.float32, [None, self.model_params.dim_hidden])
        
        self.x = tf.placeholder(tf.float32, [None, self.model_params.dim_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.model_params.dim_output])
        
        self.rnn_input = tf.reshape(self.x, [-1, 1, self.model_params.dim_input])
        self.GRU_cell = tf.keras.layers.GRU(units=self.model_params.dim_hidden, return_state=False)
        self.hidden_state = self.GRU_cell(self.rnn_input, initial_state=self.pre_h)
        
        self.y = self.hidden_state
        for layer in self.layers:
            self.y = layer.forward(self.y)
    
    def init_info(self):
        return dict(
            h = np.zeros(shape=(self.model_params.dim_hidden, ))
        )
    
    def get_saver_mapping(self):
        saver_mapping = NeuralNetwork.get_saver_mapping()
        for var in self.GRU_cell.trainable_variables:
            saver_mapping[var.name.replace('gru_cell/', '')] = var
        return saver_mapping
    
    def train(self, previous_info_batch, x_batch, y_batch):
        pre_h_batch = list(map(lambda x: x.h, previous_info_batch))
        self.sess.run(self.train_step, feed_dict={self.pre_h: pre_h_batch, self.x: x_batch, self.y_: y_batch})
    
    def forward_batch(self, previous_info_batch, x_batch):
        pre_h_batch = list(map(lambda x: x.h, previous_info_batch))
        h, a_value = self.sess.run([self.hidden_state, self.y], feed_dict={self.pre_h: pre_h_batch, self.x: x_batch})
        return dict(h = h), a_value
    
    def forward(self, previous_info, x):
        pre_h = previous_info.h 
        h, a_value = self.sess.run([self.hidden_state, self.y], feed_dict={self.pre_h: [pre_h], self.x: [x]})
        return dict(h = h[0]), a_value[0]
    
class LSTMNet(NeuralNetwork):
    def create_flow(self):
        self.pre_h = tf.placeholder(tf.float32, [None, self.model_params.dim_hidden])
        self.pre_c = tf.placeholder(tf.float32, [None, self.model_params.dim_hidden])
        
        self.x = tf.placeholder(tf.float32, [None, self.model_params.dim_input])
        self.y_ = tf.placeholder(tf.float32, [None, self.model_params.dim_output])
        
        self.rnn_input = tf.reshape(self.x, [-1, 1, self.model_params.dim_input])
        self.LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.model_params.dim_hidden, state_is_tuple=True)
        self.state_in = tf.nn.rnn_cell.LSTMStateTuple(self.pre_c, self.pre_h)
        self.output_LSTM , self.state_LSTM = tf.nn.dynamic_rnn(self.LSTM_cell, self.rnn_input, initial_state=self.state_in, time_major=False)
        self.LSTM_context, self.hidden_state = self.state_LSTM
        
        self.y = self.hidden_state
        for layer in self.layers:
            self.y = layer.forward(self.y)
    
    def init_info(self):
        return dict(
            h = np.zeros(shape=(self.model_params.dim_hidden, )),
            c = np.zeros(shape=(self.model_params.dim_hidden, )),
        )
    
    def get_saver_mapping(self):
        saver_mapping = NeuralNetwork.get_saver_mapping()
        for var in self.LSTM_cell.trainable_variables:
            saver_mapping[var.name] = var
        return saver_mapping
    
    def train(self, previous_info_batch, x_batch, y_batch):
        pre_h_batch = list(map(lambda x: x.h, previous_info_batch))
        pre_c_batch = list(map(lambda x: x.c, previous_info_batch))
        self.sess.run(self.train_step, feed_dict={self.pre_h: pre_h_batch, self.pre_c: pre_c_batch, self.x: x_batch, self.y_: y_batch})
    
    def forward_batch(self, previous_info_batch, x_batch):
        pre_h_batch = list(map(lambda x: x.h, previous_info_batch))
        pre_c_batch = list(map(lambda x: x.c, previous_info_batch))
        h, c, a_value = self.sess.run([self.hidden_state, self.LSTM_context, self.y], feed_dict={self.pre_h: pre_h_batch, self.pre_c: pre_c_batch, self.x: x_batch})
        return dict(h = h, c = c), a_value
    
    def forward(self, previous_info, x):
        pre_h = previous_info.h
        pre_c = previous_info.c 
        h, c, a_value = self.sess.run([self.hidden_state, self.LSTM_context, self.y], feed_dict={self.pre_h: [pre_h], self.pre_c: [pre_c], self.x: [x]})
        return dict(h = h[0], c = c[0]), a_value[0]