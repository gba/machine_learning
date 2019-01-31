#!/usr/bin/env python3

import sys

import mnist

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

import keras.utils.np_utils as ks_utils


'''
Author          : Gustav Baardsen
First version   : January 2019

A program to read integers from pictures using supervised
learning.

The exercise is a classical one, and was suggested on

https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/

under 'Identify your Digits Dataset'.

The pictures may be downloaded from 
http://yann.lecun.com/exdb/mnist/


The MNIST digit classification problem is discussed in detail 
in the Tensorflow manual 

https://www.tensorflow.org/tutorials/estimators/cnn 

for a case with a convolutional neural network.

'''


def setup_trained_network(x_train,
                          y_train,
                          opt_algorithm = 'adam',
                          loss_function = tf.losses.softmax_cross_entropy,
                          error_metrics = ['accuracy'],
                          n_per_layer = [8, 8, 8],
                          n_iterations = 5,
                          n_batch = 30):
    
    # Construct a neural network
    network = ks.Sequential()
    network.add(ks.layers.Dense(units = n_per_layer[0],
                                activation = tf.nn.relu,
                                input_shape = x_train[0].shape))
    for i in range(1, len(n_per_layer)):
        network.add(ks.layers.Dense(units = n_per_layer[i],
                                    activation = tf.nn.relu))
        
    network.add(ks.layers.Dense(units = y_train.shape[1],
                                activation = tf.nn.softmax))
    
    # Choose the optimization method and error metrics
    network.compile(optimizer = opt_algorithm,
                    loss = loss_function,
                    metrics = error_metrics)
    # Train the neural network
    network.fit(x_train,
                y_train,
                epochs = n_iterations,
                batch_size = n_batch)
    
    return network


# def setup_convolutional_network():


#     return network



def validate(network,
             input_val,
             output_val):
    '''
    Validate the trained 'network' using a development or 
    training set. 
    '''
    # Test the optimized network
    loss, metrics = network.evaluate(input_val,
                                     output_val)
    print('\nError in the development set:')
    print('\nLoss:', loss)
    print('Accuracy:', metrics, '\n')
    
    predictions = network.predict(input_val)
    
    n = input_val.shape[0]
    points = np.arange(n)
    
    predict       = np.argmax(predictions,
                              axis = 1)
    output_valid  = np.argmax(output_val,
                              axis = 1)
    
    v = Visualiser1D(x_points   = [points, points],
                     y_points   = [output_valid, predict],
                     colors     = ['b', 'r'],
                     markers    = ['s', '^'],
                     linetypes  = ['None', 'None'],
                     labels     = ['Correct values', 'Predictions'])
    n_plots = 10
    for i in range(n_plots):
        v.make_plot(x_limits = [50*i, 50*(i+1)],
                    y_limits = [-1, 11.0])
    
class Visualiser1D:
    '''
    Class for 1D plots.
    '''
    def __init__(self,
                 x_points,
                 y_points,
                 colors,
                 markers,
                 linetypes,
                 labels):
        
        m = 'Error. The lists x_points, y_points, and ' + \
            'labels must have the same lengths.'
        assert (len(x_points) == len(y_points)) and \
            (len(labels) == len(x_points)) and \
            (len(colors) == len(x_points)) and \
            (len(markers) == len(x_points)) and \
            (len(linetypes) == len(linetypes)), m
        
        for i in range(len(x_points)):
            m = 'Error. Each Numpy array x_points[i] must ' + \
                'have the same lenght as the corresponding ' + \
                'array y_points[i].'
            assert x_points[i].shape == y_points[i].shape, m
            
        self.x_points   = x_points
        self.y_points   = y_points
        self.colors     = colors
        self.markers    = markers
        self.linetypes = linetypes
        self.labels     = labels
        
    def make_plot(self,
                  legend_location = 'upper right',
                  line_type = 'None',
                  x_limits = None,
                  y_limits = None):
        
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = plt.gca()
        
        n_graphs = len(self.x_points)
        for i in range(n_graphs):
            
            ax.plot(self.x_points[i],
                    self.y_points[i],
                    color = self.colors[i],
                    marker = self.markers[i],
                    linestyle = self.linetypes[i],
                    label = self.labels[i])
            
        #     Legend
        l = plt.legend(loc = legend_location,
                       labelspacing = 0.1)
        frame = l.get_frame()
        frame.set_lw(0.6)
        
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        
        plt.show()

        

class DataSet:
    '''
    Container class for a picture data set.
    '''
    def __init__(self,
                 in_train,
                 out_train,
                 in_dev,
                 out_dev,
                 in_test,
                 out_test):
        
        self.in_train   = in_train
        self.out_train  = out_train
        self.in_dev     = in_dev
        self.out_dev    = out_dev
        self.in_test    = in_test
        self.out_test   = out_test
        
        
        
        
def floatlist2matrix(elements):
    '''
    Given a linear list containing N**2 floating-point
    numbers, return a (N, N) Numpy array.
    '''
    n = int(round(np.sqrt(len(elements))))
    
    return np.reshape(np.array(elements), (n, n))


def split_array_axis0(array, n):
    '''
    Split array along the first axis.
    '''
    return array[:n, :], array[n:, :]


def load_pictures(directory,
                  show = False):
    '''
    Load the MNIST pictures.
    '''
    pictures = mnist.MNIST(directory)
    
    in_train, out_train = pictures.load_training()
    in_test, out_test   = pictures.load_testing()
    
    if show:
        # Plot a few figures
        n = 10
        for i in range(n):
            
            pixels = floatlist2matrix(in_train[i])
            
            plt.imshow(pixels)
            plt.show()
            
    n_train   = len(in_train)
    n_devtest = len(in_test)
    n_dev     = int(n_devtest / 2)
    n_test    = n_devtest - n_dev
    
    print('\nSize of training set     :', n_train)
    print('Size of development set  :', n_dev)
    print('Size of test set         :', n_test, '\n')

    # Let the input be Numpy arrays, reshape, and
    # normalize the data.
    input_train      = np.array(in_train) / 255.0
    output_train     = np.reshape(np.array(out_train),
                                  (n_train, 1))
    input_devtest    = np.array(in_test) / 255.0
    output_devtest   = np.reshape(np.array(out_test),
                                  (n_devtest, 1))
    
    # Split into development and test sets
    input_dev, input_test = \
        split_array_axis0(input_devtest,
                          n_dev)
    output_dev, output_test = \
        split_array_axis0(output_devtest,
                          n_dev)
    #
    # [0] -> [1. 0. 0. 0. 0. 0, 0, 0, 0, 0]
    # [2] -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # [3] -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #
    # etc.
    #
    output_tr = ks_utils.to_categorical(output_train,
                                        num_classes = 10)
    output_de = ks_utils.to_categorical(output_dev,
                                        num_classes = 10)
    output_te = ks_utils.to_categorical(output_test,
                                        num_classes = 10)
    
    return DataSet(input_train,
                   output_tr,
                   input_dev,
                   output_de,
                   input_test,
                   output_te)

def main():
    
    show_pictures = False 
    directory = 'pictures'
    data = load_pictures(directory,
                         show_pictures)
    
    # Train a conventional deep neural network...
    n_mainloop = 15
    network_layout = [32, 16, 8, 8, 8]
    network = setup_trained_network(data.in_train,
                                    data.out_train,
                                    n_iterations = n_mainloop,
                                    n_per_layer = network_layout)
    # ...and validate it.
    validate(network,
             data.in_dev,
             data.out_dev)
    
    
    
        
if __name__ == "__main__":
    main()
    
    
