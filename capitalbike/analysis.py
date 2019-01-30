#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

'''
Author          : Gustav Baardsen
First version   : January 2019 
'''


class BikingData:
    
    def __init__(self,
                 data_file):
        
        self.data = pd.read_csv(data_file)
        
        # Randomly shuffle the rows
        self.data = shuffle(self.data)
        
        # Replace categories by numbers
        self.data['Bike number'] = \
            self.data['Bike number'].astype('category').cat.codes
        self.data['Member type'] = \
            self.data['Member type'].astype('category').cat.codes
        
        # Remove columns with categories that are not numbers
        self.data = self.data.drop(['Start date',
                                    'End date',
                                    'Start station',
                                    'End station'],
                                   axis = 1)
        #print(self.data)
        
    def get_categories(self):
        return self.data.columns
    
    def get_columns(self, col_indices):
        return self.data.get_values()[:, col_indices]
    
    def get_array(self):
        return self.data.get_values()
    
    
def normalize_columns(data):
    '''
    Normalize the columns of the array 'data'.
    '''
    min_values = np.amin(data, axis = 0)
    max_values = np.amax(np.absolute(data),
                         axis = 0)
    return (data - min_values) / (max_values - min_values)


def split_columns(data, ratios):
    '''
    Split the columns of the set 'data' into three parts
    according to 'ratios'.
    '''
    assert (len(ratios) == 2) and \
        (ratios[0] + ratios[1] < 1.0)
    
    n_total = data.shape[0]
    n1 = int(ratios[0] * n_total)
    n2 = min(n1 + int(ratios[1] * n_total),
             n_total)
    
    return data[:n1, :], data[n1:n2, :], data[n2:, :]


def get_train_test_dev_sets(input_data,
                            output_data,
                            ratios_train_dev = [0.8, 0.199]):
    '''
    Divide the data set into training, development, and test
    sets.
    '''
    # Normalize the data
    input_norm  = normalize_columns(input_data)
    output_norm = normalize_columns(output_data)
    
    # Divide into training and test sets
    in_train, in_dev, in_test = \
        split_columns(input_norm,
                      ratios_train_dev)
    out_train, out_dev, out_test = \
        split_columns(output_norm,
                      ratios_train_dev)
    
    n_train = in_train.shape[0]
    n_dev   = in_dev.shape[0] 
    n_test  = in_test.shape[0] 
    print('\nSize of training set:    ', n_train)
    print('Size of development set: ',   n_dev)
    print('Size of test set:        ',   n_test, '\n')
    
    return in_train, in_dev, in_test, \
        out_train, out_dev, out_test


def setup_trained_network(x_train,
                          y_train,
                          opt_algorithm = 'adam',
                          loss_function = 'mean_squared_error',
                          error_metrics = ['mse'],
                          n_per_layer = [8, 8, 8, 1],
                          n_iterations = 1,
                          n_batch = 30):
    
    # Construct a neural network
    network = ks.Sequential()
    network.add(ks.layers.Dense(units = n_per_layer[0],
                                activation = tf.nn.relu,
                                input_shape = x_train[0].shape))
    for i in [1, len(n_per_layer) - 2]:
        network.add(ks.layers.Dense(units = n_per_layer[1],
                                    activation = tf.nn.relu))
            
    network.add(ks.layers.Dense(units = n_per_layer[-1],
                                activation = tf.nn.tanh))
    
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


class NNPredictor:
    '''
    Class for prediciton using a neural network.
    '''
    network = None
    
    def __init__(self,
                 input_data,
                 output_data):
        
        self.input_data = input_data
        self.output_data = output_data
        
    def partition_data(self,
                       ratios_train_dev = [0.8, 0.199]):
        '''
        Partition the data into training, development, and
        test sets.
        '''
        self.in_train, self.in_dev, self.in_test, \
            self.out_train, self.out_dev, self.out_test = \
                get_train_test_dev_sets(self.input_data,
                                        self.output_data)
        
    def train_network(self,
                      optimization = 'adam',
                      loss = 'mean_squared_error',
                      metrics = ['mse'],
                      n_neurons = [8, 8, 8, 1],
                      n_mainloop = 1,
                      batch_size = 10):
        '''
        Train a neural network using self.x_train and 
        self.y_train.
        '''
        self.network = setup_trained_network(self.in_train,
                                             self.out_train,
                                             opt_algorithm = optimization,
                                             loss_function = loss,
                                             error_metrics = metrics,
                                             n_per_layer = n_neurons,
                                             n_iterations = n_mainloop,
                                             n_batch = batch_size)
        
    def validate(self):
        '''
        Validate the trained network.
        '''
        m = 'Error. The neural network has not been trained.'
        assert self.network is not None, m
        
        # Test the optimized network
        loss, mse = self.network.evaluate(self.in_dev,
                                          self.out_dev)
        print('\nErrors in the development set:')
        print('\nLoss:', loss)
        print('Mean-square error:', mse, '\n')
        
        
        predictions = self.network.predict(self.in_dev)
        
        n = self.in_dev.shape[0]
        points = np.arange(n)
        
        predict     = np.reshape(predictions, (n))
        output_dev  = np.reshape(self.out_dev, (n))
        
        v = Visualiser1D(x_points   = [points, points],
                         y_points   = [output_dev, predict, ],
                         colors     = ['b', 'r'],
                         markers    = ['s', '^'],
                         linetypes  = ['None', 'None'],
                         labels     = ['Correct values', 'Predictions'])
        v.make_plot(x_limits = [0, 50],
                    y_limits = [0, 0.1])
        
    def test(self):
        '''
        Test the trained network.
        '''
        predictions = network.predict(self.in_test)
        print('\nPrediction; Correct value')
        for i in range(self.in_test.shape[0]):
            
            print(predictions[i, 0], self.out_test[i, 0])
            #print(x_test[i, :], y_test[i, 0])
            
        print('')
        
        
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
        
        
def main():
    
    #
    # The data can be downloaded from
    # https://s3.amazonaws.com/capitalbikeshare-data/index.html
    #
    data_file = '2017Q1-capitalbikeshare-tripdata.csv'
    
    data = BikingData(data_file)
    
    input_data  = data.get_array()[:, 1:]
    output_data = data.get_array()[:, 0:1]
    
    analyser = NNPredictor(input_data,
                           output_data)
    
    analyser.partition_data()
    analyser.train_network()
    analyser.validate()
    
    
    
    #print(data.get_categories())
    
    
if __name__ == "__main__":
    main()
    
    
    
