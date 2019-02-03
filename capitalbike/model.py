#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

import keras.utils.np_utils as ks_utils
import keras.regularizers as reg

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression


'''
Author          : Gustav Baardsen
First version   : January 2019 

A program to predict the class of biker based on data such
as duration, start station, end station, and bike ID.

The exercise was suggested on
https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/
under the title 'Trip history data set'.

The required data may be downloaded from
https://s3.amazonaws.com/capitalbikeshare-data/index.html


To write the code below, the tutorial

https://www.tensorflow.org/tutorials/keras/basic_classification

was useful.

'''


class BikingData:
    
    def __init__(self,
                 data_file):
        
        self.data = pd.read_csv(data_file)
        
        # Randomly shuffle the rows
        self.data = shuffle(self.data)
        #print(self.data)
        
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

    def print_categories(self):
        
        print('\nCategories in the data:\n')
        variables = self.get_categories()
        for i in range(variables.shape[0]):
            print(' ', variables[i])
            
        print('')
    
    
def normalize_columns(data):
    '''
    Normalize the columns of the array 'data' so that
    the mean is zero and the standard deviation is one.
    
    data      : data[i, j] is the sample i for variable j.
    '''
    
    m = 'Error. The parameter "data" must be a ' + \
        'two-dimensional Numpy.'
    assert len(data.shape) == 2, m
    
    mu = np.mean(data, axis = 0)
    s  = np.std(data, axis = 0, ddof = 1)
    
    return (data - mu) / s


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
    #output_norm = normalize_columns(output_data)
    
    # Divide into training and test sets
    in_train, in_dev, in_test = \
        split_columns(input_norm,
                      ratios_train_dev)
    out_train, out_dev, out_test = \
        split_columns(output_data,
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
                          loss_function = 'mean_absolute_percentage_error',
                          error_metrics = ['accuracy'],
                          n_per_layer = [8, 8, 8],
                          n_iterations = 1,
                          n_batch = 30,
                          reg_parameters = [0.01, 0.01]):
    '''
    Train a neural network with a given number of layers.
    '''
    
    # Construct a neural network
    network = ks.Sequential()
    network.add(ks.layers.Dense(units = n_per_layer[0],
                                activation = tf.nn.relu,
                                kernel_regularizer = reg.l1_l2(l1 = reg_parameters[0],
                                                               l2 = reg_parameters[1]),
                                input_shape = x_train[0].shape))
    for i in range(1, len(n_per_layer) - 1):
        network.add(ks.layers.Dense(units = n_per_layer[i],
                                    activation = tf.nn.relu,
                                    kernel_regularizer = reg.l1_l2(l1 = reg_parameters[0],
                                                                   l2 = reg_parameters[1])))
        
    network.add(ks.layers.Dense(units = y_train.shape[1],
                                activation = tf.nn.softmax,
                                kernel_regularizer = reg.l1_l2(l1 = reg_parameters[0],
                                                               l2 = reg_parameters[1])))
    
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


def ratio_correctly_predicted_false(predictions,
                                    labels):
    '''
    Compute the ratio of correctly predicted false values.
    '''
    sum_labels = predictions + labels

    n_labels = labels.shape[0]
    n_false  = n_labels - np.count_nonzero(labels)
    n_correctly_pred_false = n_labels - \
                        np.count_nonzero(sum_labels)
    return float(n_correctly_pred_false) / float(n_false)


def ratio_correctly_predicted_true(predictions,
                                   labels):
    '''
    Compute the ratio of correctly predicted true values.
    '''
    correctly_predicted_true = predictions * labels

    n_true = np.count_nonzero(labels)
    n_correctly_pred_true = np.sum(correctly_predicted_true)
    
    return float(n_correctly_pred_true) / float(n_true)
    

def classify_from_regression(predictions,
                             labels):
    '''
    Classify regression results.
    '''
    predicted_classes = (predictions > 0.5).astype(int)
    
    n_correctly_pred  = np.sum(predicted_classes == labels)
    n_samples         = labels.shape[0]
    
    return float(n_correctly_pred) / float(n_samples)
    

class NNPredictor:
    '''
    Class for prediciton using a neural network.
    '''
    network = None
    
    def __init__(self,
                 input_train,
                 input_dev,
                 input_test,
                 output_train,
                 output_dev,
                 output_test):
        
        self.in_train  = input_train
        self.in_dev    = input_dev
        self.in_test   = input_test
        self.out_train = output_train
        self.out_dev   = output_dev
        self.out_test  = output_test
        
    def train_network(self,
                      optimization = 'adam',
                      loss = tf.losses.softmax_cross_entropy,
                      metrics = ['accuracy'],
                      n_neurons = [8, 8, 8],
                      n_mainloop = 4,
                      batch_size = 30,
                      reg_params = [0.01, 0.01]):
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
                                             n_batch = batch_size,
                                             reg_parameters = reg_params)
        
    def validate(self,
                 input_val,
                 output_val):
        '''
        Validate the trained network.
        '''
        m = 'Error. The neural network has not been trained.'
        assert self.network is not None, m
        
        # Test the optimized network
        loss, metrics = self.network.evaluate(input_val,
                                              output_val)
        print('\nError in the development set:')
        print('\nLoss:', loss)
        print('Accuracy:', metrics, '\n')
        
        
        predictions = self.network.predict(input_val)
        
        n = input_val.shape[0]
        points = np.arange(n)
        
        predict     = np.argmax(predictions,
                                axis = 1)
        output_dev  = np.argmax(output_val,
                                axis = 1)
        
        v = Visualiser1D(x_points   = [points, points],
                         y_points   = [output_dev, predict],
                         colors     = ['b', 'r'],
                         markers    = ['s', '^'],
                         linetypes  = ['None', 'None'],
                         labels     = ['Correct values', 'Predictions'])
        n_plots = 10
        for i in range(n_plots):
            v.make_plot(x_limits = [50*i, 50*(i+1)],
                        y_limits = [0, 1.5])
            
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

    def get_predictions(self,
                        in_data):
        predictions_float = self.network.predict(in_data)
        return np.argmax(predictions_float,
                         axis = 1)


def plot_covariance(data):
    '''
    Plot the covariance matrix of 'data'.
    
    data      : data[i, j] is the sample i for variable j.
    '''
    normalized_data = normalize_columns(data)
    covariance = np.cov(np.transpose(normalized_data), ddof = 1)
    
    print('\nCovariance matrix for the data:\n')
    n = covariance.shape[0]
    
    for i in range(n):
        for j in range(n):
            print(covariance[i, j],
                  end = " ")
        print('')
    print('')
    
    
    plt.imshow(np.absolute(covariance))
    plt.colorbar()
    
    plt.title('Absolute values of the covariance matrix')
    
    items = np.arange(n, dtype=int)
    values = ['Duration',
              'Start station',
              'End station',
              'Bike',
              'Member type']
    plt.xticks(items, values, size = 8)
    plt.yticks(items, values, size = 8)
    
    plt.show()
    
    
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
    # Before running the program, download the data from
    #
    # https://s3.amazonaws.com/capitalbikeshare-data/index.html
    #
    # The program is run by calling ./analysis.py.
    #
    data_file = '2017Q1-capitalbikeshare-tripdata.csv'
    
    data = BikingData(data_file)

    data.print_categories()
    
    
    input_data  = data.get_array()[:, :4]
    output_data = data.get_array()[:, 4:5]
    

    # Plot the covariance matrix between the variables
    plot_covariance(data.get_array())
    
    
    output_data_c = ks_utils.to_categorical(output_data,
                                            num_classes = 2)
    
    # Partition the data set into training, development,
    # and test sets. The input and output data are also
    # normalized.
    input_train, input_dev, input_test, \
        output_train, output_dev, output_test = \
            get_train_test_dev_sets(input_data,
                                    output_data_c)
    
    # First, classify using linear regression with
    # a least-squares error functional
    regressor = LinearRegression()
    regressor.fit(input_train,
                  output_train[:, 0])
    output_linreg = regressor.predict(input_dev)
    
    accuracy = classify_from_regression(output_linreg,
                                        output_dev[:, 1])
    print('\nClassification accuracy of linear regression:',
          accuracy,
          '\n')

    
    # Create a neural-network predictor object
    analyser = NNPredictor(input_train,
                           input_dev,
                           input_test,
                           output_train,
                           output_dev,
                           output_test)
    
    opt_algorithm   = 'adam'
    loss_function   = tf.losses.softmax_cross_entropy
    error_metrics   = ['accuracy']
    n_per_layer     = [4, 4, 4]
    n_iterations    = 4
    n_batch         = 500
    reg_parameters  = [0.0, 0.0]
    
    print('Next, a neural network is trained...')
    analyser.train_network(optimization = opt_algorithm,
                           loss = loss_function,
                           metrics = error_metrics,
                           n_neurons = n_per_layer,
                           n_mainloop = n_iterations,
                           batch_size = n_batch,
                           reg_params = reg_parameters)
    
    analyser.validate(analyser.in_dev,
                      analyser.out_dev)
    
    
    #print(data.get_categories())
    
    
if __name__ == "__main__":
    main()
    
    
    
