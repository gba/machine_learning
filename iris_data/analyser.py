#!/usr/bin/env python3

'''
Program for machine learning analysis of simple
data sets.

Author: Gustav Baardsen

A clustering algorithm is used to distinguish clusters
of points realted to the classical Iris data set.

'''

import sys
import numpy as np

from copy import deepcopy



def read_data(data_file):
    '''
    Reads data of the format 
    
    float,float,float, ...,float,string
    
    and returns a Numpy array for the floating point
    numbers and a list containing the corresponding
    strings.
    '''
    f = open(data_file, 'r')
    
    numbers = []
    names = []
    for l in f:
        
        string_list = l.split(",")
        
        names.append(string_list[-1].strip())
        row = []
        for i in range(len(string_list) - 1):
            
            row.append(float(string_list[i]))
            
        numbers.append(row)
        
    f.close()
    
    return np.array(numbers), names


def print_data(numbers, names):
    '''
    '''
    n = numbers.shape[0]
    
    print('\nNumbers; Name\n')
    for i in range(n):
        
        print(numbers[i, :], ', ', names[i])
        
    print('')
    
    
def split_data(numbers, names):
    '''
    Split the data set into different chunks for each
    label.
    '''
    n_elements = numbers.shape[0]
    
    names_unique = list(set(names))
    
    numbers_classes = []
    for name in names_unique:
        
        numbers_this = []
        for i in range(n_elements):
            
            if names[i] == name:
                
                numbers_this.append(numbers[i, :])
                
        numbers_classes.append(np.array(numbers_this))
        
    return names_unique, numbers_classes


def get_traning_test_sets(data, ratio_training):
    '''
    Given a list of data sets for a number of categories, 
    return training and test such that 'ratio_training' % 
    of the elements in each data set is used
    for the training set.
    
    data          : A list of Numpy arrays containing data sets
                    corresponding different categories.
    '''
    training_sets = []
    test_sets = []
    for i in range(len(data)):
        
        n_elements = data[i].shape[0]
        n_training = int(ratio_training * n_elements)
        
        training_sets.append(data[i][0:n_training, :])
        test_sets.append(data[i][n_training:, :])
        
    return training_sets, test_sets


def plot_clusters(points, clusters,
                  xlabel='Property 1',
                  ylabel='Property 2',
                  title='k-means clustering',
                  plot_name='clusters'):
    
    import numpy as np
    import matplotlib 
    import matplotlib.pyplot as pyplot
    
    from matplotlib import rcParams
    from matplotlib.ticker import AutoMinorLocator

    #     Use Latex font as default
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['DejaVu Serif']
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rc('legend',**{'fontsize':14})
    
    #     Initialize a figure
    fig = pyplot.figure(figsize=(7, 6.3), dpi=100)
    ax = pyplot.gca()

    for k in range(len(points)):

        data = points[k]
        rand_color = np.random.rand(3)
        
        ax.plot(data[:, 0], data[:, 1],
                color=rand_color,
                marker='s',
                linestyle='None',
                markeredgecolor=rand_color,
                label='Cluster ' + str(k + 1))
        
    ax.plot(clusters[:, 0], clusters[:, 1],
            color='r', marker='^',
            linestyle='None',
            markeredgecolor='r',
            label='Cluster center')
    
    #     Set axis labels
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
    pyplot.title(title)
    
    #     Legend
    l = pyplot.legend(loc='lower right', labelspacing=0.1)
    frame = l.get_frame()
    frame.set_lw(0.6)
    
    #     Save the plot
    name = plot_name 
    fig.savefig(name + '.pdf', format='pdf', dpi=1000)

    pyplot.show()
    

def plot_2d(numbers, names, clusters,
            xlabel='Property 1',
            ylabel='Property 2',
            title='2D plot of data set',
            plot_name='cross_section'):
    
    names_unique, data_sets = split_data(numbers, names)
    
    import numpy as np
    import matplotlib 
    import matplotlib.pyplot as pyplot

    from matplotlib import rcParams
    from matplotlib.ticker import AutoMinorLocator


    #     Use Latex font as default
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['DejaVu Serif']
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rc('legend',**{'fontsize':14})
    
    #     Initialize a figure
    fig = pyplot.figure(figsize=(7, 6.3), dpi=100)
    ax = pyplot.gca()
    
    for i in range(len(names_unique)):
        
        name = names_unique[i]
        rand_color = np.random.rand(3)
        
        ax.plot(data_sets[i][:, 0], data_sets[i][:, 1],
                color=rand_color, marker='s',
                linestyle='None',
                markeredgecolor=rand_color,
                label=name)

    ax.plot(clusters[:, 0], clusters[:, 1],
            color='r', marker='^',
            linestyle='None',
            markeredgecolor='r',
            label='Cluster center')
        
    #     Set axis labels
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
    pyplot.title(title)
    
    #     Legend
    l = pyplot.legend(loc='lower right', labelspacing=0.1)
    frame = l.get_frame()
    frame.set_lw(0.6)
    
    #     Save the plot
    name = plot_name 
    fig.savefig(name + '.pdf', format='pdf', dpi=1000)

    pyplot.show()
    

def get_kmeans(data, n_clusters, tolerance=1e-5):
    '''
    The k-means algorithm, as given in Figure 7.3 of 
    E. Alpaydin, Introduction to Machine Learning, 
    The MIT Press, Cambridge, Massachusetts (2004).
    
    data          : Data array. 
    n_clusters    : Number of clusters.
    ''' 
    (n_points, n_dimensions) = data.shape

    centers = np.mean(data, axis=0) + \
              np.random.rand(n_clusters, n_dimensions) * \
              np.std(data, axis=0)
    
    error = 2 * tolerance
    while error > tolerance:
        
        category = np.zeros((n_points, n_clusters),
                            dtype=int)
        for p in range(n_points):
            
            data_point = data[p, :]
            
            min_distance = 1e15
            cluster_index = -1
            for k in range(n_clusters):
                
                center = centers[k, :]
                
                distance = \
                    np.linalg.norm(center - data_point)
                
                if distance < min_distance:
                    cluster_index = k
                    
                    min_distance = deepcopy(distance)
                    
            category[p, cluster_index] = 1
            
        old_centers = deepcopy(centers)
        centers = np.zeros((n_clusters, n_dimensions))
        for k in range(n_clusters):
            
            associated = category[:, k]
            
            if np.sum(associated) > 0:
                centers[k, :] = np.dot(associated.T, data) / \
                                np.sum(associated)
            else:
                centers[k, :] = old_centers[k, :]
            
        diff_array = old_centers - centers
        
        differences = np.linalg.norm(diff_array, axis=1)
        
        error = np.amax(differences)

    points = []
    for k in range(n_clusters):
        
        associated = np.where(category[:, k])
        points_this = data[associated, :][0]
        
        points.append(points_this)
        
    return centers, points



def main():
    
    #
    # Read the classical Iris data set, which contains
    # data about three different flowers.
    #
    data_file = "data/iris.data"
    
    numbers, names = read_data(data_file)
    
    # Print the data set to the command line
    print_iris = True
    if print_iris:
        print_data(numbers, names)
        
    # Set up a list of parameter pairs
    n = 4
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append([i, j])
            
    #
    # For each pair of parameters, find two clusters in the
    # correlation plot.
    #
    for p in pairs:
        
        parameters = np.array(p)

        #
        # Search for two clusters
        #
        n_clusters = 2
        tolerance = 1e-7
        clusters, points = get_kmeans(numbers[:, parameters],
                                      n_clusters,
                                      tolerance)
        #print('Clusters:', clusters)
        
        suffix = '_' + str(p[0]) + '_' + str(p[1])
        
        plot_2d(numbers[:, parameters],
                names,
                clusters,
                plot_name = 'cross_section' + suffix)
        
        plot_clusters(points,
                      clusters,
                      plot_name = 'clusters' + suffix)
        
                  
    
if __name__ == "__main__":
    main()
    
    
    
