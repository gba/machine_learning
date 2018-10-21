
'''
Program for machine learning analysis of simple
data sets.

Author: Gustav Baardsen
'''

import numpy as np


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


def plot_2d(numbers, names,
            xlabel='Property 1',
            ylabel='Property 2',
            title='2D plot of data set',
            plot_name='cross_section'):
    
    names_unique, data_sets = split_data(numbers, names)
    
    import numpy as np
    import matplotlib 
    import matplotlib.pyplot as pyplot
    
    from matplotlib.ticker import AutoMinorLocator


    #     Use Latex font as default
    matplotlib.rc('font',
                  **{'family':'serif',
                     'serif':['Helvetica']})
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
    
    
    
    
    
def main():
    
    data_file = "data/iris.data"
    
    numbers, names = read_data(data_file)
    
    #print_data(numbers, names)
    
    parameters = np.array([1, 3])
    plot_2d(numbers[:, parameters], names)
    
    
if __name__ == "__main__":
    main()
    
    
    
