import csv
import math
import sys
from copy import deepcopy
from random import random, randrange
import numpy
import matplotlib.pyplot as plt
from numpy import ones, vstack
from numpy.linalg import lstsq


def accuracy_metrics(actual, predicted):
    """ Calculate accuracy metric.

    Parameters
    ----------
    actual : list
        list of actual values
    predicted : list
        list of predicted values

    Returns
    -------
    float
        Accuracy
    """
    count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            count += 1
    return (count / float(len(actual))) * 100


def get_piecewise_equations(number_of_segments):
    """Create piecewise equations

    Parameters
    ----------
    number_of_segments : int
        Number of piecewise segments

    Returns
    -------
    list
        List of piecewise equations
    """
    number_of_segments = number_of_segments - 2
    min_x = -5
    max_x = 5
    x_range = max_x - min_x

    x_values = []
    y_values = []

    data_points = x_range / number_of_segments

    # Equation = slope, constant, range_1, range_2
    # y = mx +c ; if x > range_1 and x < range_2
    equations = [[0, 0, -sys.maxsize, min_x]]

    for i in range(number_of_segments + 1):
        x_values.append(min_x)
        min_x = min_x + data_points
        y_values.append(logistic_function(x_values[i]))

    for i in range(number_of_segments):
        points = [(x_values[i], y_values[i]), (x_values[i + 1], y_values[i + 1])]
        x, y = zip(*points)
        numpy_stack_values = vstack([x, ones(len(x))]).T
        slope, constant = lstsq(numpy_stack_values, y)[0]
        equations.append([slope, constant, x_values[i], x_values[i + 1]])
    equations.append([0, 1, max_x, sys.maxsize])

    return equations


def cross_validation_batch(data, number_of_splits):
    """ Split dataset into batches.

    Parameters
    ----------
    data : list
        Dataset.
    number_of_splits : int
        number of batches.

    Returns
    -------
    list
        dataset split in batches
    """
    batches = []
    copy_data = list(data)
    batch_size = int(len(data) / number_of_splits)
    for i in range(number_of_splits):
        batch = []
        while len(batch) < batch_size:
            batch.append(copy_data.pop(randrange(len(copy_data))))
        batches.append(batch)
    return batches


def plot_graph(title, xlabel, ylabel, x, y):
    """Plot graph using given parameters.

    Parameters
    ----------
    title : str
        Title for the graph
    xlabel : str
        X axis label.
    ylabel : str
        Y axis label.
    x : list
        x values
    y :list
        y values

    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.show()


def load_csv(filename):
    """Load csv data.

    Parameters
    ----------
    filename : str
        Filename.

    Returns
    -------
    list
        CSV data.

    """
    csv_data = []
    with open(filename, 'r') as fp:
        rows = csv.reader(fp)
        for csv_row in rows:
            if not csv_row:
                continue
            cols = []
            for col in csv_row:
                if col == '?' or not col:
                    col = None
                else:
                    col = float(col)
                cols.append(col)
            csv_data.append(cols)
    return csv_data


def str_column_to_int(dataset, column):
    """Convert string column values into int"""
    csv_lookup = {}
    for i, value in enumerate(set([row[column] for row in dataset])):
        csv_lookup[value] = i
    for row in dataset:
        row[column] = csv_lookup[row[column]]
    return csv_lookup


def get_min_max_dataset_values(dataset):
    """
    Parameters
    ----------
    dataset : list
        dataset.

    Returns
    -------
    list
        list of min and max values for each column
    """
    numpy.array(dataset)
    min_and_max_values = [[min(col), max(col)] for col in zip(*dataset)]
    return min_and_max_values


def normalize_data(data, min_and_max_values):
    """ Normalize dataset.

    Parameters
    ----------
    data : list
        Dataset.
    min_and_max_values : list
        min and max values for each column in the dataset.

    """
    for row in data:
        for i in range(len(row) - 1):
            row[i] = (row[i] - min_and_max_values[i][0]) / (min_and_max_values[i][1] - min_and_max_values[i][0])


def initialize_weights(neurons_in_layer1, neurons_in_layer2):
    """ Initialize weights randomly for neurons connected from layer 1 to layer 2.

    Parameters
    ----------
    neurons_in_layer1 : int
        Number of neurons in first layer.
    neurons_in_layer2 : int
        Number of neurons in second layer.

    Returns
    -------
    list
        Randomly generated weights.
        Example : [{'weights':[w1,w2...]}, ...]
    """
    weighted_neurons_with_bias = []
    total_number_of_weights_and_bias = neurons_in_layer1 + 1
    for i in range(neurons_in_layer2):
        random_weights = []
        for j in range(total_number_of_weights_and_bias):
            random_weights.append(random())
        weighted_neurons_with_bias.append({'weights': random_weights})
    return weighted_neurons_with_bias


def replace_missing_data_with_mean(dataset):
    """ Fill missing data with the mean of that column

    Parameters
    ----------
    dataset : list
        Dataset

    Returns
    -------
    list
        Dataset
    """
    ndataset = numpy.array(dataset, dtype=numpy.float)
    column_mean = numpy.nanmean(ndataset, axis=0)
    indices = numpy.where(numpy.isnan(ndataset))
    ndataset[indices] = numpy.take(column_mean, indices[1])
    return ndataset.tolist()


def create_neural_network(input_attributes, output_classes, hidden_layer_neurons):
    """ Create/Initialize a single hidden layered neural network based on input parameters.

    Parameters
    ----------
    input_attributes : int
        The number of input attributes or parameters for the neural network.
    output_classes : int
        The number of output classes for the neural network.
    hidden_layer_neurons : int
        The number of hidden neurons withint he hidden layer.

    Returns
    -------
    dict
        Neural network.
        Example :
        {
            'hidden_layer': [{'weights':[w1,w2...]},...],
            'output_layer': [{'weights':[w3,w4...]},...],
        }

    """
    neural_network = {'hidden_layer': initialize_weights(input_attributes, hidden_layer_neurons),
                      'output_layer': initialize_weights(hidden_layer_neurons, output_classes)}
    return neural_network


def neuron_activation_function(input_vector, weight_vector):
    """ Calculate weighted sum along with bias for each neuron's activation function.

    Parameters
    ----------
    input_vector : list
        Input vector values.
    weight_vector : list
        Weight vector values.

    Returns
    -------
    float
        Net weighted sum along with bias for each neuron.

    """
    net_weighted_sum = 0
    number_of_activation_function = len(weight_vector) - 1
    for i in range(number_of_activation_function):
        net_weighted_sum += float(input_vector[i]) * weight_vector[i]
    return net_weighted_sum + weight_vector[-1]


def logistic_function(net_neuron_activation_value):
    """ Calculate the logistic function value i.e. sigmoid function value.

    Parameters
    ----------
    net_neuron_activation_value : float
        Weighted sum along with bias for a neuron i.e. neuron activation value.

    Returns
    -------
    float
        Value of sigmoid function.
    """
    sigmoid = 1 / (1 + math.exp(-net_neuron_activation_value))
    return sigmoid


def derivative_of_logistic_function(logistic_function_value):
    """ Calculate derivative of sigmoid function.

    Parameters
    ----------
    logistic_function_value : float
        Value of sigmoid function.

    Returns
    -------
    float
        Derivative of sigmoid function.
    """
    derivative_sigmoid = logistic_function_value * (1 - logistic_function_value)
    return derivative_sigmoid


def piecewise_function(value, piecewise_equations):
    """ Calculate the logistic function value i.e. piecewise linea equation value.

    Parameters
    ----------
    value : float
        Value of linear equation's conditions
    piecewise_equations : list
        List of piecewise equations

    Returns
    -------
    float
        Value of sigmoid function.
    """
    for equation in piecewise_equations:
        if equation[2] <= value < equation[3]:
            return (value * equation[0]) + equation[1]


def derivative_of_piecewise_function(value, piecewise_equations):
    """Calculate derivative of linear equations.

    Parameters
    ----------
    value : float
        Value of logistic function.
    piecewise_equations : list
        List of piecewise equations

    Returns
    -------
    float
        Derivative of sigmoid function.
    """
    for equation in piecewise_equations:
        if equation[2] <= value < equation[3]:
            return equation[0]


def forward_propagation_with_piecewise(input_vector, neural_network_layer, piecewise_equations):
    """ Calculate the output vector for next layer in neural network based
        on net neuron activation values of previous layer in neural network.

    Parameters
    ----------
    input_vector : list
        List of input vector values.
    neural_network_layer : dict
        Layer attributes within the neural network.
    piecewise_equations : list
        List of piecewise equations

    Returns
    -------
    list
        List of input vector for the next layer within neural network.
    """
    next_input_vector = []
    for vector in neural_network_layer:
        net_neuron_activation_value = neuron_activation_function(input_vector, vector['weights'])
        vector['activation_output'] = piecewise_function(net_neuron_activation_value, piecewise_equations)
        next_input_vector.append(vector['activation_output'])
    return next_input_vector


def feed_forward_propagation_with_piecewise(neural_network, input_vector, piecewise_equations):
    """ Feed forward propagation for single hidden layered neural network.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    input_vector : list
        Input vector values
    piecewise_equations : list
        List of piecewise equations.

    Returns
    -------
    list
        Output vector values.

    """
    next_input_vector = forward_propagation_with_piecewise(input_vector, neural_network['hidden_layer'],
                                                           piecewise_equations)
    output_vector = forward_propagation(next_input_vector, neural_network['output_layer'])
    return output_vector


def back_propagation_with_piecewise(neural_network, expected_output, piecewise_equations):
    """ Back propagation algorithm for a single hidden layered neural network.
        Update neural network with calculated delta values.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    expected_output : list
        Expected output for a particular input.
    piecewise_equations : list
        List of piecewise equations.

    """
    errors = []
    for i in range(len(neural_network['output_layer'])):
        neuron = neural_network['output_layer'][i]
        errors.append(expected_output[i] - neuron['activation_output'])

    for i in range(len(neural_network['output_layer'])):
        neuron = neural_network['output_layer'][i]
        neuron['delta'] = errors[i] * derivative_of_logistic_function(neuron['activation_output'])

    errors = []
    for i in range(len(neural_network['hidden_layer'])):
        error = 0.0
        for neuron in neural_network['output_layer']:
            error += (neuron['weights'][i] * neuron['delta'])
        errors.append(error)
    for i in range(len(neural_network['hidden_layer'])):
        neuron = neural_network['hidden_layer'][i]
        neuron['delta'] = errors[i] * derivative_of_piecewise_function(neuron['activation_output'],
                                                                       piecewise_equations)


def forward_propagation(input_vector, neural_network_layer):
    """ Calculate the output vector for next layer in neural network based
        on net neuron activation values of previous layer in neural network.

    Paramters
    ---------
    input_vector : list
        List of input vector values.
    neural_network_layer : dict
        Layer attributes within the neural network.

    Returns
    -------
    list
        List of input vector for the next layer within neural network.
    """
    next_input_vector = []
    for vector in neural_network_layer:
        net_neuron_activation_value = neuron_activation_function(input_vector, vector['weights'])
        vector['activation_output'] = logistic_function(net_neuron_activation_value)
        next_input_vector.append(vector['activation_output'])
    return next_input_vector


def feed_forward_propagation(neural_network, input_vector):
    """Feed forward propagation for single hidden layered neural network.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    input_vector : list
        Input vector values

    Returns
    -------
    list
        Output vector values.

    """
    next_input_vector = forward_propagation(input_vector, neural_network['hidden_layer'])
    output_vector = forward_propagation(next_input_vector, neural_network['output_layer'])
    return output_vector


def back_propagation(neural_network, expected_output):
    """ Back propagation algorithm for a single hidden layered neural network.
        Update neural network with calculated delta values.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    expected_output : list
        Expected output for a particular input.

    """
    errors = []
    for i in range(len(neural_network['output_layer'])):
        neuron = neural_network['output_layer'][i]
        errors.append(expected_output[i] - neuron['activation_output'])

    for i in range(len(neural_network['output_layer'])):
        neuron = neural_network['output_layer'][i]
        neuron['delta'] = errors[i] * derivative_of_logistic_function(neuron['activation_output'])

    errors = []
    for i in range(len(neural_network['hidden_layer'])):
        error = 0.0
        for neuron in neural_network['output_layer']:
            error += (neuron['weights'][i] * neuron['delta'])
        errors.append(error)
    for i in range(len(neural_network['hidden_layer'])):
        neuron = neural_network['hidden_layer'][i]
        neuron['delta'] = errors[i] * derivative_of_logistic_function(neuron['activation_output'])


def redefine_weights(neural_network, input_vector, learning_rate):
    """ Update weights after back_propagation.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    input_vector : list
        Input vector values
    learning_rate : float
        learning rate.

    """
    input_values = input_vector[:-1]
    for neuron in neural_network['hidden_layer']:
        for i in range(len(input_values)):
            neuron['weights'][i] += learning_rate * neuron['delta'] * float(input_values[i])
        neuron['weights'][-1] += learning_rate * neuron['delta']

    input_values = [neuron['activation_output'] for neuron in neural_network['hidden_layer']]

    for neuron in neural_network['output_layer']:
        for i in range(len(input_values)):
            neuron['weights'][i] += learning_rate * neuron['delta'] * input_values[i]
        neuron['weights'][-1] += learning_rate * neuron['delta']


def train_neural_network(neural_network, train_dataset, test_dataset, learning_rate, iterations, feed_forward_function,
                         back_propagation_function, piecewise=None):
    """ Train neural network.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    train_dataset : list
        train dataset
    test_dataset : list
        test dataset
    learning_rate : float
        learning rate
    iterations : int
        number of iterations
    feed_forward_function : function
        feed forward function name
    back_propagation_function : function
        back propagation function name
    piecewise : list
        piecewise equations

    Returns
    -------
    x_value, mse, accuracies : (list, list, list)

    """
    mse = []
    x_value = []
    accuracies = []
    for iteration in range(iterations):
        x_value.append(iteration)
        error = 0
        for dataset_entry in train_dataset:
            if piecewise:
                feed_forward_output = feed_forward_function(neural_network, dataset_entry, piecewise)
            else:
                feed_forward_output = feed_forward_function(neural_network, dataset_entry)
            expected = [dataset_entry[-1]]
            error += (expected[0] - feed_forward_output[0]) ** 2

            if piecewise:
                back_propagation_function(neural_network, expected, piecewise)
            else:
                back_propagation_function(neural_network, expected)

            redefine_weights(neural_network, dataset_entry, learning_rate)
        mse.append(error/len(train_dataset))

        predictions = []
        for dataset_entry in test_dataset:
            prediction = test_neural_network(neural_network, dataset_entry, feed_forward_function, piecewise)
            predictions.append(prediction)

        actual = [row[-1] for row in test_dataset]
        accuracy = accuracy_metrics(actual, predictions)
        accuracies.append(accuracy)

    return x_value, mse, accuracies


def test_neural_network(neural_network, input_vector, feed_forward_function, piecewise=None):
    """ Test neural network.

    Parameters
    ----------
    neural_network : dict
        Neural Network.
    feed_forward_function : function
        feed forward function name
    piecewise : list
        piecewise equations

    """
    if piecewise:
        result = feed_forward_function(neural_network, input_vector, piecewise)
    else:
        result = feed_forward_function(neural_network, input_vector)
    if result[0] >= 0.50:
        return 1
    else:
        return 0


def run_code(filename):
    """ Train, Test and evaluate neural network on a given dataset. """
    data = load_csv(filename)
    dataset = replace_missing_data_with_mean(data)
    str_column_to_int(dataset, len(dataset[0]) - 1)
    min_and_max_values = get_min_max_dataset_values(dataset)
    normalize_data(dataset, min_and_max_values)

    learning_rate = 0.08
    iterations = 25
    splits = 2

    batches = cross_validation_batch(dataset, splits)

    number_of_inputs = len(dataset[0]) - 1
    number_of_outputs = 1
    number_of_hiddens = 2 * number_of_inputs

    neural_network = create_neural_network(number_of_inputs, number_of_outputs, number_of_hiddens)

    # Sigmoid
    print("Using Sigmoid :- ")
    total = []
    for batch in batches:
        train_set = list(batches)
        train_set.remove(batch)
        train_set = sum(train_set, [])
        test_set = []
        for row in batch:
            row_copy = list(row)
            test_set.append(row_copy)

        network = deepcopy(neural_network)
        x_values, mse, accuracies = train_neural_network(network, train_set, test_set, learning_rate, iterations,
                                                         feed_forward_propagation, back_propagation)

        plot_graph('Accuracy graph for Sigmoid \nBatch - ' + str(batches.index(batch) + 1), 'Iterations', 'Accuracies',
                   x_values, accuracies)

        plot_graph('MSE graph for Sigmoid \nBatch - ' + str(batches.index(batch) + 1), 'Iterations', 'MSE', x_values, mse)

        predictions = []
        for row in test_set:
            prediction = test_neural_network(network, row, feed_forward_propagation)
            predictions.append(prediction)

        actuals = [row[-1] for row in test_set]
        accuracy = accuracy_metrics(actuals, predictions)
        total.append(accuracy)
    print('Accuracies : %s' % total)
    print('Mean Accuracy sigmoid: %.3f%%' % (sum(total) / float(len(total))))

    # Piecewise
    print("Using Piecewise Function Approximation:- ")
    segments = [4, 6, 8, 10, 12, 14]
    for segment in segments:
        total = []
        for batch in batches:
            train_set = list(batches)
            train_set.remove(batch)
            train_set = sum(train_set, [])
            test_set = []
            for row in batch:
                row_copy = list(row)
                test_set.append(row_copy)

            network_piecewise = deepcopy(neural_network)
            x_values, mse, accuracies = train_neural_network(network_piecewise, train_set, test_set, learning_rate,
                                                             iterations, feed_forward_propagation_with_piecewise,
                                                             back_propagation_with_piecewise,
                                                             get_piecewise_equations(segment))

            plot_graph('Accuracy graph for ' + str(segment) + ' Linear Piecewise Segments' + '\nBatch - ' +
                       str(batches.index(batch) + 1), 'Iterations', 'Accuracies', x_values, accuracies)

            plot_graph('MSE graph for ' + str(segment) + ' Linear Piecewise Segments' + ' \nBatch - ' +
                       str(batches.index(batch) + 1), 'Iterations', 'MSE', x_values, mse)

            predictions = []
            for row in test_set:
                prediction = test_neural_network(network_piecewise, row, feed_forward_propagation_with_piecewise,
                                                 get_piecewise_equations(segment))
                predictions.append(prediction)

            actuals = [row[-1] for row in test_set]
            accuracy = accuracy_metrics(actuals, predictions)
            total.append(accuracy)
            print("Accuracy with %d piecewise segments for batch %d = %.3f%%" %
                  (segment, (batches.index(batch) + 1), (accuracy)))

        print("Number of piecewise linear segment - ", segment)
        print('Accuracies : %s' % total)
        print('Mean Accuracy with %d linear segments : %.3f%%' % (segment, (sum(total) / float(len(total)))))


# Execute code
run_code("bank_dataset.csv")
