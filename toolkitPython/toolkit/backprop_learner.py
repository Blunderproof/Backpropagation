from .supervised_learner import SupervisedLearner
import numpy as np
from random import random
from random import seed
from random import randrange
from collections import OrderedDict
import matplotlib.pyplot as plt

class BackpropLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """
    def __init__(self):
        # Learning Rate
        self.lr = 0.025
        self.n_hidden = 1
        self.momentum = 0.9
        self.labels = []
        self.velocity = []
        self.network = []
        self.dataset = []
        self.usingMomentum = True

    # n_inputs: 1 for each col in dataset, #n_hidden: user-set variable
    # n_outputs: length of the set of possible outputs (2 for T/F)
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        # TODO: Maybe change the final variable to a 1, as we've done in this class
        hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    # Calculate neuron activation for an input
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation
    
    # Transfer neuron activation
    def transfer(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    # Forward propagate input to a network output
    def forward_propagate(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    # Calculate the derivative of an neuron output
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])
    
    # Update network weights with error
    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    if(self.usingMomentum == True):  # NEW METHOD OF UPDATING WEIGHTS WITH MOMENTUM
                        self.velocity[j] = l_rate * neuron['delta'] * inputs[j] + self.momentum * self.velocity[j]
                        neuron['weights'][j] += self.velocity[j]
                    else:  # OLD METHOD OF UPDATING WEIGHTS
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                if(self.usingMomentum==True):
                    self.velocity[-1] =  l_rate * neuron['delta'] + self.momentum * self.velocity[j]
                    neuron['weights'][-1] += self.velocity[-1]
                else:
                    neuron['weights'][-1] += l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train_network(self, train_set, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train_set:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(n_outputs)]
                expected[int(row[-1])] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    
    def predictVal(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        seed(1)
        self.dataset = np.hstack((features.data, labels.data)).tolist()
        self.velocity = [0] * (len(self.dataset[0])-1)
        # n_inputs: number of obs
        # n_outputs: set of possible values (2 for T/F, 0/1)
        n_inputs = len(self.dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in self.dataset]))
        self.initialize_network(n_inputs, self.n_hidden, n_outputs)
        self.train_network(self.dataset, self.lr, 500, n_outputs)

    # provides one observation, expects label to be updated with the predicted value
    def predict(self, observation, label):
        """
        :type observation: [float]
        :type label: [float]
        """
        prediction = self.predictVal(observation)
        del label[:]
        label += [prediction]

   