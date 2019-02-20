from .supervised_learner import SupervisedLearner
import numpy as np
from random import random
from random import shuffle
from random import seed
from random import randrange
from collections import OrderedDict
import matplotlib.pyplot as plt
import math


class BackpropLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """
    def __init__(self):
        # Learning Rate
        self.lr = 0.1
        self.n_hidden = 1
        self.momentum = 0.9
        self.labels = []
        self.velocity = []
        self.network = []
        self.dataset = []
        self.bssf = None
        self.bssf_validation_mse = np.inf
        self.usingMomentum = True

    # n_inputs: 1 for each col in dataset, #n_hidden: user-set variable for num of nodes in the hidden layer
    # n_outputs: length of the set of possible outputs (2 for T/F)
    def initialize_network(self, n_inputs, n_hidden, n_outputs):
        hidden_layer = [{'weights':[random() for i in range(n_inputs)]} for i in range(n_hidden)]
        # Set the bias weight to 1
        for node in hidden_layer:
            node['weights'].append(1)
        self.network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(n_hidden)]} for i in range(n_outputs)]
        # Set the bias weight to 1
        for node in output_layer:
            node['weights'].append(1)
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

    # Train a network, stop when n_epochs is exhausted or no improvement seen in test_set after 5 epochs.
    def train_network(self, l_rate, n_epoch, n_outputs):
        validation_mses = []
        validation_accuracies = []
        train_mses = []
        epochs_since_improvement = 0

        for epoch in range(n_epoch):
            train_sum_error = 0
            validation_sum_error = 0
            for row in self.train_set:
                outputs = self.forward_propagate(row)
                expected = [0] * n_outputs
                expected[int(row[-1])] = 1
                train_sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            # Check the validation set's MSE
            for row in self.validation_set:
                outputs = self.forward_propagate(row)
                expected = [0] * n_outputs
                expected[int(row[-1])] = 1
                validation_sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            train_mse = train_sum_error / len(self.train_set)
            validation_mse = validation_sum_error / len(self.validation_set)
            validation_accuracy = self.getAccuracy(self.validation_set)
            # check for new best solution so far (bssf)
            if validation_mse < self.bssf_validation_mse:
                epochs_since_improvement = 0
                self.bssf_validation_mse = validation_mse
                # deep copy the current best network
                self.bssf = [row[:] for row in self.network]
            else:
                epochs_since_improvement +=1
            train_mses.append(train_mse)
            validation_mses.append(validation_mse)
            validation_accuracies.append(validation_accuracy)
            # check if no improvement reached among the last 5 most recent test_mse values 
            if epochs_since_improvement >= 5:
                self.plotQ2(train_mses, validation_mses, validation_accuracies)
                print("No more VS MSE improvement!")
                break
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, validation_mse))
    
    # plotting for question 3 of the report:
    #   1) train_mse vals, test_mse vals along y axis
    #       - epoch num along x axis
    #   2) test set accuracy along y axis
    #       - epoch num along x axis
    def plotQ2(self, train_mses, test_mses, test_accuracies):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(train_mses, linestyle='--', marker='o', color='b', label="Train Set MSE")
        plt.plot(test_mses, linestyle='--', marker='o', color='r', label="Validation Set MSE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Sum Error")
        plt.legend()

        plt.subplot(212)
        plt.plot(test_accuracies, linestyle='--', marker='o', color='b', label="Validation Set Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Classification Accuracy")
        plt.legend()
        plt.show()


    def getAccuracy(self, dataset):
        correct = 0
        # find predictions
        for obs in dataset:
            prediction = self.predictVal(obs)
            if prediction == obs[-1]:
                correct += 1
        return (correct / len(dataset))
 
    def predictVal(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    def train(self, features, labels):
        self.dataset = np.hstack((features.data, labels.data)).tolist()
        shuffle(self.dataset)

        # split dataset into training, validation and test sets
        train_ratio, validation_ratio, test_ratio = .60, .15, .25
        train_start_idx = math.floor((len(self.dataset)-1)*(train_ratio))
        test_start_idx =  math.ceil((len(self.dataset)-1) - ((len(self.dataset)-1)*(1 - (train_ratio+validation_ratio))))
        self.train_set, self.validation_set, self.test_set = self.dataset[train_start_idx:], self.dataset[train_start_idx:test_start_idx], self.dataset[test_start_idx:]
        
        # n_inputs: number of obs
        # n_outputs: set of possible values (2 for T/F, 0/1)
        self.n_inputs = len(self.train_set[0]) - 1
        self.n_hidden = self.n_inputs * 2
        self.n_outputs = len(set([row[-1] for row in self.train_set]))
        self.velocity = [0] * self.n_hidden

        self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
        self.train_network(self.lr, 5000, self.n_outputs)

    # provided one observation, reset the label to the predicted value
    def predict(self, observation, label):
        prediction = self.predictVal(observation)
        del label[:]
        label += [prediction]

   