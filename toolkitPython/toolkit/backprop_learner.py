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
        self.lr = 0.25
        self.momentum = .9
        self.labels = []
        self.velocity = []
        self.network = []
        self.dataset = []
        self.bssf = None
        self.bssf_validation_mse = np.inf
        self.usingMomentum = True

        # used for specific questions
        self.q3Metrics = []
        self.q4Metrics = []
        self.currQuestion = 3
        self.lr_set = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
        self.hn_set = [1, 2, 4, 8, 16, 32, 64]
        self.momentum_set = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


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
        # while len(inputs) < len(self.network[0]['weights']):
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
            errors = []
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
            for row in self.train_set:
                outputs = self.forward_propagate(row)
                expected = [0] * n_outputs
                expected[int(row[-1])] = 1
                train_sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            train_mse = train_sum_error / len(self.train_set)
            validation_mse = self.getMSE(self.validation_set)

            # metrics only used of question 2
            if self.currQuestion == 2:
                validation_accuracy = self.getAccuracy(self.validation_set)
                train_mses.append(train_mse)
                validation_mses.append(validation_mse)
                validation_accuracies.append(validation_accuracy)

            # check for new best solution so far (bssf)
            if validation_mse < self.bssf_validation_mse and abs(validation_mse - self.bssf_validation_mse) > 0.001:
                epochs_since_improvement = 0
                self.bssf_validation_mse = validation_mse
                # deep copy the current best network
                self.bssf = [row[:] for row in self.network]
            else:
                epochs_since_improvement +=1

            
            # check if no improvement reached among the last 5 most recent validation_mse values 
            if epochs_since_improvement >= 5:
                if self.currQuestion == 2:
                    self.plotQ2(train_mses, validation_mses, validation_accuracies)
                elif self.currQuestion == 3:
                    test_mse = self.getMSE(self.test_set)
                    self.q3Metrics.append({
                        "lr" : l_rate,
                        "epochs": epoch,
                        "xcoords" : [test_mse, train_mse, validation_mse],
                        "ycoords" : [l_rate] * 3})
                elif self.currQuestion == 4:
                    test_mse = self.getMSE(self.test_set)
                    self.q4Metrics.append({
                        "xcoords": [self.n_hidden] * 3,
                        "ycoords": [test_mse, train_mse, validation_mse],
                    })
                print("No more VS MSE improvement!")
                break
            
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, validation_mse))
    
    def getMSE(self, dataset):
        sum_error = 0
        for row in dataset:
            outputs = self.forward_propagate(row)
            expected = [0] * self.n_outputs
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        return sum_error / len(dataset)
    # plotting for question 2 of the report:
    #   1) train_mse vals, validation_mse vals along y axis
    #       - epoch num along x axis
    #   2) validation set accuracy along y axis
    #       - epoch num along x axis
    def plotQ2(self, train_mses, validation_mses, validation_accuracies):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(train_mses, linestyle='--', marker='o', color='b', label="Train Set MSE")
        plt.plot(validation_mses, linestyle='--', marker='o', color='r', label="Validation Set MSE")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Sum Error")
        plt.legend()

        plt.subplot(212)
        plt.plot(validation_accuracies, linestyle='--', marker='o', color='b', label="Validation Set Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Classification Accuracy")
        plt.legend()
        plt.show()

    # plotting for question 3 of the report:
    #   1) train_mse val, validation_mse val, test_mse val at stopping point along x axis
    #       - given learning rate along y axis
    #   2) number of epochs needed to reach best VS solution along y axis
    #       - given learning rate along x axis
    def plotQ3(self):
        print(self.q3Metrics)
        for result in self.q3Metrics:
            plt.scatter(result['xcoords'], result['ycoords'], c=('r','b','y'))
        plt.xlim(xmin=0)
        plt.xlabel("MSE - Color Coded")
        plt.ylabel("Learning Rate")
        plt.title("MSE by Learning Rate (Red: Test, Blue: Train, Yellow: Validation)")
        plt.show()
        
        for result in self.q3Metrics:
            plt.scatter(result['lr'], result['epochs'])
        plt.xlabel("Learning Rate")
        plt.ylabel("Number of epochs until convergence")
        plt.title("Learning Rate vs Learning Time (Epoch Count at convergence)")
        plt.show()

    # plotting for question 4 and 5 of the report:
    # DONE IN EXCEL SIMPLY

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
        # Prep the vowels dataset
        self.dataset = np.hstack((features.data, labels.data))
        # Remove the 'Test or Train' and Sex characteristics from the vowels dataset
        self.dataset = np.delete(self.dataset, [0,1,2], 1).tolist()
        shuffle(self.dataset)
        

        # split dataset into training, validation and test sets
        train_ratio, validation_ratio= .60, .15 # test_ratio = .25 = (1-rest)
        train_start_idx = math.floor((len(self.dataset)-1)*(train_ratio))
        test_start_idx =  math.ceil((len(self.dataset)-1) - ((len(self.dataset)-1)*(1 - (train_ratio+validation_ratio))))
        self.train_set, self.validation_set, self.test_set = self.dataset[train_start_idx:], self.dataset[train_start_idx:test_start_idx], self.dataset[test_start_idx:]
        
        # n_inputs: number of obs * 2 by default
        # n_outputs: set of possible values (2 for T/F, 0/1)
        self.n_inputs = len(self.train_set[0]) - 1
        self.n_hidden = self.n_inputs * 2
        self.n_outputs = len(set([row[-1] for row in self.train_set]))
        self.velocity = [0] * self.n_hidden

        if self.currQuestion == 1 or self.currQuestion == 2:
            self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
            self.train_network(self.lr, 500, self.n_outputs)
        elif self.currQuestion == 3:
            for lr in self.lr_set:
                self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
                self.train_network(lr, 500, self.n_outputs)
                
                # Reset features for next iteration
                self.velocity = [0] * self.n_hidden
                self.network = []
                self.bssf = None
                self.bssf_validation_mse = np.inf
            self.plotQ3()
        elif self.currQuestion == 4:
            for hidden_count in self.hn_set:
                # self.n_hidden = hidden_count
                self.n_hidden = self.n_inputs * 2
                self.velocity = [0] * self.n_hidden

                self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
                self.train_network(self.lr, 500, self.n_outputs)
                print(self.n_hidden)

                # Reset features for next iteration if not last
                if hidden_count != self.hn_set[-1]:
                    self.velocity = [0] * self.n_hidden
                    self.network = []
                    self.bssf = None
                    self.bssf_validation_mse = np.inf
            # self.plotQ4(), just do it in excel
        elif self.currQuestion == 5:
            for momentum_val in self.momentum_set:
                self.momentum = momentum_val
                print(self.momentum)
                self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
                self.train_network(self.lr, 500, self.n_outputs)
                print(self.n_hidden)

                # Reset features for next iteration if not last
                if momentum_val != self.momentum_set[-1]:
                    self.velocity = [0] * self.n_hidden
                    self.network = []
                    self.bssf = None
                    self.bssf_validation_mse = np.inf
        elif self.currQuestion == 6:
            core_dataset = np.array(self.dataset)
            for col in range(9):
                print("Removing Col: %s" % col)
                self.dataset = np.delete(self.dataset, col, 1).tolist()
                self.initialize_network(self.n_inputs, self.n_hidden, self.n_outputs)
                self.train_network(self.lr, 500, self.n_outputs)

                print(self.getAccuracy(self.dataset))
                # Reset features for next iteration
                if col != 10:
                    self.network = []
                    self.bssf = None
                    self.bssf_validation_mse = np.inf
                    self.dataset = core_dataset
        else:
            print("Enter a valid question number, refer to the documentation")

    # provided one observation, reset the label to the predicted value
    def predict(self, observation, label):
        # Remove the 'Test or Train' and Sex characteristics to conform to model for vowels dataset
        observation = observation[2:]
        prediction = self.predictVal(observation)
        del label[:]
        label += [prediction]

   