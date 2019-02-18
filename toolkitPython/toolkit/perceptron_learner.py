from .supervised_learner import SupervisedLearner
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

class PerceptronLearner(SupervisedLearner):
    def __init__(self):
        self.weights = []
        self.learningRate = 0.1
        pass

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Matrix
        :type labels: Matrix
        """
        #print the dataset
        def printDataset():
            print("FEATURES:")
            for rowIdx in range(features.rows):
                print(features.row(rowIdx))
            print("LABELS:")
            for rowIdx in range(labels.rows):
                print(labels.row(rowIdx))

        # append the default bias of 1 to the feature rows
        for rowIdx in range(features.rows):
            row = features.row(rowIdx)
            row.append(1)

        # set the initial weights
        self.weights = [1] * (features.cols + 1)
        
        # code to run a single epoch
        def runEpoch(currWeights, learningRate):
            for rowIdx in range(features.rows):
                row = np.asarray(features.row(rowIdx))
                # get the net via dot product
                net = np.dot(row, currWeights)
                if net > 0:
                    output = 1
                else:
                    output = 0
                deltaW = []

                deltaW =  (learningRate * (labels.get(rowIdx, 0) - output)) * row
                currWeights += deltaW
            return currWeights

        # run epochs until the change between them falls in a certain range
        def runEpochs(maxVariance=0.00001, maxNumEpochs=20, returnBestFoundWeights=False, learningRate=0.1):
            print("\nLearning Rate: ", learningRate)
            # a map of index to [accuracy, weights]
            epochAccuracies = []
            epochWeights = []
            convergedEarly = False
            maxReached = 0
            initAccuracy = self.measure_accuracy(features, labels)
            for idx in range(maxNumEpochs): 
                self.weights = runEpoch(self.weights, learningRate)
                currAccuracy = self.measure_accuracy(features, labels)

                epochAccuracies.append(currAccuracy)
                epochWeights.append(list(self.weights))
                
                if returnBestFoundWeights == False:
                    if idx > 5:
                        mostRecentAccuracies = epochAccuracies[-5:]
                        accuracyVariance = np.var(mostRecentAccuracies)
                        if accuracyVariance < maxVariance:
                            print("The most recent 5 epochs had an accuracy variance of", accuracyVariance)
                            print("Epoch Count:", idx)
                            print("Terminating...")
                            print(mostRecentAccuracies)
                            convergedEarly = True
                            maxReached = idx
                            break
            
            # if not convergedEarly:
            #     maxReached = maxNumEpochs
            # # graph the different between the first and last epoch accuracy
            # xvals = [[i+1, i+2] for i in range(maxReached-1)]
            # yvals = [[1-epochAccuracies[i], 1-epochAccuracies[i+1]] for i in range(maxReached-1)]
            # yvals.insert(0, [1-initAccuracy, 1-epochAccuracies[0]])
            # xvals.insert(0, [0,1])
            # print(yvals)
            # plt.plot(xvals, yvals, color="red")
            # plt.xlim(0,maxReached)
            # plt.title("Misclassification Rate")
            # plt.xlabel("Epoch Number")
            # plt.ylabel("Weights Accuracy")
            # plt.ylim(0,1)
            # plt.show()
            # See the report for how this additional function is set to work
            if returnBestFoundWeights == True:
                print("Final converged accuracy: ", epochAccuracies[-1])
                bestEpochIdx = epochAccuracies.index(max(epochAccuracies))
                print("Best found accuracy:", max(epochAccuracies))
                deltaInWeights = sum(np.abs(np.asarray(self.weights) - np.asarray(epochWeights[bestEpochIdx])))
                print("Delta in weights:", deltaInWeights)
                self.weights = epochWeights[bestEpochIdx]
                

        # Run through multiple generations of epochs for part 6 of the lab
        # currLearningRate = self.learningRate
        # for i in range(3):
        #     runEpochs(returnBestFoundWeights=True, learningRate=currLearningRate)
        #     currLearningRate = currLearningRate/5
        
        runEpochs()
        # #plot the data
        def plotDataset():
            # separate the results to their respective colors
            redx =[]; redy = []; bluex = []; bluey = []
            for rowIdx in range(features.rows):
                if labels.row(rowIdx)[0] == 0:
                    redx.append(features.row(rowIdx)[0])
                    redy.append(features.row(rowIdx)[1])
                else:
                    bluex.append(features.row(rowIdx)[0])
                    bluey.append(features.row(rowIdx)[1])
            plt.scatter(redx, redy, color="red")
            plt.scatter(bluex, bluey, color="blue")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.title("Nonlinearly Separable Dataset")
            left, right = plt.xlim()

            slopePoint1 = -(self.weights[0]*left+self.weights[2]/self.weights[1]) 
            slopePoint2 = -(self.weights[0]*right+self.weights[2]/self.weights[1])
            plt.plot([left,right], [slopePoint1,slopePoint2])

            plt.show()


    def predict(self, features, labels):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        if len(features) != len(self.weights):
            features.append(1)

        net = np.dot(features, self.weights)
        if net > 0:
            output = 1
        else:
            output = 0
        del labels[:]
        labels += [output]

    def measure_accuracy(self, features, labels, confusion=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: Matrix
        :type labels: Matrix
        :type confusion: Matrix
        :rtype float
        """

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.value_count(0)
        if label_values_count == 0:
            # label is continuous
            pred = []
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred[0] = 0.0       # make sure the prediction is not biased by a previous prediction
                self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta**2
            return math.sqrt(sse / features.rows)

        else:
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.predict(feat, prediction)
                pred = int(prediction[0])
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred)+1)
                if pred == targ:
                    correct_count += 1

            return correct_count / features.rows