# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import argparse
import numpy as np
import pandas as pd
import math


class Knn(object):
    k = 0    # number of neighbors to use
    trainSet = []
    xTrain = []
    yTrain = []
    dist = []

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.xTrain = xFeat.to_numpy()
        self.yTrain = y.to_numpy()
        # print()
        col4 = np.zeros(len(self.xTrain))
        #print(col4)
        self.trainSet = np.column_stack((np.atleast_1d(self.yTrain), self.xTrain, np.empty(len(self.xTrain))))
        #print(self.trainSet)
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        # dist = np.empty([1, len(self.trainSet)])
        xFeat = xFeat.to_numpy()
        # print("Feature ", xFeat[0][0], xFeat[0][1])
        yHat = []  # variable to store the estimated class label
        j = 0
        for x1, x2 in xFeat:
            i = 0
            # print(x1, x2)
            for row in self.xTrain:
                # print(x1, x2, row[0], row[1])
                # print("squared ", math.pow((x1 - x2), 2), math.pow((row[0] - row[1]), 2))
                # print("added: ", math.pow((x1 - x2), 2) + math.pow((row[0] - row[1]), 2))
                # print(math.sqrt(math.pow((x1 - row[0]), 2) + math.pow((x2 - row[1]), 2)))
                self.trainSet[i][3] = math.sqrt(math.pow((x1 - row[0]), 2) + math.pow((x2 - row[1]), 2))
                i += 1
                # break
            # print(self.trainSet)
            output = self.trainSet[np.argsort(self.trainSet[:, 3])]
            # print(self.trainSet)
            voting = 0
            # break
            for sampleIndex in range(self.k):
                # print("Sample ", sampleIndex, ": ", output[sampleIndex][0], " ", output[sampleIndex][3])
                if output[sampleIndex][0] == 1.0:
                    voting += 1
                else:
                    voting -= 1
            if voting >= 0:
                # print("append 1")
                yHat.append([1])
            else:
                yHat.append([0])
                # print("append 0")
            j += 1
            # break
            #print()
        # print(dist)
        # TODO
        # print(yHat)
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    # print(yTrue)
    acc = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            acc += 1
    return acc / len(yTrue)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
