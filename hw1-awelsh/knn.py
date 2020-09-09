# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import argparse
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class Knn(object):
    k = 0    # number of neighbors to use
    trainSet = []   # contains the classification, features, and distance
    xTrain = []     # contains the features
    yTrain = []     # contains the classifications

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
        if type(xFeat) != np.ndarray:               # if the data isn't a numpy array, eg dataframe, convert to numpy
            self.xTrain = xFeat.to_numpy()
        else:
            self.xTrain = xFeat
        if type(y) != np.ndarray:
            self.yTrain = y.to_numpy()
        else:
            self.yTrain = y
        # set the train set columns to [1 x classification, d x features, 1 x placeholder for distances]
        self.trainSet = np.column_stack((np.atleast_1d(self.yTrain), self.xTrain, np.empty(len(self.xTrain))))
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
        if type(xFeat) != np.ndarray:           # if the data isn't a numpy array, eg dataframe, convert to numpy
            xFeat = xFeat.to_numpy()
        yHat = []  # variable to store the estimated class label
        j = 0
        for xrow in xFeat:                      # for each row of test data
            i = 0
            for row in self.xTrain:             # iterate through each row of training data
                sums = 0
                for index in range(len(xrow)):  # then iterate through each feature in a row to calculate mikowski dist
                    sums += math.pow(abs(xrow[index] - row[index]), len(xrow))
                self.trainSet[i][len(self.trainSet[0]) - 1] = sums  # set the last column (placeholder) to distance
                i += 1
            output = self.trainSet[np.argsort(self.trainSet[:, len(self.trainSet[0]) - 1])] # sort by smallest distance
            voting = 0
            for sampleIndex in range(self.k):   # get the k smallest distances and their classifications
                if output[sampleIndex][0] == 1.0:   # if equal to one, add one to vote
                    voting += 1
                else:                               # otherwise, subtract one from vote
                    voting -= 1
            if voting >= 0:                     # if the final vote is positive or 0, then classify test sample as one
                yHat.append([1])
            else:
                yHat.append([0])                # otherwise classify test sample as 0
            j += 1
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
    acc = 0
    for i in range(len(yHat)):          # count the number of correct classifications
        if yHat[i] == yTrue[i]:
            acc += 1
    return acc / len(yTrue)             # return the num correct / total test samples


# used to iterate from 1 to k to generate figure comparing different values of k.
def performance(xTrain, yTrain, xTest, yTest, k):
    acc = np.empty([k, 2])
    for i in range(1, k+1):             # for i=1 to k
        knn = Knn(i)                    # run the KNN classifier and record the accuracy for test and training data
        knn.train(xTrain, yTrain['label'])
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])
        acc[i - 1][0] = trainAcc
        acc[i - 1][1] = testAcc

    # set up plot to print the accuracy of test and training data for varying values of k
    plt.title("Training and Testing Accuracy for K-Nearest Neighbors Algorithm")
    plt.xlabel("K value")
    plt.ylabel("Percent accurate")
    plt.plot([i for i in range(1, k+1)], acc)       # start plot from 1, not 0 (default)
    plt.legend(("Training Accuracy", "Testing Accuracy"))
    plt.show()


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
    #create an instance of the model
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

    # runs the KNN from 1 to K to compare the accuracy for different values of K.
    performance(xTrain, yTrain, xTest, yTest, args.k)


if __name__ == "__main__":
    main()
