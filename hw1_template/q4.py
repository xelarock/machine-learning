# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from knn import Knn
from knn import accuracy
import matplotlib.pyplot as plt


def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance 
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """

    stdsScale = StandardScaler()                                    # use the standard scaler to scale data
    stdsScale.fit(xTrain)
    return stdsScale.transform(xTrain), stdsScale.transform(xTest)


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.T he same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    minmaxScaler = MinMaxScaler()                                   # use the minmax scaler to scale data
    minmaxScaler.fit(xTrain)
    return minmaxScaler.transform(xTrain), minmaxScaler.transform(xTest)


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    xTest : nd-array with shape m x d
        Test data 

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    irr1 = np.random.normal(0, 1, len(xTrain.index))            # create 2 set of random values from a normal
    irr2 = np.random.normal(0, 1, len(xTrain.index))            # distribution with mean of 0 and std of 1.
    xTrain["irr1"] = irr1                                       # add to training data
    xTrain["irr2"] = irr2

    irr1 = np.random.normal(0, 1, len(xTest.index))             # create 2 set of random values from a normal
    irr2 = np.random.normal(0, 1, len(xTest.index))             # distribution with mean of 0 and std of 1.
    xTest["irr1"] = irr1                                        # add to training data
    xTest["irr2"] = irr2
    return xTrain, xTest


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    model = Knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    return accuracy(yHatTest, yTest['label'])
    

def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    # # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)
    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)
    print("Test Acc (with irrelevant feature):", acc4)

    # runs the KNN classifier with the three pre-processing options for K=1 to K.
    results = np.empty([args.k, 4]) # test accuracy is recorded in this K x 4 array

    for i in range(1, args.k + 1):      # loop through values of K from 1 to K
        # no preprocessing
        print("Starting K: ", i)
        results[i - 1][0] = knn_train_test(i, xTrain, yTrain, xTest, yTest)
        print("Test Acc (no-preprocessing):", results[i - 1][0])
        # preprocess the data using standardization scaling
        xTrainStd, xTestStd = standard_scale(xTrain, xTest)
        results[i - 1][1] = knn_train_test(i, xTrainStd, yTrain, xTestStd, yTest)
        print("Test Acc (standard scale):", results[i - 1][1])
        # preprocess the data using min max scaling
        xTrainMM, xTestMM = minmax_range(xTrain, xTest)
        results[i - 1][2] = knn_train_test(i, xTrainMM, yTrain, xTestMM, yTest)
        print("Test Acc (min max scale):", results[i - 1][2])
        # add irrelevant features
        xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
        results[i - 1][3] = knn_train_test(i, xTrainIrr, yTrain, yTrainIrr, yTest)
        print("Test Acc (with irrelevant feature):", results[i - 1][3])
        print(results)

    # set up figure to display results of pre-processing with different values of K
    plt.title("Training and Testing Accuracy for K-Nearest Neighbors Algorithm")
    plt.xlabel("K value")
    plt.ylabel("Percent accurate")
    plt.plot([i for i in range(1, args.k + 1)], results)    # start plotting from 1 (default 0)
    plt.legend(("No Pre-processing", "Standard Scaling", "Min-Max Scaling", "Irrelevant Features"))
    plt.show()

if __name__ == "__main__":
    main()
