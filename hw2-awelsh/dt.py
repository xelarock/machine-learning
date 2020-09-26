import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math
import matplotlib.pyplot as plt


class Node:                                                                                 # create Node class
    left = None
    right = None
    xFeatures = None
    yClass = None
    guess = None
    featureName = None
    featureValue = None

    def __init__(self, xFeatures, yClass, left, right, guess, feature_name, feature_value):         # initialize node
        self.left = left
        self.right = right
        self.xFeatures = xFeatures
        self.yClass = yClass
        self.guess = guess
        self.featureName = feature_name
        self.featureValue = feature_value

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    tree = None
    depth = 0

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.tree = None

    def splitData(self, xFeat, y, split_feat, split_value):         # splits data set along a certain value in a feature
        mergedSet = pd.concat([xFeat, y], axis=1)

        leftX = mergedSet[mergedSet[split_feat] <= split_value]     # left set is all values less than or equal to split
        leftY = leftX.iloc[:, -1]
        leftX = leftX.drop(labels='label', axis=1)

        rightX = mergedSet[mergedSet[split_feat] > split_value]     # right set is all values greater than to split
        rightY = rightX.iloc[:, -1]
        rightX = rightX.drop(labels='label', axis=1)

        return leftX, leftY, rightX, rightY                         # return split data

    def calculate_gini(self, y):                                    # calculates the gini index
        gini = 0
        total = len(y)
        for i in range(len(y.value_counts().values)):               # for each class in the data
            gini += (y.value_counts().values[i]/total) ** 2         # sum up the fraction squared
        return 1 - gini

    def calculate_entropy(self, y):                                 # calculates entropy
        entropy = 0
        total = len(y)
        for i in range(len(y.value_counts().values)):               # for each class in the data
            val = math.log2((y.value_counts().values[i]/total))     # do log2 of the fraction
            entropy += val
        return -1 * entropy                                         # return the negative

    def grow_tree(self, xFeat, y, depth):                           # builds the decision tree and keeps track of depth
        guess = y.value_counts().idxmax()                           # guess is most common label in set

        if len(y.value_counts()) == 1:  # if only one label value (unambiguous)
            return Node(xFeat, y, None, None, guess, None, None)
        elif len(xFeat) < self.minLeafSample:  # if remaining features is empty
            return Node(xFeat, y, None, None, guess, None, None)
        elif depth >= self.maxDepth: # if depth is greater than max depth
            return Node(xFeat, y, None, None, guess, None, None)
        else:
            lowest_val = 0                                          # set values to find best split
            lowest_gini = 999
            lowest_entropy = 999
            lowest_featName = None
            for featName, feature in xFeat.iteritems():             # for each feature
                for value in xFeat[featName].unique():              # for each unique value in that feature
                    leftX, leftY, rightX, rightY = self.splitData(xFeat, y, featName, value)    # split by that value
                    if self.criterion == 'gini':                    # if criteria is gini
                        left_gini = self.calculate_gini(leftY)      # calculate left and right child gini indexes
                        right_gini = self.calculate_gini(rightY)
                        avg_gini = (left_gini * len(leftY) + right_gini * len(rightY)) / len(y)     # find the average
                        if avg_gini <= lowest_gini:                 # if its lower, than remember the lowest split info
                            lowest_gini, lowest_val, lowest_featName = avg_gini, value, featName
                    elif self.criterion == 'entropy':               # if criteria is entropy
                        left_entropy = self.calculate_entropy(leftY)    # calculate left and right
                        right_entropy = self.calculate_entropy(rightY)
                        entropy = (len(leftY)/len(y) * left_entropy) + (len(rightY)/len(y) * right_entropy) # get total
                        if entropy <= lowest_entropy:               # if less, than remember the lowest split info
                            lowest_val, lowest_featName, lowest_entropy = value, featName, entropy
            tree = Node(xFeat, y, None, None, y.value_counts().idxmax(), lowest_featName, lowest_val) # create the tree
            leftX, leftY, rightX, rightY = self.splitData(xFeat, y, lowest_featName, lowest_val)    # split along lowest
            tree.left = self.grow_tree(leftX, leftY, depth + 1)                                     # make left node
            tree.right = self.grow_tree(rightX, rightY, depth + 1)                                  # make right node
        return tree                                                 # recurse and return the final tree at the end


    def train(self, xFeat, y):
        """
        Train the decision tree model.

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

        self.tree = self.grow_tree(xFeat, y, 0)         # generate the tree
        return self

    def dtTest(self, tree, sample):
        if tree.right is None and tree.left is None:                # predict the value, if at a leaf node
            return tree.guess
        elif tree.right is not None and sample[tree.featureName] > tree.featureValue:   # if left node exist and
            return self.dtTest(tree.right, sample)                                      # sample value is in there, go
        elif tree.left is not None and sample[tree.featureName] <= tree.featureValue:   # if right node exist and
            return self.dtTest(tree.left, sample)                                       # sample value is in there, go
        return tree.guess

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

        yHat = [] # variable to store the estimated class label
        for index, sample in xFeat.iterrows():
            yHat.append(self.dtTest(self.tree, sample))             # predict the value
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
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
    # train the model
    dt.train(xTrain, yTrain['label'])

    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    """
    Below is code used to generate the 2 3D  plots (one for each criterion (gini/entropy)). Because I ran a lot of 
    models and was scared of losing my data, I ran chunks of parameters in q1.ipynb, saved the results to 
    "dt-model-accuracy.csv" and then imported that CSV file to generate the plots. See q1.ipynb for the code to generate
    models.
    """

    # dt_model_accuracy = pd.read_csv("dt-model-accuracy.csv")
    #
    # gini figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(dt_model_accuracy['max_depth'], dt_model_accuracy['min_leaf_sample'], dt_model_accuracy['gini_train'], c='b')
    # ax.set_xlabel("max depth")
    # ax.set_ylabel("min leaf sample")
    # ax.set_zlabel("gini train")
    # ax.scatter(dt_model_accuracy['max_depth'], dt_model_accuracy['min_leaf_sample'], dt_model_accuracy['gini_test'], c='r')
    # ax.set_xlabel("max depth")
    # ax.set_ylabel("min leaf sample")
    # ax.set_zlabel("gini accuracy")
    #
    # entropy figure
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(dt_model_accuracy['max_depth'], dt_model_accuracy['min_leaf_sample'], dt_model_accuracy['entropy_train'],
    #            c='b')
    # ax1.set_xlabel("max depth")
    # ax1.set_ylabel("min leaf sample")
    # ax1.set_zlabel("entropy accuracy")
    # ax1.scatter(dt_model_accuracy['max_depth'], dt_model_accuracy['min_leaf_sample'], dt_model_accuracy['entropy_test'],
    #            c='r')
    # ax1.set_xlabel("max depth")
    # ax1.set_ylabel("min leaf sample")
    # ax1.set_zlabel("entropy accuracy")
    # plt.show()


if __name__ == "__main__":
    main()
