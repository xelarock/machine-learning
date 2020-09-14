import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math


class Node:
    left = None
    right = None
    xFeatures = None
    yClass = None
    guess = None
    featureName = None
    featureValue = None

    def __init__(self, xFeatures, yClass, left, right, guess, feature_name, feature_value):
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

    def splitData(self, xFeat, y, split_feat, split_value):
        mergedSet = pd.concat([xFeat, y], axis=1)
        #print(mergedSet)
        leftX = None
        rightX = None

        leftX = mergedSet[mergedSet[split_feat] <= split_value]
        leftY = leftX.iloc[:, -1]
        leftX = leftX.drop(labels='label', axis=1)

        rightX = mergedSet[mergedSet[split_feat] > split_value]
        rightY = rightX.iloc[:, -1]
        rightX = rightX.drop(labels='label', axis=1)

        #print(leftX, leftY, rightX, rightY)

        return leftX, leftY, rightX, rightY

    def calculate_gini(self, y):
        gini = 0
        total = len(y)
        for i in range(len(y.value_counts().values)):
            gini += (y.value_counts().values[i]/total) ** 2
        return 1 - gini

    def calculate_entropy(self, y):
        entropy = 0
        total = len(y)
        # print("number of somethin: ", len(y.value_counts().values))
        for i in range(len(y.value_counts().values)):
            val = (y.value_counts().values[i]/total) * math.log2((y.value_counts().values[i]/total))
            # print("val: ", val)
            entropy += val
        return -1 * entropy

    def grow_tree(self, xFeat, y, depth):
        # print("depth: ", depth)

        guess = y.value_counts().idxmax()

        if len(y.value_counts()) == 1:  # if only one label value (unambiguous)
            return Node(xFeat, y, None, None, guess, None, None)
        elif len(xFeat) < self.minLeafSample:  # if remaining features is empty
            return Node(xFeat, y, None, None, guess, None, None)
        elif depth >= self.maxDepth:
            return Node(xFeat, y, None, None, guess, None, None)
        else:
            lowest_val = 0
            lowest_gini = 999
            largest_gain = -999
            lowest_featName = None
            if self.criterion == 'entropy':
                root_entropy = self.calculate_entropy(y)
            for featName, feature in xFeat.iteritems():
                for value in xFeat[featName].unique():
                    # print("Feature: ", featName, " Value: ", value)
                    leftX, leftY, rightX, rightY = self.splitData(xFeat, y, featName, value)
                    # print("left: ", self.calculate_gini(leftY), " right: ", self.calculate_gini(rightY))
                    if self.criterion == 'gini':
                        left_gini = self.calculate_gini(leftY)
                        right_gini = self.calculate_gini(rightY)
                        avg_gini = (left_gini * len(leftY) + right_gini * len(rightY)) / len(y)
                        if avg_gini <= lowest_gini:
                            lowest_gini, lowest_val, lowest_featName = avg_gini, value, featName
                    elif self.criterion == 'entropy':
                        left_entropy = self.calculate_entropy(leftY)
                        right_entropy = self.calculate_entropy(rightY)
                        gain = root_entropy - (len(leftY)/len(y) * left_entropy) - (len(rightY)/len(y) * right_entropy)
                        if gain >= largest_gain:
                            lowest_val, lowest_featName, largest_gain = value, featName, gain
            # print("lowest gini: ", lowest_gini, "largest gain: ", largest_gain, lowest_val, lowest_index, lowest_featName)
            tree = Node(xFeat, y, None, None, y.value_counts().idxmax(), lowest_featName, lowest_val)
            leftX, leftY, rightX, rightY = self.splitData(xFeat, y, lowest_featName, lowest_val)
            tree.left = self.grow_tree(leftX, leftY, depth + 1)
            tree.right = self.grow_tree(rightX, rightY, depth + 1)
        return tree


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

        self.tree = self.grow_tree(xFeat, y, 0)

        #print(xFeat)
        #print(y)
        return self

    def dtTest(self, tree, sample):
        if tree.right is None and tree.left is None:
            return tree.guess
        elif tree.right is not None or tree.left is not None:
            if sample[tree.featureName] <= tree.featureValue:
                return self.dtTest(tree.left, sample)
            else:
                return self.dtTest(tree.right, sample)

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
            yHat.append(self.dtTest(self.tree, sample))
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




if __name__ == "__main__":
    main()
