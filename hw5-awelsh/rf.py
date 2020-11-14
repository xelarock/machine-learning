import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import matplotlib.pyplot as plt
import sys

class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    forest = None

    class ExtendedTree(object):                     # create a class that has a classifier, selected features,
        clf = DecisionTreeClassifier()              # oob samples and trained samples
        selected_features = []
        oob_samples = []
        trained_samples = []

        def __init__(self, clf):
            self.clf = clf


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat
        self.forest = [         # initialize all the decision trees
            self.ExtendedTree(DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth,
                                                     min_samples_leaf=self.minLeafSample))
            for _ in range(self.nest)
        ]

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        # for each tree, pick the random columns and rows and store the oob rows
        # fit the classifier to the selected rows for training.
        for each_tree in self.forest:
            sample_rows_indexes = np.random.choice(xFeat.shape[0], size=xFeat.shape[0], replace=True)
            sample_cols_indexes = np.random.choice(xFeat.shape[1], size=self.maxFeat, replace=False)
            oob_row_indexes = np.array([i for i in range(xFeat.shape[0]) if i not in sample_rows_indexes])
            each_tree.clf.fit(xFeat[sample_rows_indexes[:, None], sample_cols_indexes], y[sample_rows_indexes, :])
            each_tree.oob_samples = oob_row_indexes
            each_tree.selected_features = sample_cols_indexes
            each_tree.trained_samples = sample_rows_indexes

        oob_error = []
        # for each tree, have every other tree predict the current tree's oob samples and compare to current tree preds
        for i, curr_tree in enumerate(self.forest):
            oob_rows = xFeat[curr_tree.oob_samples, :]
            other_tree_predictions = np.zeros((len(xFeat), self.nest))
            # for every other tree
            for j, other_tree in enumerate(self.forest):
                if i != j:
                    # get the set difference of the oob samples and the other tree's trained samples so the other trees
                    # only predict the oob samples it wasn't trained on
                    predict_oob_samples = np.setdiff1d(curr_tree.oob_samples, other_tree.trained_samples)
                    prediction = other_tree.clf.predict(
                        xFeat[predict_oob_samples[:, None], other_tree.selected_features])
                    other_tree_predictions[predict_oob_samples, j] = prediction
            # get the most common classification of the other trees for the current tree's oob samples
            other_tree_predictions[other_tree_predictions == 0] = np.nan
            other_tree_consensus = mode(other_tree_predictions, axis=1, nan_policy="omit")
            other_tree_consensus = np.take(other_tree_consensus[0], curr_tree.oob_samples)
            other_tree_consensus[other_tree_consensus == 0] = 1
            # get the current tree's predictions for oob sample.
            current_tree_prediction = curr_tree.clf.predict(xFeat[curr_tree.oob_samples[:, None], curr_tree.selected_features])
            # calculate the error
            oob_error.append(np.sum(other_tree_consensus != current_tree_prediction) / oob_rows.shape[0])
        return oob_error

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
            Predicted response per sample
        """
        yHat = []
        # for each tree in forest, get their prediction and then get the consensus of their predictions
        for curr_tree in self.forest:
            yHat.append(curr_tree.clf.predict(xFeat[:, curr_tree.selected_features]))
        yHat = mode(np.stack(yHat), axis=0, nan_policy="omit")[0]
        return yHat.T


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    yTrain[yTrain == 0] = -1            # change 0 labels to -1
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)
    yTest[yTest == 0] = -1              # change 0 labels to -1

    np.random.seed(args.seed)
    # nest, maxFeat, criterion, maxDepth, minLeafSample
    model = RandomForest(15, 6, "entropy", 9, 2)                # optimal parameters for q2c.
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    print("Averge OOB error:", np.average(trainStats))
    yHat = model.predict(xTest)
    error = np.sum(yHat != yTest) / yTest.shape[0]
    print("Test Error:", error)

    # optimal parameter error
    # Averge OOB error: 0.12736929913732847
    # Test Error: 0.08958333333333333

    # Code for the hyperparameter search

    # nest = [2, 5, 7, 10, 15, 20, 25, 30]
    # max_feat = [3, 4, 5, 6, 7, 8, 9]
    #
    # column_names = ['nest', 'max_feat', 'class_error']
    # df = pd.DataFrame(columns=column_names)
    #
    # for nest_val in nest:
    #     for feat in max_feat:
    #         print("Running nest val:", nest_val, "max feat:", feat)
    #         model = RandomForest(nest_val, feat, "entropy", 4, 8)
    #         trainStats = model.train(xTrain, yTrain)
    #         yHat = model.predict(xTest)
    #         error = np.sum(yHat != yTest) / yTest.shape[0]
    #         new_row = pd.Series([nest_val, feat, error], index=column_names)
    #         df = df.append(new_row, ignore_index=True)
    #
    # print(df)   # nest = 15, max_feat = 6, classification error = 0.114583
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(df['nest'], df['max_feat'], df['class_error'], c='b')
    # ax.set_xlabel("nest")
    # ax.set_ylabel("max feat")
    # ax.set_zlabel("classification error")
    #
    # column_names = ['mls', 'max_depth', 'class_error']
    # df = pd.DataFrame(columns=column_names)
    #
    # mls = [2, 5, 7, 10, 15, 20, 25, 30]
    # max_depth = [3, 4, 5, 6, 7, 8, 9]
    #
    # for mls_val in mls:
    #     for max_depth_val in max_depth:
    #         print("Running mls val:", mls_val, "max depth:", max_depth_val)
    #         model = RandomForest(15, 6, "entropy", max_depth_val, mls_val)
    #         trainStats = model.train(xTrain, yTrain)
    #         yHat = model.predict(xTest)
    #         error = np.sum(yHat != yTest) / yTest.shape[0]
    #         new_row = pd.Series([mls_val, max_depth_val, error], index=column_names)
    #         df = df.append(new_row, ignore_index=True)
    #
    # print(df)   # mls 2, max_depth = 9, classification error = 0.104167
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(df['mls'], df['max_depth'], df['class_error'], c='b')
    # ax1.set_xlabel("mls")
    # ax1.set_ylabel("max depth")
    # ax1.set_zlabel("classification error")
    #
    # plt.show()

if __name__ == "__main__":
    main()