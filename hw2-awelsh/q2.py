import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
import time
import matplotlib.pyplot as plt


def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    before = time.time()
    trainX, testX, trainY, testY = train_test_split(xFeat, y, shuffle=True, test_size=testSize)     # split data

    model.fit(trainX, trainY)                                                                       # fit data

    yHatTrain = model.predict_proba(trainX)                                                         # predict
    yHatTest = model.predict_proba(testX)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(trainY['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(testY['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)                                                                 # return the AUC
    after = time.time()
    timeElapsed = after - before
    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    mergedSet = pd.concat([xFeat, y], axis=1)                       # merge the data so that you can shuffle it
    mergedSet = shuffle(mergedSet)
    y = mergedSet.iloc[:, -1:]
    xFeat = mergedSet.iloc[:, 0:len(mergedSet.columns)-1]           # separate the features of labels again

    before = time.time()
    kf = KFold(n_splits=k)                                          # setup of k fold
    trainAuc = 0
    testAuc = 0
    for train_index, test_index in kf.split(xFeat):                 # for each fold
        trainX = xFeat.iloc[train_index]                            # set up train and test data
        testX = xFeat.iloc[test_index]
        trainY = y.iloc[train_index]
        testY = y.iloc[test_index]
        model.fit(trainX, trainY)                                   # fit to model

        yHatTrain = model.predict_proba(trainX)                     # predict
        yHatTest = model.predict_proba(testX)
        if len(yHatTrain[0]) == 1:                                  # if the predictions are all one class, add a
            yHatTrain = np.append(yHatTrain, np.zeros((len(yHatTrain), 1)), axis=1) # second column of zeros to work
        if len(yHatTest[0]) == 1:
            yHatTest = np.append(yHatTest, np.zeros((len(yHatTest), 1)), axis=1)
        # calculate auc for training
        fpr, tpr, thresholds = metrics.roc_curve(trainY['label'],
                                                 yHatTrain[:, 1])
        trainAuc += metrics.auc(fpr, tpr)
        # calculate auc for test dataset
        fpr, tpr, thresholds = metrics.roc_curve(testY['label'],
                                                 yHatTest[:, 1])
        testAuc += metrics.auc(fpr, tpr)

    trainAuc = trainAuc / k                 # sum up the accuracies and take the average
    testAuc = testAuc / k
    after = time.time()
    timeElapsed = after - before
    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    trainAuc = 0
    testAuc = 0
    before = time.time()
    for i in range(s):                                                      # for the number of samples
        trainX, testX, trainY, testY = train_test_split(xFeat, y, shuffle=True, test_size=testSize)

        model.fit(trainX, trainY)                                           # split and fit the data

        yHatTrain = model.predict_proba(trainX)                             # fit the data
        yHatTest = model.predict_proba(testX)

        # calculate auc for training
        fpr, tpr, thresholds = metrics.roc_curve(trainY['label'],
                                                 yHatTrain[:, 1])
        trainAuc += metrics.auc(fpr, tpr)
        # calculate auc for test dataset
        fpr, tpr, thresholds = metrics.roc_curve(testY['label'],
                                                 yHatTest[:, 1])
        testAuc += metrics.auc(fpr, tpr)
    trainAuc = trainAuc / s                                                 # sum and take tge average accuracies
    testAuc = testAuc / s
    after = time.time()
    timeElapsed = after - before
    return trainAuc, testAuc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training

    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.70)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.70, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.70, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)

    """
    The code below creates a table of the above data in matplotlib. This was used for the writeup. 
    """
    plt.figure()

    # table
    plt.subplot(121)

    cell_text = []
    for row in range(len(perfDF)):
        cell_text.append(perfDF.iloc[row])

    table = plt.table(cellText=cell_text, colLabels=perfDF.columns, loc='center', colWidths=[0.3 for x in perfDF.columns])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()
