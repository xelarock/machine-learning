import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    weights = None       # weights of the perceptron
    trainStats = {}

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        self.weights = np.ones((1, len(xFeat[0])))
        stats = {}

        for i in range(self.mEpoch):
            print("Epoch: ", i)
            num_wrong = 0
            for index in range(len(xFeat)):
                prediction = np.dot(self.weights, xFeat[index])[0]
                true_label = y[index][0]
                if prediction >= 0:
                    prediction = 1
                else:
                    prediction = 0
                if prediction != true_label:
                    num_wrong += 1
                    if true_label == 0:
                        self.weights = self.weights - xFeat[index]
                    elif true_label > 0:
                        self.weights = self.weights + xFeat[index]

            stats[i] = num_wrong

        return stats

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
        for index in range(len(xFeat)):
            prediction = np.dot(self.weights, xFeat[index])[0]
            if prediction >= 0:
                prediction = 1
            else:
                prediction = 0
            yHat.append(prediction)

        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    mistakes = 0
    for index in range(len(yHat)):
        if yHat[index] != yTrue[index]:
            mistakes += 1
    return mistakes


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def holdout(model, xFeat, y, testSize):
    before = time.time()
    trainX, testX, trainY, testY = train_test_split(xFeat, y, shuffle=True, test_size=testSize)     # split data

    model.trainStats = model.train(trainX, trainY)                                                  # fit data

    yHatTrain = model.predict(trainX)                                                         # predict
    yHatTest = model.predict(testX)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(trainY,
                                             yHatTrain)
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(testY,
                                             yHatTest)
    testAuc = metrics.auc(fpr, tpr)                                                                 # return the AUC
    after = time.time()
    timeElapsed = after - before
    return trainAuc, testAuc, timeElapsed


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
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    # print out the number of mistakes
    aucTrain1, aucVal1, time1 = holdout(model, xTrain, yTrain, 0.70)
    yHat = model.predict(xTest)
    print("Number of mistakes on the test dataset:")
    print(calc_mistakes(yHat, yTest))
    print("Accuracy on test dataset:")
    print(1 - calc_mistakes(yHat, yTest) / len(yTest))
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1]], columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print("\n", perfDF, "\n")

    weights_df = pd.DataFrame(model.weights, columns=pd.read_csv(args.xTrain).columns)
    sorted_df = weights_df.sort_values(by=0, axis=1, ascending=False)
    print("15 most positive weights")
    print(sorted_df.iloc[:, : 15], "\n")
    print("15 most negative weights")
    print(sorted_df.iloc[:, len(sorted_df.columns) - 15:])

    plt.plot(model.trainStats.keys(), model.trainStats.values())
    plt.xlabel("Epoch")
    plt.ylabel("Mistakes")
    plt.ylim(bottom=-5)
    plt.show()


if __name__ == "__main__":
    main()