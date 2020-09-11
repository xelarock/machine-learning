import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


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

    print("hello")
    # mergedSet = pd.concat([xTrain, yTrain], axis=1)
    # print(mergedSet)
    # leftX = mergedSet[mergedSet['alcohol'] <= 9.5]
    # leftY = leftX.iloc[:, -1]
    # leftX = leftX.drop(labels='label', axis=1)
    #
    # rightX = mergedSet[mergedSet['alcohol'] > 9.5]
    # rightY = rightX.iloc[:, -1]
    # rightX = rightX.drop(labels='label', axis=1)
    # print(leftX, leftY, rightX, rightY)

    # print(len(xTrain))
    # print(yTrain.value_counts().values[0], yTrain.value_counts().values[1])
    # zero = yTrain.value_counts().values[0]
    # one = yTrain.value_counts().values[1]
    # total = len(yTrain)

    # print("Gini1:", 1-(((zero/total) ** 2) + ((one/total) ** 2)))
    #
    # gini = 0
    # total = len(yTrain)
    # for i in range(len(yTrain.value_counts().values)):
    #     gini += (yTrain.value_counts().values[i] / total) ** 2
    # gini = 1 - gini
    # print("gini2: ", gini)

    print(xTrain['volatile acidity'].unique())

    for index, sample in xTrain.iterrows():
        print(index, sample)

    # dt1 = DecisionTree('gini', args.md, args.mls)
    # trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    # print("GINI Criterion ---------------")
    # print("Training Acc:", trainAcc1)
    # print("Test Acc:", testAcc1)
    # dt = DecisionTree('entropy', args.md, args.mls)
    # trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    # print("Entropy Criterion ---------------")
    # print("Training Acc:", trainAcc)
    # print("Test Acc:", testAcc)

if __name__ == "__main__":
    main()