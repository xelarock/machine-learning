from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd


def calc_mistakes(yHat, yTrue):             # calculates the mistakes for each model
    mistakes = 0
    for index in range(len(yHat)):          # for each sample
        if yHat[index] != yTrue[index]:     # if prediction is wrong, count it as mistake
            mistakes += 1
    return mistakes


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
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data as df's

    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    if args.xTrain == "binary_Train.csv":                   # if the inputs are binary, use BernoulliNB
        naive_bayes = BernoulliNB()
    else:                                                   # else use MultinomialNB
        naive_bayes = MultinomialNB()
    naive_bayes.fit(xTrain, yTrain.to_numpy().ravel())      # fir to model
    yHat = naive_bayes.predict(xTest)                       # predict labels
    print(naive_bayes)
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest.to_numpy()))            # calculate mistakes
    print("Accuracy on test dataset")
    print(1 - calc_mistakes(yHat, yTest.to_numpy()) / len(yTest.to_numpy()))        # calculate accuracy

    print()

    logistic_regression = LogisticRegression(max_iter=1000)                     # do logistic regression for 100 epochs
    logistic_regression.fit(xTrain, yTrain.to_numpy().ravel())                  # fit test data
    yHat = logistic_regression.predict(xTest)                                   # predict labels
    print(logistic_regression)
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest.to_numpy()))                                # calculate number of mistakes
    print("Accuracy on test dataset")
    print(1 - calc_mistakes(yHat, yTest.to_numpy()) / len(yTest.to_numpy()))    # calculate accuracy



if __name__ == "__main__":
    main()