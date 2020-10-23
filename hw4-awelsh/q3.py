from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd

def calc_mistakes(yHat, yTrue):
    mistakes = 0
    for index in range(len(yHat)):
        if yHat[index] != yTrue[index]:
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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy

    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    naive_bayes = BernoulliNB()
    trainStats = naive_bayes.fit(xTrain, yTrain.to_numpy().ravel())
    print(trainStats)
    yHat = naive_bayes.predict(xTest)
    print(yHat)
    print(yTest)
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest.to_numpy()))
    print("Accuracy on test dataset")
    print(1 - calc_mistakes(yHat, yTest.to_numpy()) / len(yTest.to_numpy()))


    # print out the number of mistakes
    # print("Number of mistakes on the test dataset")
    # print(calc_mistakes(yHat, yTest))
    # print("Accuracy on test dataset")
    # print(1 - calc_mistakes(yHat, yTest) / len(yTest))
    #
    # weights_df = pd.DataFrame(model.weights, columns=pd.read_csv(args.xTrain).columns)
    # sorted_df = weights_df.sort_values(by=0, axis=1, ascending=False)
    # print("15 most positive weights")
    # print(sorted_df.iloc[:, : 15])
    # print("15 most negative weights")
    # print(sorted_df.iloc[:, len(sorted_df.columns) - 15:])


if __name__ == "__main__":
    main()