import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter
from itertools import dropwhile
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer


def model_assessment(filename):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """
    with open(filename) as f:
        lines = pd.DataFrame(f.read().splitlines(), columns=['email'])
    all_samples = pd.DataFrame(lines['email'].str.split(' ', 1).tolist(), columns=['email_label', 'email'])
    columns_titles = ['email', 'email_label']
    all_samples = all_samples.reindex(columns=columns_titles)

    all_samples = shuffle(all_samples)
    y = all_samples.iloc[:, -1:]
    x = all_samples.iloc[:, 0:len(all_samples.columns) - 1]  # separate the features of labels again

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=.7, shuffle=True)

    return xTrain, xTest, yTrain, yTest


def build_vocab_map(xTrain):
    # emails = open(filename).read()
    # docs = emails.split("\n")
    vec = CountVectorizer(min_df=30).fit(xTrain['email'])
    bag_of_words = vec.transform(xTrain['email'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    print("num words:", len(words_freq))
    return dict(words_freq)

def construct_binary(vocab_counter, xTrain, xTest):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """

    print(xTrain)
    # print(vocab_counter.keys())
    vec = CountVectorizer(vocabulary=vocab_counter.keys()).fit(xTrain["email"])
    bag_of_words = vec.transform(xTrain["email"])
    binary_Train = pd.DataFrame(bag_of_words.toarray())
    binary_Train.columns = vocab_counter.keys()
    binary_Train.index = xTrain.index

    for col in binary_Train.columns:
        binary_Train.loc[binary_Train[col] >= 1, col] = 1

    print(binary_Train)

    pd.options.display.max_rows = 50
    print(xTest)
    vec = CountVectorizer(vocabulary=vocab_counter.keys()).fit(xTest["email"])
    bag_of_words = vec.transform(xTest["email"])
    binary_Test= pd.DataFrame(bag_of_words.toarray())
    binary_Test.columns = vocab_counter.keys()
    binary_Test.index = xTest.index

    for col in binary_Test.columns:
        binary_Test.loc[binary_Test[col] >= 1, col] = 1

    print(binary_Test)

    return binary_Train, binary_Test


def construct_count(vocab_counter, xTrain, xTest):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    print(xTrain)
    vec = CountVectorizer(vocabulary=vocab_counter.keys()).fit(xTrain["email"])
    bag_of_words = vec.transform(xTrain["email"])
    count_Train = pd.DataFrame(bag_of_words.toarray())
    count_Train.columns = vocab_counter.keys()
    count_Train.index = xTrain.index
    print(count_Train)

    # pd.options.display.max_rows = 50
    print(xTest)
    vec = CountVectorizer(vocabulary=vocab_counter.keys()).fit(xTest["email"])
    bag_of_words = vec.transform(xTest["email"])
    count_Test = pd.DataFrame(bag_of_words.toarray())
    count_Test.columns = vocab_counter.keys()
    count_Test.index = xTest.index

    print(count_Test)

    return count_Train, count_Test


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    xTrain, xTest, yTrain, yTest = model_assessment(args.data)
    vocab_dict = build_vocab_map(xTrain)
    binary_Train, binary_Test = construct_binary(vocab_dict, xTrain, xTest)
    count_Train, count_Test = construct_count(vocab_dict, xTrain, xTest)

    yTrain.to_csv("yTrain.csv", index=False)
    yTest.to_csv("yTest.csv", index=False)

    binary_Train.to_csv("binary_Train.csv", index=False)
    binary_Test.to_csv("binary_Test.csv", index=False)

    count_Train.to_csv("count_Train.csv", index=False)
    count_Test.to_csv("count_Test.csv", index=False)


if __name__ == "__main__":
    main()
