# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import argparse
import numpy as np
import pandas as pd
import time
from sklearn.utils import shuffle
from lr import LinearRegression, file_to_numpy
from standardLR import StandardLR
import matplotlib.pyplot as plt


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest, fraction_train_data=1.0):
        """
        See definition in LinearRegression class
        """
        self.beta = np.ones(len(xTrain[0]) + 1)                     # initialize betas to ones
        ones = np.ones(len(xTrain))                                 # append column of 1's to features
        onesT = np.ones(len(xTest))
        X = np.concatenate((ones[:, np.newaxis], xTrain), axis=1)
        XT = np.concatenate((onesT[:, np.newaxis], xTest), axis=1)

        # for question 3b asking for only 40% of the training data. If the fraction kept isn't 1, then split data
        if fraction_train_data != 1.0:
            X, removedX = np.split(X, [int(len(X)*fraction_train_data)])
            yTrain, removedyTrain = np.split(yTrain, [int(len(yTrain)*fraction_train_data)])
            # now only X and yTrain are used and have for example 40% of the initial training data
        trainStats = {}

        for epoch in range(self.mEpoch):                                                    # for each epoch
            start = time.time()
            mergedSet = np.concatenate((X, yTrain), axis=1)                                 # merge X and Y for shuffle
            mergedSet = shuffle(mergedSet)                                                  # shuffle the data

            yTrainShuffled = mergedSet[:, -1]                                               # get the X samples back
            xTrainShuffled = mergedSet[:, :-1]                                              # get the labels back

            x_batch_all = np.array_split(xTrainShuffled, len(X)/self.bs)    # split data int batches of the same size
            y_batch_all = np.array_split(yTrainShuffled, len(yTrain)/self.bs)

            for x_batch, y_batch in zip(x_batch_all, y_batch_all):          # for each batch
                gradient = np.zeros(5)                                      # initialize the gradient to 0
                for sample_x, sample_y in zip(x_batch, y_batch):            # for each sample in a batch
                    # calculate the gradient
                    new_grad = np.multiply(np.transpose(sample_x), sample_y - np.matmul(sample_x, self.beta))
                    gradient = np.add(new_grad, gradient)                   # sum up all the gradients
                avg_gradient = np.divide(gradient, self.bs)                 # divide by size of batch to get avg
                self.beta += self.lr * avg_gradient                         # multiply by lr and add to beta

            end = time.time()
            # save the iteration number, time, train-mse, and test-mse
            trainStats[len(x_batch_all) * epoch] = {'time': end - start,
                               'train-mse': self.mse(X, yTrain),
                               'test-mse': self.mse(XT, yTest)}
        return trainStats


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)
    #
    # ORIGINAL CODE
    #

    # model = SgdLR(args.lr, args.bs, args.epoch)
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    # print(trainStats)

    # END ORIGINAL CODE
    #

    # # Code for Question 3B ################################
    # for lr in [0.1, 0.01, 0.001, 0.0001, .00001]:     # For each learning rate, train a model and plot the train mse
    #     model = SgdLR(lr, 1, args.epoch)
    #     trainStats = model.train_predict(xTrain, yTrain, xTest, yTest, fraction_train_data=.4)
    #     results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    #     plt.plot(results, label=lr)
    #
    # plt.title("Training MSE for Various Learning Rates with Batch Size 1 and 100 Epochs")
    # plt.ylim(top=1.25, bottom=.3)                     # The plot y axis was limited so that large values wouldn't
    # plt.xlabel("Epoch")                               # squeeze the small values. Reduced distortion but
    # plt.ylabel("MSE")                                 # higher, irrelevant initial MSE's were cut off.
    # plt.legend()
    # plt.show()

    # # Code for Question 3C ################################
    # model = SgdLR(.001, 1, args.epoch)                # the optimal MSE was chose from the above plot manually
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)        # train model using lr=.001
    # results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    # plt.plot(results, label='train-mse-lr=.001')
    # results = [trainStats[i]['test-mse'] for i in trainStats.keys()]
    # plt.plot(results, label='test-mse-lr=.001')
    #
    # # plots the train and test mse for the optimal lr over the number of epochs
    #
    # plt.title("Training and Test MSE for .001 Learning Rate with Batch Size 1 and 100 Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.show()

    # # Code for Question 4A ################################
    # The following code plots different learning rates for each batch size to find the optimal one
    # makes 12 plots, one for each batch size, not included in write up because way too many plots
    list_of_batches = [1, 5, 10, 15, 26, 39, 65, 86, 129, 195, 258, len(xTrain)]
    # for batch_size in list_of_batches:                                        # for each batch size
    #     plt.figure()
    #     plt.title("Training and Test MSE for 50 Epochs")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("MSE")
    #     for lr in [0.1, 0.01, 0.001, 0.0001]:                                 # for each lr, train a model
    #         model = SgdLR(lr, batch_size, args.epoch)
    #         trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    #         results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    #         plt.ylim(top=2.0)                                                 # limit the y axis to see data better
    #         plt.plot(results, label="bs=" + str(batch_size) + " lr=" + str(lr))
    #     plt.legend()
    # plt.show()

    # optimal LR found from code above and manually selecting the best LR.
    # optimal_lr = [.001, .01, .01, .01, .01, .01, .1, .1, .1, .1, .1, .1]    # manually selected from above plots
    # train_mse = []
    # test_mse = []
    # time_list = []
    # for batch_size, lr in zip(list_of_batches, optimal_lr):                 # for each batch size and optimal lr
    #     start = time.time()                                                 # train a model
    #     model = SgdLR(lr, batch_size, args.epoch)
    #     trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    #     end = time.time()
    #     train_mse.append(trainStats[list(trainStats.keys())[-1]]['train-mse'])  # save the ending train, test mse and
    #     test_mse.append(trainStats[list(trainStats.keys())[-1]]['test-mse'])    # time it took
    #     time_list.append(end-start)
    #
    # model = StandardLR()
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)              # train a closed form model
    #
    # # print out two scatter plots, one for train mse, one for test-mse vs time taken.
    #
    # plt.figure()
    # plt.title("Training MSE vs Total Time for 50 Epochs")
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Training MSE")
    # plt.scatter(time_list, train_mse, label='sgd-mse')                                          # SGD points
    # plt.scatter(trainStats[0]['time'], trainStats[0]['train-mse'], label='closed-form-mse')     # closed form point
    # plt.ylim(top=.5, bottom=.25)                                                # limit y axis to better see the data
    #
    # plt.figure()
    # plt.title("Test MSE vs Total Time for 50 Epochs")
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Test MSE")
    # plt.scatter(time_list, test_mse, label='sgd-mse')                                           # SGD points
    # plt.scatter(trainStats[0]['time'], trainStats[0]['test-mse'], label='closed-form-mse')      # closed form point
    # plt.ylim(top=.5, bottom=.25)                                                # limit y axis to better see the data
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()

