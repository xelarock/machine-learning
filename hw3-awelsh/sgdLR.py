import argparse
import numpy as np
import pandas as pd
import time
from sklearn.utils import shuffle
from lr import LinearRegression, file_to_numpy
import matplotlib.pyplot as plt


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def divide_chunks(self, input):
        # looping till length l
        if input.ndim == 1:
            for i in range(0, len(input), self.bs):
                yield input[i:i+self.bs]
        else:
            for i in range(0, len(input), self.bs):
                yield input[i:i+self.bs, :]

    def train_predict(self, xTrain, yTrain, xTest, yTest, fraction_train_data=1.0):
        """
        See definition in LinearRegression class
        """
        self.beta = np.zeros(len(xTrain[0]) + 1)
        # print(len(xTrain))
        ones = np.ones(len(xTrain))
        onesT = np.ones(len(xTest))
        # print(ones.T)
        X = np.concatenate((ones[:, np.newaxis], xTrain), axis=1)
        XT = np.concatenate((onesT[:, np.newaxis], xTest), axis=1)

        if fraction_train_data != 1.0:
            X, removedX = np.split(X, [int(len(X)*fraction_train_data)])
            yTrain, removedyTrain = np.split(yTrain, [int(len(yTrain)*fraction_train_data)])
            print(len(X), len(removedX), len(yTrain), len(removedyTrain))
        trainStats = {}
        # print(xTrain, yTrain)

        for epoch in range(self.mEpoch):
            start = time.time()
            mergedSet = np.concatenate((X, yTrain), axis=1)
            mergedSet = shuffle(mergedSet)

            yTrainShuffled = mergedSet[:, -1]
            xTrainShuffled = mergedSet[:, :-1]
            # print(xTrainShuffled, yTrainShuffled)

            # print("!!!!!!!!!")
            # print(list(self.divide_chunks(xTrainShuffled)))
            x_batch_all = np.array_split(xTrainShuffled, len(X)/self.bs)
            # print(x_batch_all)
            # print("!!!!!!!!!++++++++")
            y_batch_all = np.array_split(yTrainShuffled, len(yTrain)/self.bs)
            # print("!!!!!!!!!---------------")
            # print("Len: ", np.shape(x_batch_all))
            # print(y_batch_all)
            #
            # print(epoch)

            for x_batch, y_batch in zip(x_batch_all, y_batch_all):
                gradient = np.zeros(5)
                # print("size:", len(x_batch))
                for sample_x, sample_y in zip(x_batch, y_batch):
                    new_grad = np.multiply(np.transpose(sample_x), sample_y - np.matmul(sample_x, self.beta))
                    # print("new grad:", new_grad)
                    gradient = np.add(new_grad, gradient)
                    # print("sample: ", sample_x, sample_y, gradient)
                    #gradient = np.add(np.matmul(np.transpose(sample_x), sample_y - np.matmul(sample_x, self.beta)), gradient)
                # avg_gradient = np.matmul(np.transpose(x_batch), y_batch - np.matmul(x_batch, self.beta))
                # break
                avg_gradient = np.divide(gradient, self.bs)
                # print(avg_gradient)
                self.beta += self.lr * avg_gradient
                # print(self.beta)

            end = time.time()
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            trainStats[len(x_batch_all) * epoch] = {'time': end - start,
                               'train-mse': self.mse(X, yTrain),
                               'test-mse': self.mse(XT, yTest)}
            # print("epoch:", epoch, )
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
    # model = SgdLR(args.lr, args.bs, args.epoch)
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    # print(trainStats)
    # for key, value in trainStats.items():
    #     print(key, ' : ')
    #     for key1, value1 in value.items():
    #         print(key1, ' : ', value1)

    # Code for Question 3B
    # model = SgdLR(args.lr, args.bs, args.epoch)
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest, fraction_train_data=.4)
    # for key, value in trainStats.items():
    #     print(key, ' : ')
    #     for key1, value1 in value.items():
    #         print(key1, ' : ', value1)

    # print(trainStats.keys())
    # for lr in [0.1, 0.01, 0.001, 0.0001]:
    #     print("LEARNING RATE OF: ", lr)
    #     model = SgdLR(lr, 1, args.epoch)
    #     trainStats = model.train_predict(xTrain, yTrain, xTest, yTest, fraction_train_data=.4)
    #     # for key, value in trainStats.items():
    #     #     print(key, ' : ')
    #     #     for key1, value1 in value.items():
    #     #         print(key1, ' : ', value1)
    #     results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    #     plt.plot(results, label=lr)
    #
    # plt.title("Training MSE for Various Learning Rates with Batch Size 1 and 100 Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.show()

    # Code for Question 3C
    # model = SgdLR(.001, 1, args.epoch)
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    # results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
    # plt.plot(results, label='train-mse-lr=.001')
    # results = [trainStats[i]['test-mse'] for i in trainStats.keys()]
    # plt.plot(results, label='test-mse-lr=.001')
    #
    # plt.title("Training and Test MSE for .001 Learning Rate with Batch Size 1 and 50 Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.show()

    # Code for Question 4A

    # 5, 10, 15, 26, 39, 65, 86, 129, 195, 258,
    for batch_size in [1, 5, 10, 15, 26, 39, 65, 86, 129, 195, 258, len(xTrain)]:
        plt.figure()
        plt.title("Training and Test MSE for 50 Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            # print("LEARNING RATE OF: ", lr)
            model = SgdLR(lr, batch_size, args.epoch)
            trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
            if trainStats[0]['train-mse'] < 10.0:
                print("LEARNING RATE OF:", lr, "batch size of:", batch_size)
                results = [trainStats[i]['train-mse'] for i in trainStats.keys()]
                plt.plot(results, label="bs=" + str(batch_size) + " lr=" + str(lr))
        plt.legend()
    plt.show()




if __name__ == "__main__":
    main()

