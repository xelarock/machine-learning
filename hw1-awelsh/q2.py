from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()                             # loads iris data set and partitions the data into the three
                                                        # flowers with their four features. Total 12 arrays of data
setosaSepalLength = iris.data[:50, :1]
versicolourSepalLength = iris.data[50:100, :1]
virginicaSepalLength = iris.data[100:, :1]

setosaSepalWidth = iris.data[:50, 1:2]
versicolourSepalWidth = iris.data[50:100, 1:2]
virginicaSepalWidth = iris.data[100:, 1:2]

setosaPetalLength = iris.data[:50, 2:3]
versicolourPetalLength = iris.data[50:100, 2:3]
virginicaPetalLength = iris.data[100:, 2:3]

setosaPetalWidth = iris.data[:50, 3:]
versicolourPetalWidth = iris.data[50:100, 3:]
virginicaPetalWidth = iris.data[100:, 3:]


def q2b():                                              # code for figure for Question 2B
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Features of Iris Flowers', fontsize=14, fontweight='bold')        # sets title of figure

    sepalLength = np.concatenate((setosaSepalLength, versicolourSepalLength, virginicaSepalLength), axis=1)
    # print(sepalLength)                                # merges the sepal length of each flower into a 50*3 array

    sepalLengthPlot = fig.add_subplot(141)              # creates the plot for sepal length
    sepalLengthPlot.boxplot(sepalLength, labels=("Setosa", "Versicolour", "Virginica")) # set box plot values and labels
    sepalLengthPlot.set_xlabel("Flower Type")           # set x axis label
    sepalLengthPlot.set_ylabel("Sepal Length")          # set y axis label
    sepalLengthPlot.set_title("Sepal Length of Three Species of Irises")    # set figure title

    sepalWidth = np.concatenate((setosaSepalWidth, versicolourSepalWidth, virginicaSepalWidth), axis=1)
    # print(sepalWidth)                                 # merges the sepal width of each flower into a 50*3 array

    sepalWidthPlot = fig.add_subplot(142)               # creates the plot for sepal width
    sepalWidthPlot.boxplot(sepalWidth, labels=("Setosa", "Versicolour", "Virginica"))   # set box plot values and labels
    sepalWidthPlot.set_xlabel("Flower Type")            # set x axis label
    sepalWidthPlot.set_ylabel("Sepal Width")            # set y axis label
    sepalWidthPlot.set_title("Sepal Width of Three Species of Irises")  # set y axis label

    petalLength = np.concatenate((setosaPetalLength, versicolourPetalLength, virginicaPetalLength), axis=1)
    # print(petalLength)                                # merges the petal length of each flower into a 50*3 array

    petalLengthPlot = fig.add_subplot(143)              # creates the plot for petal length
    petalLengthPlot.boxplot(petalLength, labels=("Setosa", "Versicolour", "Virginica")) # set box plot values and labels
    petalLengthPlot.set_xlabel("Flower Type")           # set x axis label
    petalLengthPlot.set_ylabel("Petal Length")          # set y axis label
    petalLengthPlot.set_title("Petal Length of Three Species of Irises")    # set y axis label

    petalWidth = np.concatenate((setosaPetalWidth, versicolourPetalWidth, virginicaPetalWidth), axis=1)
    # print(petalWidth)                                   # merges the petal width of each flower into a 50*3 array

    petalWidthPlot = fig.add_subplot(144)               # creates the plot for petal width
    petalWidthPlot.boxplot(petalWidth, labels=("Setosa", "Versicolour", "Virginica"))   # set box plot values and labels
    petalWidthPlot.set_xlabel("Flower Type")            # set x axis label
    petalWidthPlot.set_ylabel("Petal Width")            # set y axis label
    petalWidthPlot.set_title("Petal Width of Three Species of Irises")      # set y axis label

    plt.subplots_adjust(left=.05, right=.95)            # adjst margins of overall figure to improve readability


def q2c():
    fig = plt.figure()                                  # create figure and set layout for readability
    fig.tight_layout()
    fig.suptitle('Length vs Width for Iris Features', fontsize=14, fontweight='bold')   # set title

    sepalPlot = fig.add_subplot(121)    # for sepal data, add the 3 flower's data. length on x, width on y
    sepalPlot.scatter(setosaSepalLength, setosaSepalWidth, label='Setosa')
    sepalPlot.scatter(versicolourSepalLength, versicolourSepalWidth, label='Verisicolor')
    sepalPlot.scatter(virginicaSepalLength, virginicaSepalWidth, label='Virginica')
    sepalPlot.set_xlabel("Sepal Length")                # set x axis label
    sepalPlot.set_ylabel("Sepal Width")                 # set y axis label
    plt.legend(loc="upper left")                        # set location of legend to upper left

    petalPlot = fig.add_subplot(122)    # for petal data, add the 3 flower's data. length on x, width on y
    petalPlot.scatter(setosaPetalLength, setosaPetalWidth, label='Setosa')
    petalPlot.scatter(versicolourPetalLength, versicolourPetalWidth, label='Verisicolor')
    petalPlot.scatter(virginicaPetalLength, virginicaPetalWidth, label='Virginica')
    petalPlot.set_xlabel("Petal Length")                # set x axis label
    petalPlot.set_ylabel("Petal Width")                 # set y axis label
    plt.legend(loc="upper left")                        # set location of legend to upper left


q2b()           # run code for plot for Q2B
q2c()           # run code for plot for Q2C
plt.show()      # plot both figures

# answer to Q2d, see write-up for formatted response
# q2d Based on Part B, Setosa's ca be identified by their small petal length and width. They also have a smaller sepal
# length paired with a longer sepal width based on Part C. Verisicolor and Virginica are harder to separate. Using the
# figure from Part B, Virginica's tend to have longer sepal length, petal length, and petal width. They also have the
# longest petal length-petal width pair (Figure from 2C). The rest are likely to be Verisicolor and have the middle
# values of petal length and width (Figure from 2B, 2C)


