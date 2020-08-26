from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

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


def q2b():
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Features of Iris Flowers', fontsize=14, fontweight='bold')

    sepalLength = np.concatenate((setosaSepalLength, versicolourSepalLength, virginicaSepalLength), axis=1)
    print(sepalLength)

    sepalLengthPlot = fig.add_subplot(141)
    sepalLengthPlot.boxplot(sepalLength, labels=("Setosa", "Versicolour", "Virginica"))
    sepalLengthPlot.set_xlabel("Flower Type")
    sepalLengthPlot.set_ylabel("Sepal Length")
    sepalLengthPlot.set_title("Sepal Length of Three Species of Irises")

    sepalWidth = np.concatenate((setosaSepalWidth, versicolourSepalWidth, virginicaSepalWidth), axis=1)
    print(sepalWidth)

    sepalWidthPlot = fig.add_subplot(142)
    sepalWidthPlot.boxplot(sepalWidth, labels=("Setosa", "Versicolour", "Virginica"))
    sepalWidthPlot.set_xlabel("Flower Type")
    sepalWidthPlot.set_ylabel("Sepal Width")
    sepalWidthPlot.set_title("Sepal Width of Three Species of Irises")

    petalLength = np.concatenate((setosaPetalLength, versicolourPetalLength, virginicaPetalLength), axis=1)
    print(petalLength)

    petalLengthPlot = fig.add_subplot(143)
    petalLengthPlot.boxplot(petalLength, labels=("Setosa", "Versicolour", "Virginica"))
    petalLengthPlot.set_xlabel("Flower Type")
    petalLengthPlot.set_ylabel("Petal Length")
    petalLengthPlot.set_title("Petal Length of Three Species of Irises")

    petalWidth = np.concatenate((setosaPetalWidth, versicolourPetalWidth, virginicaPetalWidth), axis=1)
    print(petalWidth)

    petalWidthPlot = fig.add_subplot(144)
    petalWidthPlot.boxplot(petalWidth, labels=("Setosa", "Versicolour", "Virginica"))
    petalWidthPlot.set_xlabel("Flower Type")
    petalWidthPlot.set_ylabel("Petal Width")
    petalWidthPlot.set_title("Petal Width of Three Species of Irises")

    plt.subplots_adjust(left=.05, right=.95)


def q2c():
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Length vs Width for Iris Features', fontsize=14, fontweight='bold')

    sepalPlot = fig.add_subplot(121)
    sepalPlot.scatter(setosaSepalLength, setosaSepalWidth, label='Setosa')
    sepalPlot.scatter(versicolourSepalLength, versicolourSepalWidth, label='Verisicolor')
    sepalPlot.scatter(virginicaSepalLength, virginicaSepalWidth, label='Virginica')
    sepalPlot.set_xlabel("Sepal Length")
    sepalPlot.set_ylabel("Sepal Width")
    plt.legend(loc="upper left")

    petalPlot = fig.add_subplot(122)
    petalPlot.scatter(setosaPetalLength, setosaPetalWidth, label='Setosa')
    petalPlot.scatter(versicolourPetalLength, versicolourPetalWidth, label='Verisicolor')
    petalPlot.scatter(virginicaPetalLength, virginicaPetalWidth, label='Virginica')
    petalPlot.set_xlabel("Petal Length")
    petalPlot.set_ylabel("Petal Width")
    plt.legend(loc="upper left")


q2b()
q2c()
plt.show()

# q2d Based on Part B, Setosa's ca be identified by their small petal length and width. They also have a smaller sepal
# length paired with a longer sepal width based on Part C. Verisicolor and Virginica are harder to separate. Using the
# figure from Part B, Virginica's tend to have longer sepal length, petal length, and petal width. They also have the
# longest petal length-petal width pair (Figure from 2C). The rest are likely to be Verisicolor and have the middle
# values of petal length and width (Figure from 2B, 2C)


