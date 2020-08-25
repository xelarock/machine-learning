from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()

fig = plt.figure()
fig.tight_layout()
fig.suptitle('Features of Iris Flowers', fontsize=14, fontweight='bold')

setosaSepalLength = iris.data[:50, :1]
versicolourSepalLength = iris.data[50:100, :1]
virginicaSepalLength = iris.data[100:, :1]

sepalLength = np.concatenate((setosaSepalLength, versicolourSepalLength, virginicaSepalLength), axis=1)
print(sepalLength)

sepalLengthPlot = fig.add_subplot(141)
sepalLengthPlot.boxplot(sepalLength, labels=("Setosa", "Versicolour", "Virginica"))
sepalLengthPlot.set_xlabel("Flower Type")
sepalLengthPlot.set_ylabel("Sepal Length")
sepalLengthPlot.set_title("Sepal Length of Three Species of Irises")

setosaSepalWidth = iris.data[:50, 1:2]
versicolourSepalWidth = iris.data[50:100, 1:2]
virginicaSepalWidth = iris.data[100:, 1:2]

sepalWidth = np.concatenate((setosaSepalWidth, versicolourSepalWidth, virginicaSepalWidth), axis=1)
print(sepalWidth)

sepalWidthPlot = fig.add_subplot(142)
sepalWidthPlot.boxplot(sepalWidth, labels=("Setosa", "Versicolour", "Virginica"))
sepalWidthPlot.set_xlabel("Flower Type")
sepalWidthPlot.set_ylabel("Sepal Width")
sepalWidthPlot.set_title("Sepal Width of Three Species of Irises")

setosaPetalLength = iris.data[:50, 2:3]
versicolourPetalLength = iris.data[50:100, 2:3]
virginicaPetalLength = iris.data[100:, 2:3]

petalLength = np.concatenate((setosaPetalLength, versicolourPetalLength, virginicaPetalLength), axis=1)
print(petalLength)

petalLengthPlot = fig.add_subplot(143)
petalLengthPlot.boxplot(petalLength, labels=("Setosa", "Versicolour", "Virginica"))
petalLengthPlot.set_xlabel("Flower Type")
petalLengthPlot.set_ylabel("Petal Length")
petalLengthPlot.set_title("Petal Length of Three Species of Irises")

setosaPetalWidth = iris.data[:50, 3:]
versicolourPetalWidth = iris.data[50:100, 3:]
virginicaPetalWidth = iris.data[100:, 3:]

petalWidth = np.concatenate((setosaPetalWidth, versicolourPetalWidth, virginicaPetalWidth), axis=1)
print(petalWidth)

petalWidthPlot = fig.add_subplot(144)
petalWidthPlot.boxplot(petalWidth, labels=("Setosa", "Versicolour", "Virginica"))
petalWidthPlot.set_xlabel("Flower Type")
petalWidthPlot.set_ylabel("Petal Width")
petalWidthPlot.set_title("Petal Width of Three Species of Irises")

plt.subplots_adjust(left=.05, right=.95)
plt.show()



