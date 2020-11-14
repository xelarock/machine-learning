import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def q1(xTrain, xTest, yTrain, yTest):
    xTrain_scaled = StandardScaler().fit_transform(xTrain)
    xTest_scaled = StandardScaler().fit_transform(xTest)

    scaled_model = LogisticRegression(penalty="none")
    scaled_model.fit(xTrain_scaled, yTrain.values.ravel())
    print("Score:", scaled_model.score(xTest_scaled, yTest.values.ravel()))

    yHatTest = scaled_model.predict_proba(xTest_scaled)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    print("Test Auc:", testAuc)

    sklearn_pca = PCA(n_components=9)
    xTrain_pca = sklearn_pca.fit_transform(xTrain_scaled)

    df_labels = xTrain.columns

    for i, component in enumerate(sklearn_pca.components_):
        for j, feature in enumerate(df_labels):
            print("Component", i, "with feature", feature, "has", component[j], "variance")
        print()
    print()
    print("Amount of Variance Captured by Components", sum(sklearn_pca.explained_variance_ratio_))

    pca_model = LogisticRegression(penalty="none")
    pca_model.fit(xTrain_pca, yTrain.values.ravel())

    xTest_pca = sklearn_pca.transform(xTest_scaled)

    yHatTest = pca_model.predict_proba(xTest_pca)
    # calculate auc for training
    fpr_pca, tpr_pca, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc_pca = metrics.auc(fpr_pca, tpr_pca)
    print("Test Auc:", testAuc_pca)

    lw = 2
    plt.plot(fpr, tpr, color='orange',
             lw=lw, label='Normalized ROC curve (area = %0.2f)' % testAuc)
    plt.plot(fpr_pca, tpr_pca, color='royalblue',
             lw=lw, label='PCA ROC curve (area = %0.2f)' % testAuc_pca)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Normalized and PCA Wine Dataset')
    plt.legend(loc="lower right")

    plt.show()

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    np.random.seed(args.seed)

    q1(xTrain, xTest, yTrain, yTest)


if __name__ == "__main__":
    main()