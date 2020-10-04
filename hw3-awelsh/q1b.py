# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    xTrain = pd.read_csv('new_xTrain-pre-select-features.csv')          # this is the data before selection of features
    yTrain = pd.read_csv('eng_yTrain.csv')

    mergedSet = pd.concat([xTrain, yTrain], axis=1)                     # merge X and Y

    corr = mergedSet.corr(method='pearson')                             # calculate Pearson correlation
    ax = sns.heatmap(corr,                                              # make heat map
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap="RdBu")

    for feature in corr.iloc[-1].index:                                 # print out features that have a correlation
        if corr.iloc[-1][feature] > .2 or corr.iloc[-1][feature] < -.2: # greater than .2 or less than -.2
            print(feature, corr.iloc[-1][feature])
    plt.show()                                                          # show heat map

if __name__ == "__main__":
    main()