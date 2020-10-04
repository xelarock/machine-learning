import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    xTrain = pd.read_csv('new_xTrain-pre-select-features.csv')
    yTrain = pd.read_csv('eng_yTrain.csv')

    mergedSet = pd.concat([xTrain, yTrain], axis=1)

    corr = mergedSet.corr(method='pearson')
    ax = sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap="RdBu")

    # print(corr)

    for feature in corr.iloc[-1].index:
        if corr.iloc[-1][feature] > .2 or corr.iloc[-1][feature] < -.2:
            print(feature, corr.iloc[-1][feature])

    plt.show()

if __name__ == "__main__":
    main()