# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS.
# Alex Welsh

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # extracted the day of the year and the time of the day in minutes as 2 features.

    daydf = pd.to_datetime(df['date']).dt.dayofyear         # convert date to day of year
    datedf = pd.to_datetime(df['date'])
    timedf = datedf.dt.hour * 60 + datedf.dt.minute         # convert clock time total minutes
    df.insert(loc=0, column='time', value=timedf)           # appended them to the df
    df.insert(loc=0, column='day_of_year', value=daydf)
    df = df.drop(columns=['date'])                          # dropped the old date column
    return df


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # the following features had a Pearson correlation greater than .2 or less than -.2, and were selected because
    # that was a relatively high correlation compared to the other features.
    df = df[['time', 'lights', 'T2', 'RH_out']]
    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """

    # use the minmax scaler to scale data
    scaled_trainDF = MinMaxScaler().fit_transform(trainDF)
    scaled_testDF = MinMaxScaler().fit_transform(testDF)

    # had to convert then back to dataframes from numpy arrays
    scaled_trainDF = pd.DataFrame(scaled_trainDF, index=trainDF.index, columns=trainDF.columns)
    scaled_testDF = pd.DataFrame(scaled_testDF, index=testDF.index, columns=testDF.columns)

    return scaled_trainDF, scaled_testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
