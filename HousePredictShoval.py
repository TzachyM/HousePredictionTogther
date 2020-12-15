# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:56:27 2020

@author: ggpen
"""
#taasd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.decomposition import PCA
import skimage.io
import seaborn as sns
from collections import Counter
from pandas.api.types import is_numeric_dtype


def fix_conditions(df, col):
    df.loc[df[col] != 'Norm', col] = 1
    df.loc[df[col] == 'Norm', col] = 0
    return df


def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    # df = df.fillna(df.mode().iloc[0])
    return df


def features_engineering(df):
    # mergin condition 1 and 2 to 1 column
    df = fix_conditions(df, 'Condition1')
    df = fix_conditions(df, 'Condition2')
    df['Condition'] = df['Condition1'] + df['Condition2']
    df.drop(['Condition1', 'Condition2', 'Utilities', 'MSSubClass', 'RoofStyle'], axis=1,
            inplace=True)  # MSSubClass , RoofStyle
    # make the yearbuild a range of years
    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)
    # make the YearRemodAdd a range of years
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1920, 1940, 1960, 1980, 2000, 2020]
    df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)

    return df


def visual_data(df):
    # view year build with salePrice
    df.plot.scatter(x='YearBuilt', y='SalePrice', xlim=(1880, 2020))
    plt.show()
    # view RoofStyle with salePrice
    sns.boxplot(x='RoofStyle', y="SalePrice", data=df)
    plt.show()
    # view RoofMatl  with salePrice
    sns.boxplot(x='RoofMatl', y="SalePrice", data=df)
    plt.show()

if __name__ == '__main__':
    # load data
    train_data = pd.read_csv(r'House.csv')
    test_data = pd.read_csv(r'TestHouse.csv')
    # preproccesing
    visual_data(train_data)  # visualizition
    id_test = test_data['Id']  # save Id for later to submit prediction
    df = pd.concat([train_data, test_data]).reset_index(drop=True)  # marge both train and test
    df = df.drop(['Id', 'SalePrice'], axis=1)  # drop useless columns
    df = fill_na(df)  # fill all the nulls after viewing the data
    df = features_engineering(df)  # change the features to better results
