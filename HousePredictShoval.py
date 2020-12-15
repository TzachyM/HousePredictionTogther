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


def fix_conditions(df, col, val):
    df.loc[df[col] != val, col] = 0
    df.loc[df[col] == val, col] = 1
    return df


def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for name in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']:
        df[name] = df[name].fillna('NA')
    # df = df.fillna(df.mode().iloc[0])
    return df


def features_engineering(df):
    # mergin condition 1 and 2 to 1 column
    df = fix_conditions(df, 'Condition1', 'Norm')
    df = fix_conditions(df, 'Condition2', 'Norm')
    df['Condition'] = df['Condition1'] + df['Condition2']
    df.drop(['Condition1', 'Condition2', 'Utilities', 'MSSubClass', 'RoofStyle', 'Foundation'], axis=1,
            inplace=True)  # MSSubClass , RoofStyle
    # make the yearbuild a range of years
    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)
    # make the YearRemodAdd a range of years
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1920, 1940, 1960, 1980, 2000, 2020]
    df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)
    # change roof matirial
    df = fix_conditions(df, 'RoofMatl', 'WdShngl')
    # group MasVnrArea by ranges
    labels = [0, 1, 2]
    bins = [0, 1, 400, 2000]
    dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0, 'Mn':2 , 'No':1	}
    df.MasVnrArea = pd.cut(df.MasVnrArea, bins, labels=labels, include_lowest=True)
    df.ExterQual = df.ExterQual.map(dict)
    df.ExterCond = df.ExterCond.map(dict)
    df.BsmtQual = df.BsmtQual.map(dict)
    df.BsmtCond = df.BsmtCond.map(dict)
    df.BsmtExposure = df.BsmtExposure.map(dict)
    return df


def visual_data(df):
    # view year build with salePrice

    df.plot.scatter(x='YearBuilt', y='SalePrice', xlim=(1880, 2020))

    # view RoofStyle with salePrice
    plt.figure()
    sns.boxplot(x='RoofStyle', y="SalePrice", data=df)

    # view RoofMatl  with salePrice
    plt.figure()
    sns.boxplot(x='RoofMatl', y="SalePrice", data=df)
    # view MasVnrArea  with salePrice
    df.plot.scatter(x='MasVnrArea', y='SalePrice')
    # view ExterQual  with salePrice
    plt.figure()
    sns.boxplot(x='ExterQual', y="SalePrice", data=df)
    # view Foundation  with salePrice
    plt.figure()
    sns.boxplot(x='Foundation', y="SalePrice", data=df)
    # view BsmtQual  with salePrice
    plt.figure()
    sns.boxplot(x='BsmtQual', y="SalePrice", data=df)
    # view BsmtQual  with salePrice
    plt.figure()
    sns.boxplot(x='BsmtCond', y="SalePrice", data=df)

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
