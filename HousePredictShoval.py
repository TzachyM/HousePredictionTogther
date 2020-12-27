# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:56:27 2020
@author: ggpen
"""
#16/12
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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def rmsle_cv(model,train, y_train):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


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
    # view
    df.plot.scatter(x='1stFlrSF', y='SalePrice')
    # view
    plt.show()
    df.plot.scatter(x='2ndFlrSF', y='SalePrice')


def detect_outliers(df, n):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in df.columns:
        if(is_numeric_dtype(df[col])):
            # 1st quartile (25%)
            Q1 = np.percentile(df[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.6 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def fix_conditions(df, col, val):
    # df.loc[df[col] != val, col] = 0
    # df.loc[df[col] == val, col] = 1
    # df[col] = df[col].astype('int64')
    return df


def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for name in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','FireplaceQu', 'GarageType','GarageFinish','GarageQual', 'GarageCond','MiscFeature']:
        df[name] = df[name].fillna('NA')
    df = df.fillna(df.mode().iloc[0])
    return df


def features_engineering(df):
    # mergin condition 1 and 2 to 1 column
    df = fix_conditions(df, 'Condition1', 'Norm')
    df = fix_conditions(df, 'Condition2', 'Norm')
    df['Condition'] = df['Condition1'] + df['Condition2']
    df.drop(['Condition1', 'Condition2', 'Utilities','HalfBath','GarageArea','EnclosedPorch','MoSold','YrSold',
    '3SsnPorch','ScreenPorch', 'PoolArea' , 'PoolQC','MiscVal',
    'MSSubClass', 'RoofStyle', 'Foundation','BsmtFinSF2', 'BsmtFinType2', 'LowQualFinSF',
    'BsmtHalfBath','BsmtFullBath','GarageYrBlt'], axis=1, inplace=True)
    # make the yearbuild a range of years
    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)
    # make the YearRemodAdd a range of years
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1920, 1940, 1960, 1980, 2000, 2020]
    #df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)
    # change roof matirial
    #df = fix_conditions(df, 'RoofMatl', 'WdShngl')
    # group MasVnrArea by ranges
    labels = [0, 1, 2]
    bins = [0, 1, 400, 2000]
    dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0, 'Mn':2 , 'No':1,'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2,'Unf': 1,'Av':3}
    # dealing with basement
    #df.MasVnrArea = pd.cut(df.MasVnrArea, bins, labels=labels, include_lowest=True)
    df.ExterQual = df.ExterQual.map(dict)
    df.ExterCond = df.ExterCond.map(dict)
    df.BsmtQual = df.BsmtQual.map(dict)
    df.BsmtCond = df.BsmtCond.map(dict)
    df.BsmtExposure = df.BsmtExposure.map(dict)
    df.BsmtFinType1 = df.BsmtFinType1.map(dict)
    labels = [0, 1, 2, 3, 4]
    bins = [0, 1, 500, 1000, 1500, 3000]
    df.BsmtFinSF1 = pd.cut(df.BsmtFinSF1, bins, labels=labels, include_lowest=True)
    df.BsmtUnfSF = pd.cut(df.BsmtUnfSF, bins, labels=labels, include_lowest=True)
    # heating engeeniring

    df.HeatingQC = df.HeatingQC.map(dict)
    #dealing with inside sizes
    df['2ndFlrSF'] = pd.cut(df['2ndFlrSF'], bins, labels=labels, include_lowest=True)
    df = fix_conditions(df, 'KitchenAbvGr', 1)
    df.KitchenQual = df.KitchenQual.map(dict)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.map({2:2,5:4, 3:4,4:4, 6:6, 7:7, 8:8, 9:9, 11:11 , 12:11, 10:11, 14:14,13:13,15:15,1:1})
    #df = fix_conditions(df, 'Functional', 'Maj2')
    df.FireplaceQu = df.FireplaceQu.map(dict)
    # dealing with Garage
    df.GarageFinish = df.GarageFinish.map({'NA':0, 'Fin':3, 'RFn':2, 'Unf':1})
    df.GarageQual = df.GarageQual.map(dict)
    df.GarageCond = df.GarageCond.map(dict)
    #other
    df.PavedDrive = df.PavedDrive.map({'N': 0,  'Y': 2, 'P': 1})
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1, 100, 200, 300, 400, 1000]
    df.WoodDeckSF = pd.cut(df.WoodDeckSF, bins, labels=labels, include_lowest=True)
    labels = [0, 1, 2, 3, ]
    bins = [0, 1, 100, 200, 1000]
    df.OpenPorchSF = pd.cut(df.OpenPorchSF, bins, labels=labels, include_lowest=True)
    return df

def normal(train, test):
    norm = MinMaxScaler()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test


class AveragingModels:
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):

        # Train cloned base models
        for model in self.models:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models ])
        return np.mean(predictions, axis=1)



if __name__ == '__main__':
    # load data
    train_data = pd.read_csv(r'House.csv')
    test_data = pd.read_csv(r'TestHouse.csv')
    # preproccesing
    Outliers_to_drop = detect_outliers(train_data,4)
    train_data.drop(Outliers_to_drop,inplace=True)
    # visual_data(train_data)  # visualizition
    id_test = test_data['Id']  # save Id for later to submit prediction
    y = train_data['SalePrice']
    df = pd.concat([train_data, test_data]).reset_index(drop=True)  # marge both train and test
    df = df.drop(['Id', 'SalePrice'], axis=1)  # drop useless columns
    df = fill_na(df)  # fill all the nulls after viewing the data
    df = features_engineering(df)  # change the features to better
    df = df.apply(lambda x: np.log1p(x) if (is_numeric_dtype(x) and x.skew() > 1 ) else x)
    y = np.log1p(y)
    print('after skew ', abs(df.skew()).mean() )
    df = pd.get_dummies(df)
    train = df.iloc[:train_data.shape[0], :]
    test = df.iloc[train_data.shape[0]:, :]
    test = test.reset_index(drop=True)
    #noramlize data
    train, test = normal(train, test)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,random_state=42)

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11,random_state=42)
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005,random_state=42))

    models = [ lasso, GBoost, model_lgb, RandomForestRegressor()]
    estimators = []
    for model in models:
        model_name = type(model).__name__
        estimators.append((model_name, model))


    # reg = StackingRegressor( estimators = estimators, final_estimator = RandomForestRegressor())
    # y_pred = reg.fit(train, y).predict(test)
    # #
    # # y_pred = av.predict(test)
    av = AveragingModels(models)
    av.fit(train, y)
    y_pred = av.predict(test)
    y_pred = np.exp(y_pred)
    submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})

    submission.to_csv('submission.csv', index=False)

    submission = pd.read_csv('submission.csv')

    print(submission)