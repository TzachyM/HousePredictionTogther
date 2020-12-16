import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def import_data():
    train = pd.read_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\train.csv')
    test = pd.read_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\test.csv')
    return train, test


def feat_engineering():
    train, test = import_data()


def binary_fix(df, col, value):
    df.loc[df[col] == value, col] = 0
    df.loc[df[col] != 0, col] = 1
    return df


def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for name in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']:
        df[name] = df[name].fillna('NA')
    df.fillna(df.mode().iloc[0], inplace=True)

    return df

def outlier_remove(df):
    df.drop(df[df.BsmtFinSF1 > 3000], inplace=True, axis=0)

def visual(df):
    # view year build with salePrice

    df.plot.scatter(x='YearBuilt', y='SalePrice', xlim=(1880, 2020))
    # view RoofStyle with salePrice
    plt.figure()
    sns.boxplot(x='RoofStyle', y="SalePrice", data=df)
    # view RoofMatl  with salePrice
    plt.figure()
    sns.boxplot(x='RoofMatl', y="SalePrice", data=df)
    # view MasVnrArea with salePrice
    df.plot.scatter(x='MasVnrArea', y='SalePrice')
    # view MasVnrArea with salePrice
    df.plot.scatter(x='BsmtFinSF1', y='SalePrice')
    df.plot.scatter(x='TotalBsmtSF', y='SalePrice')

if __name__ == "__main__":
    # train,test = feat_engineering()

    train, test = import_data()
    #visual(train)
    #train = outlier_remove(train)
    id_test = test.Id
    y_train = train.SalePrice
    df = pd.concat([train, test]).reset_index(drop=True).drop(['Id', 'SalePrice'], axis=1)
    # Condition feat
    df = binary_fix(df, 'Condition1', 'Norm')
    df = binary_fix(df, 'Condition2', 'Norm')
    df['Condition'] = df.Condition1 + df.Condition2

    #df.drop(['Condition1', 'Condition2', 'Utilities','BsmtFinSF2', 'BsmtFinType2', 'LowQualFinSF','BsmtFullBath','BsmtHalfBath','HalfBath'], axis=1, inplace=True)  # 'MSSubClass','Foundation','RoofStyle
    #df = fill_na(df)

    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)

    labels = [0, 1, 2, 3, 4]
    bins = [0, 1950, 1970, 1990, 2000, 2020]
    df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)  # maybe remove

    df = binary_fix(df, 'RoofMatl', 'WdShngl')

    labels = [0, 1, 2]
    bins = [0, 1, 400, 2000]
    df.MasVnrArea = pd.cut(df.MasVnrArea, bins, labels=labels, include_lowest=True)#maybe leave it numerical
    value_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Na': 0, 'Av': 3, 'Mn': 2, 'No': 1, 'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
    df.ExterQual = df.ExterQual.map(value_dict)
    df.ExterCond = df.ExterCond.map(value_dict)
    df.BsmtQual = df.BsmtQual.map(value_dict)
    df.BsmtCond = df.BsmtCond.map(value_dict)
    df.BsmtExposure = df.BsmtExposure.map(value_dict)
    df.BsmtFinType1 = df.BsmtFinType1.map(value_dict)
    labels = [0, 1, 2, 3, 4]
    bins = [0, 1, 500, 1000, 1500, 3000]
    df.BsmtFinSF1 = pd.cut(df.BsmtFinSF1, bins, labels=labels, include_lowest=True)
    df.BsmtUnfSF = pd.cut(df.BsmtUnfSF, bins, labels=labels, include_lowest=True)
    df.loc[(df.Heating == 'GasA') | (df.Heating == 'GasW'), 'Heating'] = 1
    df.loc[df.Heating != 1, 'Heating'] = 0
    df.HeatingQC = df.HeatingQC.map(value_dict)
    df['TotalFlr'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['2ndFlrSF'] = pd.cut(df['2ndFlrSF'], bins, labels=labels, include_lowest=True)
    df = binary_fix(df, 'KitchenAbvGr', 1)
    df.KitchenQual = df.KitchenQual.map(value_dict)
    df.BsmtFinSF1 = pd.cut(df.BsmtFinSF1, bins, labels=labels, include_lowest=True)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.map({1: 1, 2: 2, 3: 4, 4: 4, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 11,
                                           12: 11, 13: 13, 14: 14, 15: 15})
    df.Functional = binary_fix(df, 'Functional', 'Maj2')
