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


def condition_fix(df, col):
    df.loc[df[col] == 'Norm', col] = 0
    df.loc[df[col] != 0, col] = 1
    return df


def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)

    df.fillna(df.mode().iloc[0], inplace=True)
    return df


def visual(train):
    sns.lineplot(data=train, x="YearBuilt", y="SalePrice")
    plt.show()

if __name__ == "__main__":
    # train,test = feat_engineering()

    train, test = import_data()
    visual(train)
    sns.boxplot(x='RoofStyle', y="SalePrice", data=train)

    print('working?')
    id_test = test.Id
    y_train = train.SalePrice
    df = pd.concat([train, test]).reset_index(drop=True).drop(['Id', 'SalePrice'], axis=1)
    # Condition feat
    df = condition_fix(df, 'Condition1')
    df = condition_fix(df, 'Condition2')
    df['Condition'] = df.Condition1 + df.Condition2

    df.drop(['Condition1', 'Condition2', 'Utilities'], axis=1, inplace=True)  # 'MSSubClass'
    df = fill_na(df)

    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)

    labels = [0, 1, 2, 3, 4]
    bins = [0, 1950, 1970, 1990, 2000, 2020]
    df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)  # maybe remove
