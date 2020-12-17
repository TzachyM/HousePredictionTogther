import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor

def import_data():
    train = pd.read_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\train.csv')
    test = pd.read_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\test.csv')
    return train, test

def data_prep():
    train, test = import_data()
    #Outlier removel
    train = outlier_remove(train)
    #Divding data
    id_test = test.Id
    y_train = train.SalePrice
    df = pd.concat([train, test]).reset_index(drop=True).drop(['Id', 'SalePrice'], axis=1)
    # Visual data
    #visual(train)
    #Feature engineering
    df = feat_engineering(df)
    train = df.iloc[:train.shape[0], :]
    test = df.iloc[train.shape[0]:, :]
    return train, test, y_train, id_test

def feat_engineering(df):
    #Fill NaN
    df = fill_na(df)
    #Features
    df = binary_fix(df, 'Condition1', 'Norm')
    df = binary_fix(df, 'Condition2', 'Norm')
    df['Condition'] = df.Condition1 + df.Condition2
    labels = [0, 1, 2, 3]
    bins = [0, 1900, 1950, 2000, 2020]
    df.YearBuilt = pd.cut(df.YearBuilt, bins, labels=labels, include_lowest=True)
    labels = [0, 1, 2, 3, 4]
    bins = [0, 1950, 1970, 1990, 2000, 2020]
    #df.YearRemodAdd = pd.cut(df.YearRemodAdd, bins, labels=labels, include_lowest=True)  # maybe remove
    df = binary_fix(df, 'RoofMatl', 'WdShngl')
    labels = [0, 1, 2]
    bins = [0, 1, 400, 2000]
    #df.MasVnrArea = pd.cut(df.MasVnrArea, bins, labels=labels, include_lowest=True) #maybe leave it numerical
    value_dict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Na': 0, 'Av': 3, 'Mn': 2, 'No': 1, 'GLQ': 6, 'ALQ': 5,
                  'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
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
    df.Heating = df.Heating.astype(int)
    df.HeatingQC = df.HeatingQC.map(value_dict)
    df['TotalFlr'] = df['1stFlrSF'] + df['2ndFlrSF']
    df['2ndFlrSF'] = pd.cut(df['2ndFlrSF'], bins, labels=labels, include_lowest=True)
    df = binary_fix(df, 'KitchenAbvGr', 1)
    df.KitchenQual = df.KitchenQual.map(value_dict)
    df.TotRmsAbvGrd = df.TotRmsAbvGrd.map({1: 1, 2: 2, 3: 4, 4: 4, 5: 4, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11, 11: 11,
                                           12: 11, 13: 13, 14: 14, 15: 15})
    df.Functional = binary_fix(df, 'Functional', 'Maj2')
    df.FireplaceQu = df.FireplaceQu.map(value_dict)
    df.GarageFinish = df.GarageFinish.map({'Na': 0, 'Fin': 3, 'RFn': 2, 'Unf': 1})
    df.GarageQual = df.GarageQual.map(value_dict)
    df.GarageCond = df.GarageCond.map(value_dict)
    df.PavedDrive = df.PavedDrive.map({'Y': 2, 'P': 1, 'N': 0})
    labels = [0, 1, 2, 3, 4, 5]
    bins = [0, 1, 100, 200, 300, 400, 1000]
    df.WoodDeckSF = pd.cut(df.WoodDeckSF, bins, labels=labels, include_lowest=True)
    labels = [0, 1, 2, 3]
    bins = [0, 1, 100, 200, 1000]
    df.OpenPorchSF = pd.cut(df.OpenPorchSF, bins, labels=labels, include_lowest=True)
    #Columns drop
    df.drop(['Condition1', 'Condition2', 'Utilities','BsmtFinSF2', 'BsmtFinType2', 'LowQualFinSF','BsmtFullBath',
     'BsmtHalfBath', 'HalfBath', 'GarageYrBlt', 'GarageArea', 'EnclosedPorch','3SsnPorch', 'ScreenPorch', 'PoolArea',
     'PoolQC', 'Fence', 'MiscVal', 'MoSold', 'YrSold', 'MSSubClass', 'Foundation', 'RoofStyle'], axis=1, inplace=True)  # 'MSSubClass','Foundation','RoofStyle'
    #Skewness fix
    print("Skewness before fix:", df.skew().mean())
    df = df.apply(lambda x: np.log1p(x) if is_numeric_dtype(x) and np.abs(x.skew())>2 else x)
    print("Skewness after fix:", df.skew().mean())
    df = pd.get_dummies(df)
    return df

def normal(train, test):
    norm = MinMaxScaler()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test

def binary_fix(df, col, value):
    # df.loc[df[col] == value, col] = 0
    # df.loc[df[col] != 0, col] = 1
    # df[col] = df[col].astype(int)
    return df

def fill_na(df):
    df['Alley'] = df['Alley'].fillna('No')
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    for name in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'FireplaceQu', 'GarageType', 'GarageFinish',
                 'GarageQual','GarageCond', 'MiscFeature']:
        df[name] = df[name].fillna('Na')
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

def outlier_remove(df, iqr_step=1.3, n=3):
    outlier_index = []
    for col in df.columns:
        if(is_numeric_dtype(df[col])):
            q1 = np.percentile(df[col], 25)
            q3 = np.percentile(df[col], 75)
            iqr = q3-q1
            step = iqr_step*iqr
            outlier_col = df[(df[col]<q1-step) | (df[col]>q3+step)].index
            outlier_index.extend(outlier_col)
    outlier_indices = Counter(outlier_index)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    df.drop(multiple_outliers, inplace=True)
    return df

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
    # view BsmtFinSF1 with salePrice
    df.plot.scatter(x='BsmtFinSF1', y='SalePrice')
    # view TotalBsmtSF with salePrice
    df.plot.scatter(x='TotalBsmtSF', y='SalePrice')


if __name__ == "__main__":

    train, test, y_train, id_test = data_prep()
    # Normal the data
    train, test = normal(train, test)
    X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2, random_state=0)
    model = GradientBoostingRegressor(random_state=0, learning_rate=0.05, n_estimators=400)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    y_pred_test = model.predict(X_test)
    y_pred = model.predict(test)
    print(np.sqrt(mean_squared_log_error(np.abs(y_test), np.abs(y_pred_test))))

    submission = pd.DataFrame({'Id': id_test, 'SalePrice': y_pred})

    submission.to_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\test.submission.csv',
                      index=False)

    #submission = pd.read_csv(r'C:\Users\tzach\Dropbox\DC\Primrose\Excercies\Kaggle\House Price\test.submission.csv')

    #print(submission)



