import pandas as pd 
import numpy as np


def read_csv():
    df = pd.read_csv('house_prices_train.csv')
    return df


def new_df():
    df = read_csv()
    df1 = df[['SalePrice', 'GarageArea', 'LotArea', 'TotalBsmtSF', 'GrLivArea', '2ndFlrSF', '1stFlrSF']]
    return df1


def drop_duplicates():
    df1 = new_df()
    df1 = df1.drop_duplicates()
    return df1


def save_new_df():
    df1 = drop_duplicates()
    df1.to_csv('house_prices_clean.csv')


def read_new_df():
    save_new_df()
    df_clean = pd.read_csv('house_prices_clean.csv')
    return df_clean


def outlier(df):
    q1 = np.percentile(df,25)
    q3 = np.percentile(df,75)
    
    iqr = q3-q1
    
    min_range = q1 - iqr*1.5
    max_range = q3 + iqr*1.5
    
    outliers = df[(df<min_range) | (df>max_range)]
    return outliers
