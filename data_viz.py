import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from data_clean import read_new_df
from data_clean import outlier

df_clean =  read_new_df()


def df_infos():
    describe = df_clean.describe()
    return describe


def plot_variables():  
    sns.set()
    sns.set_palette("Paired")
    variables_list = ['SalePrice', 'GarageArea', 'LotArea', 'TotalBsmtSF', 'GrLivArea',
       '2ndFlrSF', '1stFlrSF']  
    fig, axes = plt.subplots(nrows=len(variables_list), ncols=2, figsize = (14, 20), constrained_layout=True)
    for i, variable in enumerate(variables_list): 
        sns.histplot(data = df_clean, x=variable, kde = True,   ax=axes[i,0])
        sns.boxplot(data = df_clean, x=variable, ax=axes[i,1])
    fig.suptitle('Histogrammes et boxplots des variables numériques', size=20)
    return plt.show()


def count_outliers(df):
    for col in df.columns:
        outliers = outlier(df[col])
        if len(outliers):
            print(f"* La colonne {col} à {(outliers.count()/1459*100).round(3)}% d'outliers")
        else:
            print(f"* {col} n'a pas d'outliers.")


def display_pairplot(df):
    return sns.pairplot(df)
    

def display_corr_matrix():
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    return sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .75}, annot=True).set_title('Matrice de corrélation', size=20)


def display_regplot():
    items = ['GrLivArea', 'GarageArea']
    fig, axes = plt.subplots(nrows=1,ncols=2, figsize = (14,6))
    fig.suptitle('Analyse des regplots selon le prix de vente', fontsize= 20,)
    for item, axe in zip(items, axes.flat):
        axe.set_title(f"Prix de vente en fonction de {item}")   
        g = sns.regplot(data=df_clean, x=item, y="SalePrice", ax = axe)
    return g