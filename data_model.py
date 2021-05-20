from data_clean import read_new_df
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

df_clean = read_new_df()

def statsmodels_lr():
    X = df_clean[['GrLivArea']]
    y = df_clean[['SalePrice']]
    model = smf.ols('y ~ X', data = df_clean)
    results = model.fit()
    return results.summary()

def sklearn_lr():
    X = df_clean[['GrLivArea']]
    y = df_clean[['SalePrice']]
    model = LinearRegression()
    model_fit = model.fit(X, y) # entrainement du modele
    return model.score(X, y), model_fit # évaluation avec le coefficient de corrélation

def sklearn_pred(p2):
    a, model_fit = sklearn_lr()
    prediction = [[p2]]
    result = model_fit.predict(prediction)
    return print(f'Le prix predictif pour une surface de {prediction[0][0]} pieds carrés de l\'appartement est de {result[0][0].round(3)} $')