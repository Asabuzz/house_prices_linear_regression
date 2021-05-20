from data_clean import read_new_df
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df_clean = read_new_df()

def create_train_test():
    X = df_clean[['GrLivArea']]
    y = df_clean[['SalePrice']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    return print(f'Train set: {X_train.shape} \nTest set: {X_test.shape}'), X_train, X_test, y_train, y_test

def train_model():
    message, X_train, X_test, y_train, y_test = create_train_test()
    model = LinearRegression()
    r = model.fit(X_train, y_train) # entrainement du modele
    return print(f'{model.score(X_train, y_train)}\n{model.score(X_test, y_test)}'), r

def predict_model():
    message, X_train, X_test, y_train, y_test = create_train_test()
    message_2, r = train_model()
    results= r.predict(X_test)
    prediction =[]
    for result in results:
        prediction.append(round(result[0]))
    target = []
    for price in y_test['SalePrice']:
        target.append(price)
    compare = pd.DataFrame(list(zip(target, prediction)), columns = ['Reals_values', 'Prédictives_values'])
    return compare