import xgboost as xgb
import pandas as pd

from XGBoostTuner import XGBoostTuner
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":
    data = pd.read_csv('../data/winequality-white.csv', sep=';')
    # data_red = pd.read_csv('../data/winequality-red.csv', sep=';')
    # data = pd.concat([data, data_red])
    data['type'] = [0 if x < 7 else 1 for x in data['quality']]

    X = data.drop(['quality', 'type'], axis=1)
    y = data['type']
    # X, y = data.iloc[:,:-1],data.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    xg_clas = xgb.XGBClassifier(use_label_encoder=False)
    xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
    preds = xg_clas.predict(X_test)
    fitness_score = f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"Base fitness score: {fitness_score}")
    print(f"Base accuracy score: {accuracy}")

    xgb_tuner = XGBoostTuner(X, y)
    xgb_tuner.run()
