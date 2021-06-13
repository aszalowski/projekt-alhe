import xgboost as xgb
import pandas as pd

from XGBoostTuner import XGBoostTuner

if __name__ == "__main__":
    data = pd.read_csv('../data/winequality-white.csv', sep=';')
    data['type'] = [0 if x < 7 else 1 for x in data['quality']]

    X = data.drop(['quality', 'type'], axis=1)
    y = data['type']
    # X, y = data.iloc[:,:-1],data.iloc[:,-1]
    
    print(y.unique())

    xgb_tuner = XGBoostTuner(X, y)
    xgb_tuner.run()
