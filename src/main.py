import xgboost as xgb
import pandas as pd

from XGBoostTuner import XGBoostTuner

if __name__ == "__main__":
    data = pd.read_csv('../data/winequality-white.csv', sep=';')
    X, y = data.iloc[:,:-1],data.iloc[:,-1]

    xgb_tuner = XGBoostTuner(X, y)
    xgb_tuner.run()
