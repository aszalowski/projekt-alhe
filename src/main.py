import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/winequality-white.csv', sep=';')

X, y = data.iloc[:,:-1],data.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_clas = xgb.XGBClassifier()
xg_clas.fit(X_train,y_train)
preds = xg_clas.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
f1 = f1_score(y_test, preds, average="macro")
print("MSE: %f" % (rmse))
print (f"F1 score: {f1}")

