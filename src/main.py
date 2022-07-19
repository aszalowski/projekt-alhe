import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing

from XGBoostTuner import XGBoostTuner
from preprocessing import processWineData

def compare_local_global():
    history = []
    for i in range(10):
        filename = '../data/winequality-white.csv'
        X, y = processWineData(filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        xg_clas = xgb.XGBClassifier(use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        no_param_fitness_score = f1_score(y_test, preds)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='local')
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        local_fitness_score = f1_score(y_test, preds)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='global')
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        global_fitness_score = f1_score(y_test, preds)


        history.append((i, no_param_fitness_score, local_fitness_score, global_fitness_score))
        df = pd.DataFrame(history)  
        df.to_excel("compare.xlsx")

        print()
        print()

def compare_crossover():
    history = []
    for i in range(10):
        filename = '../data/winequality-white.csv'
        X, y = processWineData(filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='global', crossover_prob=0.2)
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        fitness_score_02 = f1_score(y_test, preds)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='global', crossover_prob=0.7)
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        fitness_score_07 = f1_score(y_test, preds)


        history.append((i, fitness_score_02, fitness_score_07))
        df = pd.DataFrame(history)  
        df.to_excel("compare.xlsx")

        print(i)
        print()

def compare_mutation():
    history = []
    for i in range(10):
        filename = '../data/winequality-white.csv'
        X, y = processWineData(filename)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='global', mutation_prob=0.2)
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        fitness_score_02 = f1_score(y_test, preds)

        xgb_tuner = XGBoostTuner(X_train, y_train, variation='global', mutation_prob=0.7)
        xgb_tuner.run(10)
        params = xgb_tuner.getBestParams()

        xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        fitness_score_07 = f1_score(y_test, preds)


        history.append((i, fitness_score_02, fitness_score_07))
        df = pd.DataFrame(history)  
        df.to_excel("compare_mut.xlsx")

        print(i)
        print()


if __name__ == "__main__":
    filename = '../data/winequality-white.csv'
    X, y = processWineData(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xg_clas = xgb.XGBClassifier(use_label_encoder=False)
    xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
    preds = xg_clas.predict(X_test)
    fitness_score = f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"Base fitness score: {fitness_score}")
    print(f"Base accuracy score: {accuracy}")

    xgb_tuner = XGBoostTuner(X_train, y_train)
    xgb_tuner.run()
    params = xgb_tuner.getBestParams()

    xg_clas = xgb.XGBClassifier(**params, use_label_encoder=False)
    xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
    preds = xg_clas.predict(X_test)
    fitness_score = f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    
    print(f"Base fitness score: {fitness_score}")
    print(f"Base accuracy score: {accuracy}")




