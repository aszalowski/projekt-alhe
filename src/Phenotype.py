from helpers import initializeXGBoostParameters
from Parameters import DiscreteParameter
from sklearn.metrics import f1_score
import xgboost as xgb
import random

class Phenotype:
    """
    Class defines single phenotype - set of parameters which are optimized by tuner

    :param param_list: list of parameters
    :param fitness_score: calculated fitness score for current parameters
    """

    def __init__(self):
        self.param_list = initializeXGBoostParameters()

    def mutate(self, mutation_prob):
        for parameter in self.param_list:
            if random.uniform(0, 1) < mutation_prob:
                parameter.mutate()

    def calculate_fitness(self, X_train, X_test, y_train, y_test):
        xg_clas = xgb.XGBClassifier(self.get_params())  #use_label_encoder = False
        xg_clas.fit(X_train, y_train)                   #eval_metric = "mlogloss"
        preds = xg_clas.predict(X_test)
        self.fitness_score = f1_score(y_test, preds, average="micro")
        print(self.param_list)
        print(self.fitness_score)

    def crossover(self, second_parent, crossover_point=0.5):
        for (p1, p2) in zip(self.param_list, second_parent.param_list):
            p1.value = crossover_point * p1.value + (1 - crossover_point) * p2.value
            if isinstance(p1, DiscreteParameter):
                p1.value = int(round(p1.value))

    def get_params(self):
        params = {}
        for parameter in self.param_list:
            params[parameter.name] = parameter.value
        return params