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
        self.fitness_score = None

    def mutate(self, mutation_prob):
        for parameter in self.param_list:
            if random.uniform(0, 1) < mutation_prob:
                parameter.mutate()

    def calculate_fitness(self, X_train, X_test, y_train, y_test):
        xg_clas = xgb.XGBClassifier(**self.get_params(), use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        self.fitness_score = f1_score(y_test, preds, average="micro")

    def get_params(self):
        params = {}
        for parameter in self.param_list:
            params[parameter.name] = parameter.value
        return params

    def __repr__(self):
        string = f'Phenotype[fitness={self.fitness_score}](\n'
        for p in self.param_list:
            string += f'\t{p}\n'
        string += ')'
        return string

    @staticmethod
    def crossover(parent1, parent2, crossover_point=0.5):
        child = Phenotype()
        for (child_param, p1, p2) in zip(child.param_list, parent1.param_list, parent2.param_list):
            child_param.value = crossover_point * p1.value + (1 - crossover_point) * p2.value
            if isinstance(child_param, DiscreteParameter):
                child_param.value = int(round(p1.value))

        # print("Created new child.")
        # print(f'parent1: {parent1}')
        # print(f'parent2: {parent2}')
        # print(f'child: {child}')
        return child 
