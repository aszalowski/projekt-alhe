from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import random

class Phenotype:
    """
    Class defines single phenotype - set of parameters which are optimized by tuner

    :param param_list: list of parameters
    :param fitness_score: calculated fitness score for current parameters
    """

    def __init__(self, paramIniFunction):
        self.paramIniFunction = paramIniFunction
        self.param_list = paramIniFunction()
        self.fitness_score = None
        self.accuracy = None

    def mutate(self, mutation_prob):
        for parameter in self.param_list:
            if random.uniform(0, 1) < mutation_prob:
                parameter.mutate()

    def calculate_fitness(self, X_train, X_test, y_train, y_test):
        xg_clas = xgb.XGBClassifier(**self.get_params(), use_label_encoder=False)
        xg_clas.fit(X_train, y_train, eval_metric = "mlogloss")                   
        preds = xg_clas.predict(X_test)
        self.fitness_score = f1_score(y_test, preds)
        self.accuracy = accuracy_score(y_test, preds)

    def get_params(self):
        params = {}
        for parameter in self.param_list:
            params[parameter.name] = parameter.value
        return params

    def __repr__(self):
        string = f'Phenotype[fitness={self.fitness_score}, acc={self.accuracy}](\n'
        for p in self.param_list:
            string += f'\t{p}\n'
        string += ')'
        return string

    @staticmethod
    def crossover(parent1, parent2, crossover_point=0.5):
        child = Phenotype(parent1.paramIniFunction)
        for (child_param, p1, p2) in zip(child.param_list, parent1.param_list, parent2.param_list):
            if random.uniform(0, 1) < crossover_point:
                child_param.value = p1.value
            else:
                child_param.value = p2.value
        return child 
