from sklearn.model_selection import train_test_split
import random

from Phenotype import Phenotype
from Roulette import Roulette, RankSelection
from helpers import initializeXGBoostParameters

class XGBoostTuner:
    """
    Defines evolutionary algorithm - its selection, mutation and crossover operators 

    :params X_train and y_train: data used for training XGBoost Classifier
    :params X_test and y_test: data used for rating each phenotype
    :param fitness_history: stores best fitness_score in each itaration
    :param population: list of phenotypes
    """
    def __init__(self, X, y, population_size = 100, test_size=0.2, crossover_prob=0.3, mutation_prob=0.5, variation="global", defaultParamValues=None):
        self.population_size = population_size
        self.test_size = test_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.variation = variation
        self.defaultParamValues = defaultParamValues
        self.X, self.y = X, y


        self.fitness_history = []
        self.population = []

    def run(self, iterations=30):
        self.generate_population()
        self.rating()
        for _ in range(iterations):
            print(self.fitness_history)
            self.crossover()
            self.mutation()
            self.rating()
            p = max(self.population, key=lambda x: x.fitness_score)
            print(f"Best phenotype: {p}")
        print(self.fitness_history)

    def sortPopulation(self):
        self.population.sort(key=lambda p: p.fitness_score)

    def getBestParams(self):
        p = max(self.population, key=lambda x: x.fitness_score)
        return p.get_params()


    def generate_population(self):
        for _ in range(self.population_size):
            self.population.append(Phenotype(lambda: initializeXGBoostParameters(self.variation, self.defaultParamValues)))

    def crossover(self):
        new_population = []

        self.sortPopulation()
        roulette = RankSelection(self.population)
        while len(new_population) != self.population_size:
            if random.uniform(0, 1) < self.crossover_prob:
                first_parent_index, second_parent_index = roulette.chooseIndexes()
                first_parent = self.population[first_parent_index]
                second_parent = self.population[second_parent_index]
                new_population.append(Phenotype.crossover(first_parent, second_parent))
            else:
                new_population.append(self.population[roulette.chooseIndex()])
        self.population = new_population

    def mutation(self):
        for phenotype in self.population:
            phenotype.mutate(self.mutation_prob)

    def rating(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size)
        best_fitness = 0
        for phenotype in self.population:
            phenotype.calculate_fitness(self.X_train, self.X_test, self.y_train, self.y_test)
            if phenotype.fitness_score > best_fitness:
                best_fitness = phenotype.fitness_score
        self.fitness_history.append(best_fitness)
