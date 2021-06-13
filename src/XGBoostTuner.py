from sklearn.model_selection import train_test_split
from Phenotype import Phenotype
import random

POPULATION_SIZE = 5
TEST_SIZE = 0.2
CROSSOVER_PROB = 0.5
PHENOTYPE_MUTATION_PROB = 0.05
PARAMETER_MUTATION_PROB = 0.2

class XGBoostTuner:
    """
    Defines evolutionary algorithm - its selection, mutation and crossover operators 

    :params X_train and y_train: data used for training XGBoost Classifier
    :params X_test and y_test: data used for rating each phenotype
    :param fitness_history: stores best fitness_score in each itaration
    :param population: list of phenotypes
    """
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
        self.fitness_history = []

    def run(self, iterations=5):
        self.generate_population()
        self.rating()
        for i in range(iterations):
            print (self.fitness_history)
            #self.selection()
            self.crossover()
            self.mutation()
            self.rating()

    def selection(self):
        pass


    def generate_population(self):
        self.population = []
        for x in range(POPULATION_SIZE):
            self.population.append(Phenotype())

    def crossover(self):
        for phenotype in self.population:
            if random.uniform(0, 1) < CROSSOVER_PROB:
                second_parent = self.population[random.randint(0, POPULATION_SIZE - 1)]
                phenotype.crossover(second_parent)

    def mutation(self):
        for phenotype in self.population:
            if random.uniform(0, 1) < PHENOTYPE_MUTATION_PROB:
                phenotype.mutate(PARAMETER_MUTATION_PROB)

    def rating(self):
        best_fitness = 0
        for phenotype in self.population:
            phenotype.calculate_fitness(self.X_train, self.X_test, self.y_train, self.y_test)
            if phenotype.fitness_score > best_fitness:
                best_fitness = phenotype.fitness_score
        self.fitness_history.append(best_fitness)
