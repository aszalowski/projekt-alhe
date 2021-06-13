from sklearn.model_selection import train_test_split
from Phenotype import Phenotype
from Roulette import Roulette, RankSelection
import random

POPULATION_SIZE = 100
TEST_SIZE = 0.2
CROSSOVER_PROB = 1
PHENOTYPE_MUTATION_PROB = 1
PARAMETER_MUTATION_PROB = 0.1

class XGBoostTuner:
    """
    Defines evolutionary algorithm - its selection, mutation and crossover operators 

    :params X_train and y_train: data used for training XGBoost Classifier
    :params X_test and y_test: data used for rating each phenotype
    :param fitness_history: stores best fitness_score in each itaration
    :param population: list of phenotypes
    """
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=10)
        self.fitness_history = []
        self.population = []

    def run(self, iterations=10):
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


    def generate_population(self):
        for _ in range(POPULATION_SIZE):
            self.population.append(Phenotype())

    def crossover(self):
        new_population = []

        self.sortPopulation()
        roulette = RankSelection(self.population)
        while len(new_population) != POPULATION_SIZE:
            if random.uniform(0, 1) < CROSSOVER_PROB:
                first_parent_index, second_parent_index = roulette.chooseIndexes()
                first_parent = self.population[first_parent_index]
                second_parent = self.population[second_parent_index]
                new_population.append(Phenotype.crossover(first_parent, second_parent))
            else:
                new_population.append(self.population[roulette.chooseIndex()])
        self.population = new_population

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
