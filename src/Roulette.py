import numpy

class Roulette:
    def __init__(self, population):
        populationFitnessSum = sum(phenotype.fitness_score for phenotype in population)
        self.probabilites = [phenotype.fitness_score / populationFitnessSum for phenotype in population]

    def chooseIndexes(self):
        return numpy.random.choice(range(len(self.probabilites)), size=2, p=self.probabilites, replace=False)

    def chooseIndex(self):
        return numpy.random.choice(range(len(self.probabilites)), p=self.probabilites)

