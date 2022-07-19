import numpy

class Roulette:
    def __init__(self, population):
        print(population)
        populationFitnessSum = sum(phenotype.fitness_score for phenotype in population)
        self.probabilities = [phenotype.fitness_score / populationFitnessSum for phenotype in population]
        minProbability = min(self.probabilities)

        self.probabiliteis = [probability - minProbability for probability in self.probabilities]
        # print(self.probabilities)

    def chooseIndexes(self):
        return numpy.random.choice(range(len(self.probabilities)), size=2, p=self.probabilities, replace=False)

    def chooseIndex(self):
        return numpy.random.choice(range(len(self.probabilities)), p=self.probabilities)

class RankSelection:
    def __init__(self, population):
        # print(self.probabilities)
        sumOfRanks = (1 + len(population))*len(population) / 2

        self.probabilities = [(i + 1) / sumOfRanks for i in range(len(population))]

    def chooseIndexes(self):
        return numpy.random.choice(range(len(self.probabilities)), size=2, p=self.probabilities, replace=False)

    def chooseIndex(self):
        return numpy.random.choice(range(len(self.probabilities)), p=self.probabilities)

