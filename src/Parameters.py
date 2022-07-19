from abc import ABC, abstractmethod
import random
import numpy


class Parameter(ABC):
    """
    Defines interface for Parameters used by tuner.

    :param name: name of parameters
    :param bounds: tuple of lower and upper bound of paramater
    :param mutationRate: positive number, random mutation value will be between [-mutationRate, mutationRate]
    :param defaultValue: if set initial value will not be random but instead set to it

    """
    def __init__(self,  name, bounds, mutationRate=None, defaultValue=None):

        # Sanity checks
        assert len(bounds) == 2 and bounds[0] < bounds[1]
        assert name

        self.name = name
        self.bounds = bounds
        self.mutationRate = mutationRate

        self.value = self.initializeValue(defaultValue)

    def initializeValue(self, defaultValue):
        if defaultValue:
            if defaultValue >= self.bounds[0] and defaultValue <= self.bounds[1]:
                return defaultValue
            else:
                raise Exception("Wrong default value used in param.")
        else:
            return self.randomInitialValue()

    def __repr__(self):
        return f'Parameter(name={self.name}, value={self.value}, bounds={self.bounds}, mutationRate={self.mutationRate})'


    """
    Add a random value based on mutationRate to self.value checking if we are still in bounds
    """
    def mutate(self):
        if self.mutationRate:
            mutationValue = self.randomMutationValue()

            self.value += mutationValue

            # Check bounds
            self.value = self.bounds[0] if self.value < self.bounds[0] else self.value
            self.value = self.bounds[1] if self.value > self.bounds[1] else self.value
        else:
            self.randomInitialValue()


    """
    Dervied classes must implement these methods.
    """
    @abstractmethod
    def randomInitialValue(self):
        raise NotImplementedError

    @abstractmethod
    def randomMutationValue(self):
        raise NotImplementedError


class UniformParameter(Parameter):
    def __init__(self, name, bounds, mutationRate, defaultValue=None):
        super().__init__(name, bounds, mutationRate, defaultValue)

    def __repr__(self):
        return 'Uniform' + super().__repr__()

    def randomInitialValue(self):
        return random.uniform(self.bounds[0], self.bounds[1])

    def randomMutationValue(self):
        return numpy.random.normal(0, (self.bounds[1] - self.bounds[0]) * self.mutationRate)


class DiscreteParameter(Parameter):
    def __init__(self, name, bounds, mutationRate, defaultValue=None):
        super().__init__(name, bounds, mutationRate, defaultValue)

    def __repr__(self):
        return 'Discrete' + super().__repr__()

    def randomInitialValue(self):
        return random.randint(self.bounds[0], self.bounds[1])

    def randomMutationValue(self):
        return int(round(numpy.random.normal(0, (self.bounds[1] - self.bounds[0]) * self.mutationRate)))







