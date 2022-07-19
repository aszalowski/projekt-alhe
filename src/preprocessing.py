import pandas as pd

def processWineData(filename):
    data = pd.read_csv(filename, sep=';')

    data['type'] = [0 if x < 7 else 1 for x in data['quality']]

    X = data.drop(['quality', 'type'], axis=1)
    y = data['type']

    return X, y

