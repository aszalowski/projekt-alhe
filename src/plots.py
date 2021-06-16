import seaborn as sns
import pandas as pd


filename = '../data/winequality-white.csv'
data = pd.read_csv(filename, sep=';')

bins = (0, 6, 10)
qual = ['niska', 'wysoka']
y = pd.cut(data['quality'], bins = bins, labels=qual)
plot = sns.countplot(y)
plot.set(xlabel="klasa wina", ylabel="ilość")
plot.figure.savefig("countplot.png")
