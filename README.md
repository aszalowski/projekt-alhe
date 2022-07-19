# Projekt ALHE

## Autorzy

- Adam Szałowski
- Daniel Chmielewiec

## Instalacja i uruchomienie

Kod źródłówy projektu znajduje się w repozytorium [GitLab](https://gitlab-stud.elka.pw.edu.pl/aszalows/projekt-alhe/). Wymagane moduły zapisane są w pliku *requirements.txt* i można zainstalować je przy użyciu komendy `pip install -r requirements.txt`

Przykład optymalizacji przy użyciu algorytmu genetycznego:
```python
from XGBoostTuner import XGBoostTuner
from preprocessing import processWineData

filename = '../data/winequality-white.csv'
X, y = processWineData(filename)

xgb_tuner = XGBoostTuner(X, y)
xgb_tuner.run()
params = xgb_tuner.getBestParams()
```

Skrypty z eksperymentami znajdują się w pliku *src/main.py*.

## Sprawozdanie

Sprawozdanie znajduję się w pliku sprawozdanie_ALHE.pdf. Tam też można znaleźć więcej informacji na temat implementacji oraz przeprowadzonych porównań.
