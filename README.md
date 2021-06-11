# Projekt ALHE - dokumentacja wstępna
## Treść wybranego zadania
13. Optymalizacja parametrów algorytmu xgboost z wykorzystaniem algorytmów ewolucyjnych / genetycznych. Zadanie polega na wybraniu ciekawego zbioru do zagadnienia klasyfikacyjnego oraz wykorzystanie go do predykcji za pomocą algorytmu xgboost. Główną częścią zadania będzie optymalizacja hiperparametrów (algorytmu xgboost) z wykorzystaniem algorytmu ewolucyjnego / genetycznego.


## Analiza problemu i propozycja rozwiązania
Zadanie polega na implementacji algorytmu znajdującego parametry algorytmu xgboost, pozwalające na jak najskuteczniejszą predykcję. W tym celu należy wybrać zbiór danych, na którym algorytm xgboost będzie pracował. Zbiór ten zostanie podzielony na 2 części: trenującą i testującą. Nasz algorytm będzie podawał N zestawów parametrów dla algorytmu xgboost, który to na ich podstawie będzie trenował modele przy pomocy danych trenujących. Następnie wytrenowane modele zostaną poddane testom – nastąpi predykcja na zbiorze testowym. Na podstawie miary jakości osobników, algorytm będzie generował kolejne zestawy według typowego ewolucyjnego postępowania:
* Selekcja – dla każdego z N nowych zestawów wybieranych będzie K zestawów z poprzedniego uruchomienia za pomocą selekcji ruletkowej. Im lepsza jakość konkretnego zbioru parametrów, tym większa szansa na jego wybranie.
* Krzyżowanie – wybrane zestawy parametrów będą w ramach każdego nowego uśredniane.
* Mutacja – każdy z parametrów będzie poddawany pewnej mutacji z prawdopodobieństwem p. W zależności od tego, czy parametr może przyjmować wartości ciągłe czy dyskretne, mutacja będzie polegała na dodaniu wartości wygenerowanej przy pomocy rozkładu normalnego lub zmianie wartości dyskretnej na sąsiednią.
* Sukcesja - zastosujemy sukcesję generacyjną, a więc wszystkie wyznaczone zestawy będą brane pod uwagę podczas następnej serii uruchomień algorytmu xgboost.

Opisany proces będzie powtarzał się do momentu otrzymania satysfakcjonujących wyników lub po określonej liczbie iteracji.

## Przyjęte założenia

### Środowisko

Zadanie wykonane będzie przy użyciu języka Python3. Wykorzystamy implementacje algorytmu xgboost z biblioteki [xgboost](https://github.com/dmlc/xgboost).

### Parametry

Opis wszystkich parametrów, które przyjmuje wybrana implementacja algorytmu xgboost, można znaleźć [tutaj](https://xgboost.readthedocs.io/en/latest/parameter.html). Wybraliśmy następujące 9 parametrów, które algorytm genetyczny będzie starał się dobrać: `eta, gamma, max_depth, min_child_weight, max_delta_step, subsample, lambda, alpha, scale_pos_weight` .

### Funkcja oceny

W przypadku naszego zadania jako miarę jakości danego osobnika przyjmujemy wartość funkcji F1-score liczoną według następującego wzoru:

$$\large F_1 =  \frac{TP}{TP + \frac{1}{2}(FP + FN)}$$
gdzie $TP$ - true positives; $FP$ - false positives; $FN$ - false negatives.



### Wybrany zbiór danych

Wybraliśmy zbiór zawierający informację na temat białych win ze strony [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). Problem polega na klasyfikacji wina na podstawie 11 atrybutów na jedną z klas oznaczających jakość ( wartości pomiędzy 0 i 10 ). Istotnym aspektem zbioru jest niezbalansowanie klas co oznacza, że ważne jest odpowiednie ustawienie parametrów `subsample` oraz `scale_pos_weight`, w czym pomóc ma algorytm genetyczny.

## Badanie jakości rozwiązania

Zestaw parametrów dobrany za pomocą algorytmu genetycznego porównamy z domyślnym zestawem parametrów.


