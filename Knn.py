import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

data = genfromtxt('data/yeast3.csv', delimiter=',')  # pobieranie danych z pliku
X = np.array(data[:, 0: -2])
y = np.array(data[:, -1])  # dzielenie danych, y to ostatnia kolumna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=1)  # podział danych na czesc testowa i treningowa


def pred(X_train, y_train, x_test, k):  # funkcja do predykcji wartości klasy
    distances = []
    targets = []

    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))  # odległość od i-tej wartości
        distances.append([distance, i])

        distances = sorted(distances)  # sortowanie wyników

    for i in range(k):  # Określanie wartośći klasy k-sąsiadów
        index = distances[i][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]  # wartość która występuje najczęściej


def KNN(X_train, y_train, X_test, predictions, k):
    for i in range(len(X_test)):
        predictions.append(pred(X_train, y_train, X_test[i, :], k))


BadaneK = []
OsX = list(range(1, 30))

for k in range(1, 30):
    predictions = []
    KNN(X_train, y_train, X_test, predictions, k)  # wywołanie funkcji kNN
    BadaneK.append(1 - accuracy_score(y_test, predictions))  # wynik dokładności kNN

print("Optymalne k to :")  # wszystkie najniższe wyniki dla różnych k
a = np.array(BadaneK)
b = np.where(a == a.min())
print(b[0] + 1)

plt.plot(OsX, BadaneK)  # rysowanie wykresu
plt.xlabel('Wartość K')
plt.ylabel('Błąd')
plt.show()
