import pandas as pd
import numpy as np
from random import random

from SGD import SGD_Ridge

SHARE = 0.2
FILENAME = 'synchronous machine.csv'


def split_data(data):
    # массив True и False
    split = np.array([random() < SHARE for _ in range(len(data))])
    # "~" - возвращает значение наоборот (False -> True, True -> False)
    return data[split], data[~split]


X = pd.read_csv(FILENAME, sep=';').to_numpy()

number_of_attributes = len(X[0]) - 1

# разделение датасета на обучающую и тестовую выборки
X_train, X_test = split_data(X)

X_test, y_test = np.split(X_test, [number_of_attributes], axis=1)
X_train, y_train = np.split(X_train, [number_of_attributes], axis=1)

weight = np.random.random(len(X_train[0]))  # вектор весов

w0 = -1  # фиктивный признак

sgd_ridge = SGD_Ridge(X_train=X_train, X_test=X_test, y_train=y_train,
                      y_test=y_test, weight=weight, w0=w0)


sgd_ridge.start()

sgd_ridge.plot()
