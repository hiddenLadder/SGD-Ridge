from random import random, randint
import numpy as np
import matplotlib.pyplot as plt


class SGD_Ridge:
    def __init__(self, X_train, X_test, y_train, y_test, weight):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.weight = weight
        self.X_train_length = len(self.X_train)
        self.X_test_length = len(self.X_test)

    def predict(self, i, test=False):  # ответ для i элемента
        return (self.X_test[i] if test else self.X_train[i]) @ self.weight

    def loss(self, i):  # потеря на i элементе
        return (self.predict(i) - self.y_train[i][0]) ** 2

    def quality(self):  # функционал качества
        return sum(self.loss(i) for i in range(self.X_train_length))

    def grad(self, i):
        return (self.X_train[i] @ self.weight - self.y_train[i][0]) * self.X_train[i]

    def start(self):
        curr_Q = self.quality()  # первый функционал
        eps = 0.000000001  # критерий остановки
        h = 1 / self.X_train_length  # шаг
        t = 0.00001  # параметр регуляризации тау
        k = 0
        it = 0

        while k < 10:
            it += 1
            i = np.random.choice(self.X_train.shape[0], 1, replace=False)[0]
            loss = self.loss(i)
            self.weight = (1 - (h * t)) * self.weight - h * self.grad(i)
            tmp_Q = loss * h + (1 - h) * curr_Q
            if abs(tmp_Q - curr_Q) < eps:
                k += 1
            else:
                k = 0
            print(it, " --- ", abs(tmp_Q - curr_Q))
            curr_Q = tmp_Q

        print(f'w: {self.weight}')
        print(f't: {t}')

    def plot(self):
        Y_control = []
        for i in range(self.X_test_length):
            Y_control.append(self.predict(i, test=True))
        plt.scatter(self.y_test, Y_control, c='r')
        plt.scatter(Y_control, self.y_test, c='b')
        plt.title(
            'Соотношение истинных и предсказанных ответов на тестовой выборке')
        plt.xlabel('Истинные ответы')
        plt.ylabel('Предсказанные ответы')
        plt.legend(['Истинный ответ', 'Предсказанный ответ'])
        plt.show()
