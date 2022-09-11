from random import random, randint
import numpy as np
import matplotlib.pyplot as plt


class SGD_Ridge:
    def __init__(self, X_train, X_test, y_train, y_test, weight, w0):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.weight = weight
        self.w0 = w0
        self.X_train_length = len(self.X_train)

    def predict(self, i):  # ответ для i элемента
        return self.X_train[i] @ self.weight + self.w0

    def loss(self, i):  # потеря на i элементе
        return (self.predict(i) - self.y_train[i][0]) ** 2 / (2 * self.X_train_length)

    def quality(self):  # функционал качества
        return sum(self.loss(i) for i in range(self.X_train_length))

    def grad(self, i):
        return (self.X_train[i] @ self.weight + self.w0 - self.y_train[i][0]) * self.X_train[i]

    def start(self):
        curr_Q = self.quality()  # первый функционал
        eps = 0.000000001  # критерий остановки
        h = 1 / self.X_train_length  # шаг
        t = 0.0001  # параметр регуляризации тау
        k = 0
        it = 0

        while k < 10:
            it += 1
            i = np.random.choice(self.X_train.shape[0], 1, replace=False)[0]
            loss = self.loss(i)
            self.weight = self.weight * (1 - h * t) - h * self.grad(i)
            self.w0 = self.w0 - h * loss
            tmp_Q = loss * h + (1 - h) * curr_Q
            if abs(tmp_Q - curr_Q) < eps:
                k += 1
            else:
                k = 0
            print(it, " --- ", abs(tmp_Q - curr_Q))
            curr_Q = tmp_Q

    def plot(self):
        x = np.linspace(1, self.X_train_length, self.X_train_length)
        plt.scatter(x, self.y_train, c='r')
        Y_control = []
        for i in range(self.X_train_length):
            Y_control.append(self.predict(i))
        plt.scatter(x, Y_control, c='b')
        plt.title(
            'Соотношение истинных и предсказанных ответов на тестовой выборке')
        plt.xlabel('Порядковый номер элемента выборки')
        plt.ylabel('Значение ответа')
        plt.legend(['Истинный ответ', 'Предсказанный ответ'])
        plt.show()
