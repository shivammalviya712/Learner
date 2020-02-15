import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


class Model():

    def __init__(self, data, settings):

        self.settings = settings
        self.data = data
        self.X = self.data.X
        self.fun = np.vectorize(self.f)
        self.plot_initialise()

    def train(self):

        self.prepare()
        # Iterating through the entire dataset epoch number of times
        for i in range(0, self.settings.epoch):

            delta = self.fun(self.X) * (self.U @ self.V - np.log(self.X))
            U_grad = delta @ self.V.T
            V_grad = self.U.T @ delta

            U_grad[:, self.settings.d+1] = 0
            V_grad[self.settings.d+1, :] = 0

            # Updating U and V
            self.U = self.U - self.settings.a * U_grad
            self.V = self.V - self.settings.a * V_grad

            # Plotting Loss
            self.update_loss()
            self.plot(i)

    def prepare(self):

        self.U = np.random.uniform(-1, 1, (self.data.T, self.settings.d + 2))
        self.V = np.random.uniform(-1, 1, (self.settings.d + 2, self.data.T))

        for i in range(0, self.data.T):
            self.U[self.settings.d + 1, i] = 1
            self.V[i, self.settings.d + 1] = 1

    def f(self, x):

        if x > self.settings.xmax:
            return 1

        else:
            y = (x/self.settings.xmax)**self.settings.f_a
            return y

    def update_loss(self):

        J = self.fun(self.X) * ((self.U @ self.V - np.log(self.X))**2)
        self.J = J.sum() / 2

    def plot(self, i):

        # s - Marker size
        plt.scatter(i, self.J, color = 'red', s = 3)
        plt.show(block = False)
        plt.pause(0.001)

    def plot_initialise(self):

        style.use('dark_background')
        plt.figure()
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')

    def predict(self, word):

        index = self.data.word_index[word]
        w = self.U[index, :].T + self.V[:, index]
        return w[1:-1] / 2