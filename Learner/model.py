import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


class Model():

    def __init__(self, data, settings):

        self.settings = settings
        self.data = data
        self.plot_initialise()

    def train(self):

        self.w1 = np.random.uniform(-1, 1, (self.data.words_num, self.settings.hid_lay_nodes))
        self.w2 = np.random.uniform(-1, 1, (self.settings.hid_lay_nodes, self.data.words_num))

        loss = []
        # Training through the entire dataset epoch number of times.
        for i in range(self.settings.epoch):
            avg_loss = 0
            for target, context in self.data.training_data:

                target = target.reshape(self.data.words_num, 1)
                context = context.reshape(self.data.words_num, 1)

                """Forward Propagation"""

                y, y_out, h = self.forward_prop(target)  # y = The probabilities or softmax output. # y_out = The output. # h = Intermediate output.

                """Back Propagation"""
                self.back_prop(target, context, y, y_out,h)

                self.find_loss(y_out, context)
                avg_loss += self.loss

            loss.append(avg_loss / self.data.m)

        self.plot(loss)

    def forward_prop(self, x):

        h = np.dot(self.w1.T, x)
        y_out = np.dot(self.w2.T, h)
        y = self.softmax(y_out)

        return y, y_out, h

    def back_prop(self, x, y_train, y, y_out, h):

        C = np.sum(y_train)
        e = C*y - y_train

        del_w2 = np.dot(h, e.T)
        del_w1 = np.dot(x, np.dot(self.w2, e).T)

        self.w1 = self.w1 - self.settings.learning_rate * del_w1
        self.w2 = self.w2 - self.settings.learning_rate * del_w2

    def softmax(self, y):

        y_temp = np.exp(y - np.max(y))
        y_pred = y_temp / np.sum(y_temp)

        return y_pred

    def find_loss(self, y_out, y_train):

        C = np.sum(y_train)
        temp1 = sum(np.multiply(y_out, y_train))
        temp2 = np.exp(y_out)
        temp2 = C * np.log(np.sum(temp2))

        self.loss = -temp1 + temp2

    def plot_initialise(self):

        style.use('dark_background')
        plt.figure()
        plt.ion()
        plt.xlabel('Number of iterations')
        plt.ylabel('Loss')

    def plot(self, loss):

        x = range(1, self.settings.epoch + 1)
        plt.plot(x, loss)
        plt.show(block = True)
















