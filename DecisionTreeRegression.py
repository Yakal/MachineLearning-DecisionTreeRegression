__author__ = 'Furkan Yakal'
__email__ = 'fyakal16@ku.edu.tr'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Read the data points as float
data = np.array(pd.read_csv("hw05_data_set.csv", header=None))[1:].astype(float)

x_data = data[:, 0]
y_data = data[:, 1]

# training set is formed from first 150 data points
x_train = x_data[0:150]  # x coordinates of the training data
y_train = y_data[0:150]  # y coordinates of the training data

# test set is formed from remaining data points
x_test = x_data[150:]  # x coordinates of the test data
y_test = y_data[150:]  # y coordinates of the test data
N = len(y_test)  # length of the test set

# hyperparameters
p = 25
pruning_array = np.arange(5, 55, 5)


# draws the plot
def draw_plot(x, y, title):
    plot.subplot()
    plot.scatter(x_train, y_train, alpha=0.6, c="blue", edgecolors='none', label="training")
    plot.scatter(x_test, y_test, alpha=0.6, c="red", edgecolors='none', label="test")
    plot.plot(x, y, c="black")
    plot.legend(loc=2)
    plot.xlabel('Eruption time (min)')
    plot.ylabel('Waiting time to next eruption (min)')
    plot.title(title)
    plot.show()


# root mean squared error
def rmse(y_head, y, length):
    return np.sqrt(np.sum((y - y_head) ** 2) / length)


class TreeRegression(object):
    def __init__(self):  # initialize data-structures
        self.node_indices = {1: np.arange(len(x_train))}
        self.is_terminal = {True: np.array([]), False: np.array([])}
        self.need_split = {True: np.array([1]), False: np.array([])}
        self.node_splits = {}
        self.terminal_node_means = {}

    def decision_tree_regression(self, pruning=p):  # decision tree regression
        while len(self.need_split[True]) is not 0:
            split_nodes = self.need_split[True]
            for split_node in split_nodes:
                self.need_split[True] = self.need_split[True][self.need_split[True] != split_node]
                self.need_split[False] = np.insert(self.need_split[False], len(self.need_split[False]), split_node)

                data_indices = self.node_indices[split_node]

                if len(data_indices) <= pruning:
                    self.terminal_node_means[split_node] = np.mean(y_train[data_indices])
                    self.is_terminal[True] = np.insert(self.is_terminal[True], len(self.is_terminal[True]), split_node)
                else:
                    self.is_terminal[False] = np.insert(self.is_terminal[False], len(self.is_terminal[False]),
                                                        split_node)

                    unique_values = np.sort(np.unique(x_train[data_indices]))
                    split_positions = (unique_values[0:len(unique_values) - 1] +
                                       unique_values[1:len(unique_values)]) / 2
                    scores = []

                    for sp in split_positions:
                        left_indices = data_indices[x_train[data_indices] <= sp]
                        right_indices = data_indices[x_train[data_indices] > sp]

                        left_mean = np.mean(y_train[left_indices])
                        right_mean = np.mean(y_train[right_indices])
                        scores.append(1 / len(data_indices) * (np.sum((y_train[left_indices] - left_mean) ** 2) +
                                                               np.sum((y_train[right_indices] - right_mean) ** 2)))

                    best_split = split_positions[np.argmin(scores)]

                    self.node_splits[split_node] = best_split

                    self.node_indices[2 * split_node] = data_indices[x_train[data_indices] <= best_split]
                    self.need_split[True] = np.insert(self.need_split[True], len(self.need_split[True]), 2 * split_node)

                    self.node_indices[(2 * split_node) + 1] = data_indices[x_train[data_indices] > best_split]
                    self.need_split[True] = np.insert(self.need_split[True], len(self.need_split[True]),
                                                      (2 * split_node) + 1)

    def fit_line(self):  # fits a line on the trained regression tree
        data_interval_for_plotting = np.arange(min(x_train) - 0.25, max(x_train) + 0.25, 0.01)
        y_fit = []
        for i in range(len(data_interval_for_plotting)):
            index = 1
            while index in self.is_terminal[False]:
                if data_interval_for_plotting[i] <= self.node_splits[index]:
                    index *= 2
                else:
                    index = (index * 2) + 1
            y_fit.append(self.terminal_node_means[index])
        draw_plot(data_interval_for_plotting, y_fit, "P = {}".format(p))

    def error_in_test(self):  # finds the rmse of test set
        y_head = np.arange(0, N)
        for i in y_head:
            index = 1
            while index in self.is_terminal[False]:
                if x_test[i] <= self.node_splits[index]:
                    index *= 2
                else:
                    index = (index * 2) + 1
            y_head[i] = self.terminal_node_means[index]
        return rmse(y_head, y_test, N)

    def pruning_tuning(self):  # finds the errors for different p parameters
        rmse_history = []
        for pruning_parameter in pruning_array:
            self.__init__()
            self.decision_tree_regression(pruning_parameter)
            rmse_history.append(self.error_in_test())
        plot.subplot()
        plot.plot(pruning_array, rmse_history, 'ro-', c='black')
        plot.xlabel('Pre-pruning size(P)')
        plot.ylabel('RMSE')
        plot.show()


if __name__ == "__main__":
    tree_reg = TreeRegression()
    tree_reg.decision_tree_regression()
    tree_reg.fit_line()
    print("RMSE is {} when P is {}".format(tree_reg.error_in_test(), p))
    tree_reg.pruning_tuning()
