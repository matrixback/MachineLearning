# -*- coding: utf-8 -*-
import numpy as np

def load_set():
    data = []
    label = []
    fr = open('testSet.txt', 'r')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label.append(int(line_arr[2]))
    return data, label

def sigmoid(in_x):
    return 1.0/(1 + np.exp(-in_x))

def grad_ascent(data_list, class_label):
    """我们需要求的是损失函数的最大值？"""
    data_mat = np.mat(data_list)
    label_mat = np.mat(class_label).transpose()
    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 1000
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # data_mat*weights 矩阵乘积，为假设值 y(i)。
        # 将乘积再进行 sigmoid 转换，曲线拟合中，直接用乘积，而在分类中，必须转换。
        h = sigmoid(data_mat*weights)
        # 和书上的不一致，书上的本质也是梯度下降法，不知道人家为什么将 error = label_mat -h，然后起名叫梯度上升法
        error = (h - label_mat)   # 应该是求导的那一步的 diff
        # 和公式的写法正好相反，但是思路一样.
        weights -= alpha * data_mat.transpose() * error
    return weights

def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data, label = load_set()
    data_arr = np.array(data)
    n = np.shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label[i]) == 1:
            xcord1.append(data_arr[i, 1]); ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1]); ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascentd0(data, label):
    m, n = np.shape(data)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data[i] * weights))
        error = h - label[i]
        weights -= alpha * error * data[i]
    return weights


if __name__ == '__main__':
    data, label = load_set()
    weights = stoc_grad_ascentd0(np.array(data), label)
    print weights
    plot_best_fit(weights)


