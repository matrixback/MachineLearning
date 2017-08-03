# -*- coding: utf-8 -*-
import numpy as np

def grad(x, y):
    """x 为二维矩阵，一列表示一个记录，注意转换。theta, y 都为行向量。"""
    m, n = x.shape                   # 样本个数
    epsilon = 0.000000000001
    alpha = 0.01
    theta = np.zeros(m)
    diff = np.zeros(m)
    max_itor = 10000
    cnt = 0
    err = 0
    last_err = 0

    while cnt < max_itor:
        cnt += 1
        diff = np.dot(theta, x) - y
        for i  in range(len(theta)):
            theta[i] -= alpha * sum(diff * x[i])

        new_diff = np.dot(theta, x) - y
        err = sum(new_diff * new_diff) / 2
        if abs(err - last_err) < epsilon:
            break
        else:
            last_err = err

        print ' theta: {} error1: {}'.format(theta, err)

    print 'Done: theta: {}'.format(theta)
    print '迭代次数: %d' % cnt

def grad_2(x, y):
    """不按公式的写法，按 x 一行表示一个记录的写法编程。对应，theta, y 都为列向量。"""
    m, n = x.shape                   # 样本个数
    epsilon = 0.000000000001
    alpha = 0.01
    theta = np.zeros((n, 1))
    print theta
    diff = np.zeros((n, 1))
    max_itor = 10000
    cnt = 0
    err = 0
    last_err = 0
    while cnt < max_itor:
        cnt += 1
        diff = np.dot(x, theta) - y
        theta -= alpha * np.dot(x.transpose(), diff)
        new_diff = np.dot(x, theta) - y
        err = sum(new_diff * new_diff) / 2
        if abs(err - last_err) < epsilon:
            break
        else:
            last_err = err

        print ' theta: {} error1: {}'.format(theta, err)

    print 'Done: theta: {}'.format(theta)
    print '迭代次数: %d' % cnt

def test_1():
    x = np.array([(1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 1, 7)])
    y = np.array([4, 6, 8, 10, 12, 14])
    # 将 x 进行转置，每一列表示一个记录，方便与数学公式对应。
    x = x.transpose()
    grad(x, y)

def test_2():
    x = np.array([(1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 1, 7)])
    # 注意，一维的array 转置还是一维，必须二维的才能转置。所以最好用 mat，不用 array.
    y = np.array([[4, 6, 8, 10, 12, 14]]).transpose()
    print y
    grad_2(x, y)


if __name__ == '__main__':
    test_1()

