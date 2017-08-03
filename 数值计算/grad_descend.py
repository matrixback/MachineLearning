# -*- coding: utf-8 -*-
import numpy as np

def grad(x, y):
    print x
    print y
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

if __name__ == '__main__':
    x = np.array([(1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 1, 7)])
    y = np.array([4, 6, 8, 10, 12, 14])
    # 将 x 进行转置，每一列表示一个记录，方便与数学公式对应。
    x = x.transpose()
    grad(x, y)

