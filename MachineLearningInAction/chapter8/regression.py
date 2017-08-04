# -*- coding: utf-8 -*-
from numpy import *

def load_data_set(filename):
    """文件有三列，
    第一列固定为1.0，是偏移量；
    第二列为横坐标值；
    第三列为纵坐标值。

    为什么在数据文件中多处一列为固定值 1.0 ？
    因为按照上述理论计算的 y 中，没有固定常数。我们希望有个常数。即 y = w0 + w1x1 + w2x2 + ...，其等价于 y = w0*1.0 + w1x1 + ....
    人为给 x 增加一列即可。
    """
    num_feat = len(open(filename).readline().split('\t')) - 1
    data_mat = []
    label_mat = []

    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regress(x_arr, y_arr):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T   # 一般的矩阵是列矩阵
    xTx = x_mat.T * x_mat
    # 计算行列式是否为0。linalg 线性代数库。
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
    ws = xTx.I * (x_mat.T * y_mat)
    return ws

def main():
    x_arr, y_arr = load_data_set('ex0.txt')
    ws = stand_regress(x_arr, y_arr)
    print(ws)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flattern 将 矩阵展开。mat.A 返回 ndarray 对象。
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat)
    # plt.show()

    # 计算相关性
def cal_corrcoef():
    x_arr, y_arr = load_data_set('ex0.txt')
    ws = stand_regress(x_arr, y_arr)
    print(ws)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    y_hat = x_mat * ws
    print corrcoef(y_hat.T, y_mat)

# main()
cal_corrcoef()






