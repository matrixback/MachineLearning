# -*- coding: utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def file2matrix(filename):
    "将文本文件写入矩阵"
    with open(filename) as f:
        lines = f.readlines()
        number_lines = len(lines)
        return_mat = zeros((number_lines, 3))  # 直接创建一个 0 矩阵，以后不再需要将文本从串显示转换。
        class_label_vector = []
        index = 0
        for line in lines:
            line.strip()
            list_from_line = line.split('\t')
            return_mat[index, :] = list_from_line[0:3]   # 矩阵会自动将字符串转为 int 类型。
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector

if __name__ == "__main__":
    f = "/Users/matrix/MachineLearning/datas/Ch02/datingTestSet2.txt"
    dating_data_mat, dating_labels = file2matrix(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0*array(dating_labels), 15.0*array(dating_labels))   # 后两个参数是比例和颜色，比例默认值为20，即点的大小？现在是（15-45之间），颜色也是？

    plt.savefig('dating3.png')