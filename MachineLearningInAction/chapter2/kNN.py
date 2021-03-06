# -*- coding: utf-8 -*-

import operator
from numpy import *

def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(in_x, data_set, labels, k):
    """
    :param in_x: 输入向量，需要用户输入【x, y]。
    :param data_set:
    :param labels:
    :param k:
    :return:
    """
    data_set_size = data_set.shape[0]    # 多少维数据
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set   # 得到的是每个向量与输入向量的差
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)   # 返回向量的和
    distances = sq_distances ** 0.5

    sorted_dist_indicies = distances.argsort()   # 索引排序
    class_count = {}
    # 统计前 K 个标签值
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 将字典排序时，根据那个域的值。operator.itemgetter 的参数可以为数字和名称。
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape=(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


if __name__ == '__main__':
    x = array([0.2, 0.2])
    data_set, labels = create_data_set()
    print classify(x, data_set, labels, 2)
