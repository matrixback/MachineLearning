# -*- coding: utf-8 -*-

from math import log
import operator

def calc_shannon_ent(data_set):
    """
    这个函数很简单，就是求出概率，然后求出整个的香农商
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    返回某一维等于某个值的向量组，每个向量不包含此维的值。
    """
    ret_data_set = []    # 原生数据需要调用多次，为了不影响数据，新创建一个列表对象
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        feat_list = [example[i] for example in data_set]   # 对于每个向量取某维的数据，就是得到某列的值。这里没有用切片，而是用了列表表达式，感觉不是很清楚。
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy    # 熵为负数，所以要用减号
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majorty_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 1
        class_count[vote] += 1
    sorted_class_count = sorted(class_count, key=operator.getitem(1), reverse=True)
    return sorted_class_count[0][0]
    





