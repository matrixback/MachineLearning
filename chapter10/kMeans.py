# -*- coding: utf-8 -*-
from numpy import *

def load_data_set(file_name):
    """将文件转变为列表，并转为 float 类型"""
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = map(float, cur_line)
        data_mat.append(flt_line)
    return data_mat

def dist_eclud(vec_A, vec_B):
    """计算欧式距离"""
    return sqrt(sum(power((vec_A - vec_B), 2)))

def rand_cent(data_set, k):
    """根据最大值，最小值，随机生成k个质心。"""
    n = shape(data_set)[1]
    centroids = mat(zeros((k, n)))
    # 对每一维进行计算
    for j in range(n):
        min_j = min(data_set[:, j])   # 切片运算，返回最小值
        range_j = float(max(data_set[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * random.rand(k, 1)
    return centroids

def k_means(data_set, k, dist_means=dist_eclud, create_cent=rand_cent):
    m = shape(data_set)[0]  # 数据点的个数
    cluster_assessment = mat(zeros((m, 2))) # 每个点的簇分配结果，第一列当前点所在的质心索引值，第二列记录与质心直接的距离
    centroids = create_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # 对于每个点，遍历质心，找到最近的质心
        for i in range(m):
            # 初始化最小距离和质心索引
            min_dist = inf
            min_index = -1
            for j in range(k):
                # centroids, data_set 都是二维矩阵，注意切片运算
                distJI = dist_means(centroids[j, :], data_set[i, :])
                if distJI < min_dist:
                    min_dist = distJI
                    min_index = j
            if cluster_assessment[i, 0] != min_index:
                cluster_changed = True
            cluster_assessment[i, :] = [min_index, min_dist ** 2]
            print centroids
            # 遍历质心，更新其值
            for cent in range(k):
                # 取得质心为当前质心的所有点，注意用 nonzero进行了过滤。
                # 过滤时用了 nonzero 函数，其返回值不为0，False等的序列。
                pts_in_clust = data_set[nonzero(cluster_assessment[:, 0].A == cent)[0]]
                centroids[cent, :] = mean(pts_in_clust, axis=0)
            return centroids, cluster_assessment

if __name__ == '__main__':
    data_mat = mat(load_data_set('testSet.txt'))
    cent, clust_accing = k_means(data_mat, 4)
    print cent
    print clust_accing