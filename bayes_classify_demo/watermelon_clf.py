# coding=utf-8
import csv
import numpy as np
from math import sqrt

attr_num = [3, 3, 3, 3, 3, 2]   # 计算概率时，增加分母值的大小


def loadCsv(filename):
    '''
    读文件
    :param filename: 要读取的文件
    :return: 读成numpy数组文件
    '''
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    print("dataset：",dataset)

    for i in range(1, len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    result = np.array(dataset[1:])
    return result[:, 1:]


def pre_problity(datasets):
    '''
    求先验概率
    :param datasets: 加载数据
    :return: 概率
    '''
    # 求先验概率
    pos_prob = 1.0 * (np.sum(datasets[:, -1] == 1.0) + 1) / (np.shape(datasets)[0] + 2)
    neg_prob = 1.0 * (np.sum(datasets[:, -1] == 0.0) + 1) / (np.shape(datasets)[0] + 2)
    return [pos_prob, neg_prob]


def cond_attr_problity(datasets, testdata):
    '''
    分别计算与测试数据的相同值的情况下分别是好瓜与坏瓜的概率
    :param datasets: 训练数据集
    :param testdata: 测试数据集
    :return: 概率数组，第一列表示好瓜情况下相应是的概率，第二列表示坏瓜情况下相应的概率
    '''
    cond_result = np.zeros([np.shape(datasets)[1] - 1, 2])   # 用于存相应的概率
    pos_data = datasets[datasets[:, -1] == 1.0, :]   # Label为1的所有数值  好瓜
    neg_data = datasets[datasets[:, -1] == 0.0, :]   # Label为0的所有数值   坏瓜

    for i in range(len(attr_num)):
        # 离散条件概率求解
        print("len(attr_num)",len(attr_num))
        # 是好瓜的情况下某特征的概率
        cond_result[i, 0] = 1.0 * (np.sum(pos_data[:, i] == testdata[0, i]) + 1) / (np.sum(datasets[:, -1] == 1.0) + attr_num[i])
        # 不是好瓜的情况下某特征的概率
        cond_result[i, 1] = 1.0 * (np.sum(neg_data[:, i] == testdata[0, i]) + 1) / (np.sum(datasets[:, -1] == 0.0) + attr_num[i])

    for j in range(6, 8):
        # 连续条件概率求解
        pos_mean = np.mean(datasets[(datasets[:, -1] == 1.0), j])   # 求好瓜均值
        pos_std = np.std(datasets[(datasets[:, -1] == 1.0), j])   # 求好瓜标准差
        neg_mean = np.mean(datasets[(datasets[:, -1] == 0.0), j])  # 求坏瓜均值
        neg_std = np.std(datasets[(datasets[:, -1] == 0.0), j])  # 求坏瓜标准差
        # 求概率
        cond_result[j, 0] = 1.0 / (sqrt(2 * np.pi) * pos_std) * np.exp(-1 * (testdata[0, j] - pos_mean) ** 2 / (2 * pos_std ** 2))
        cond_result[j, 1] = 1.0 / (sqrt(2 * np.pi) * neg_std) * np.exp(-1 * (testdata[0, j] - neg_mean) ** 2 / (2 * neg_std ** 2))
    return cond_result


def classify_data(cond_result, pre_result):
    '''

    :param cond_result: 求得的概率矩阵
    :param pre_result: 先验概率矩阵
    :return: 判断结果
    '''
    # 先验概率，对比值
    pos_result = pre_result[0]
    neg_result = pre_result[1]

    for i in range(np.shape(cond_result)[0]):
        pos_result *= cond_result[i, 0]    # 好瓜概率
        neg_result *= cond_result[i, 1]    # 坏瓜概率
    if pos_result > neg_result:
        print('好瓜')
        print(pos_result)
    else:
        print('坏瓜')
        print(neg_result)


def main():
    filename = 'watermelon3_0_En.csv'   # 训练数据
    dataset = loadCsv(filename)
    testname = 'test.csv'    # 测试数据
    testdata = loadCsv(testname)
    pre_result = pre_problity(dataset)   # 求先验概率
    cond_result = cond_attr_problity(dataset, testdata)   # 求各个条件概率
    classify_data(cond_result, pre_result)   # 判断


main()