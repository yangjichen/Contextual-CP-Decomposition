'''
@File    : tool.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020-11-19 17:17    Jichen Yeung    1.0         None
'''
import numpy as np
import tensorly as tl
import os


def scaling_plus_minus(para, weights, factors):
    """
    :param weights: 已做归一化，但未调整正负的lambda
    :param factors: 已做归一化，但未调整正负的因子矩阵
    :return: 调整正负值的Scaling结果
    """
    for r in range(para['rank']):
        for j in range(len(factors)):
            if factors[j][0][r]<0:
                factors[j][:,r] = factors[j][:,r]*(-1)
                weights[r] = weights[r]*(-1)
    return (weights,factors)


# 调整矩阵列的顺序：计算降序排列后每列的位置
def col_permutation(weights, factors):
    """
    :param weights: 已调整正负，但未按照大小顺序做列调整的lambda
    :param factors: 已调整正负，但未按照大小顺序做列调整的因子矩阵
    :return: 按大小顺序做列调整的结果
    """
    order_idx = np.argsort(weights)[::-1]#降序排列后对应的索引
    for i in range(len(factors)):
        factors[i] = factors[i][:, order_idx]
    weights = weights[order_idx]
    return (weights,factors)

def SNR_sigma(Y, ratio):
    """
    :param Y: 原始信号张量Y
    :param ratio: 信噪比
    :return: 满足信噪比的噪声项参数 sigma
    """
    Y_norm_square = tl.norm(Y)**2
    Y_shape = np.prod(Y.shape)
    sigma = np.sqrt(Y_norm_square/(Y_shape*ratio))
    return sigma



def mkdir(path):
    """
    :param path: 创建文件夹
    :return: 创建文件夹
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder are created  ---")

    else:
        print("---  This folder already exists  ---")

