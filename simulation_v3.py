'''
@File    : simulation_v3.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020/11/30 下午4:48    Jichen Yeung    1.0       论文实验2：运行时间比较，加入可识别性条件，但Gamma为0
'''

import tensorly as tl
import numpy as np
import tool
import pandas as pd
from tensorly.decomposition import parafac
import time


# --------数据生成------------

def simulation1(para):
    X1, X2, X3 = np.random.uniform(-1, 1, size=(para['I1'], para['q'])), \
                 np.random.uniform(-1, 1, size=(para['I2'], para['q'])), \
                 np.random.uniform(-1, 1, size=(para['I3'], para['q']))

    B1, B2, B3 = np.random.randn(para['q'], para['rank']), \
                 np.random.randn(para['q'], para['rank']), \
                 np.random.randn(para['q'], para['rank'])
    # """
    # 下面这个方法噪声比例会更高
    # """
    # B1, B2, B3 = np.random.uniform(-1, 1, size=(para['q'], para['rank'])), \
    #              np.random.uniform(-1, 1, size=(para['q'], para['rank'])), \
    #              np.random.uniform(-1, 1, size=(para['q'], para['rank']))

    # 归一化后CP组合得到的Y与非归一化结果相同
    A, B, C = X1.dot(B1), X2.dot(B2), X3.dot(B3)
    lambda1, lambda2, lambda3 = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0), np.linalg.norm(C, axis=0)
    weights = lambda1 * lambda2 * lambda3
    factors = [A / lambda1, B / lambda2, C / lambda3]

    # Step4调整符号
    weights, factors = tool.scaling_plus_minus(para, weights, factors)
    # Step5整矩阵列的顺序：计算降序排列后每列的位置
    weights, factors = tool.col_permutation(weights, factors)

    # Step5整矩阵列的顺序：计算降序排列后每列的位置

    ori_Y = tl.cp_to_tensor((weights, factors))

    sigma = tool.SNR_sigma(ori_Y, para['SNR'])  #
    error = np.random.normal(0, sigma, (para['I1'], para['I2'], para['I3']))  # 生成特定信噪比error

    Y = ori_Y + error
    SNR1 = np.around(tl.norm(ori_Y) ** 2 / tl.norm(Y - ori_Y) ** 2, decimals=2)
    SNR2 = np.around(tl.norm(Y - ori_Y) ** 2 / tl.norm(ori_Y) ** 2, decimals=2)


    """
    算法实现
    """
    # 部分参数
    modes_list = [mode for mode in range(tl.ndim(Y))]
    corvariate = [X1, X2, X3]

    pro_start = time.time()
    # 初始化
    # 首先计算投影矩阵
    project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate]

    # -------option：需要做GX估计的时候再加入这个步骤---------
    # 根据投影矩阵和调整后的因子矩阵来得到符合可识别性条件的G(X)
    GX = [project[i].dot(factors[i]) for i in range(len(factors))]
    # ------------------option：end---------------------

    # 然后将Y张量投影
    project_Y = tl.tenalg.multi_mode_dot(Y, project)
    # Step1&2:计算G1G2G3，不需要归一化
    weights, Gs = parafac(project_Y, rank=para['rank'], normalize_factors=False, n_iter_max=1000)
    # Step3借助G1G2G3计算ABC
    hat_factor = []
    for mode in modes_list:
        tildeY = tl.tenalg.multi_mode_dot(Y, project, skip=mode)
        tempGs = Gs.copy()
        tempGs[mode] = np.eye(para['rank'])
        Q = tl.unfold(tl.cp_to_tensor((None, tempGs)), mode=mode)  # 推导中的Q矩阵
        factor = tl.unfold(tildeY, mode=mode).dot(Q.T).dot(np.linalg.inv(Q.dot(Q.T)))  # 推导中因子矩阵的计算方式
        hat_factor.append(factor)  # [A,B,C]的估计值
    # Step4-normalize
    hat_weight = np.ones(para['rank'])
    for mode in modes_list:
        tempweight = np.linalg.norm(hat_factor[mode], axis=0)
        hat_weight = hat_weight * tempweight
        hat_factor[mode] = hat_factor[mode] / tempweight
    # Step4'-调整正负
    hat_weight, hat_factor = tool.scaling_plus_minus(para, hat_weight, hat_factor)
    # Step5-调整列顺序
    hat_weight, hat_factor = tool.col_permutation(hat_weight, hat_factor)

    pro_end = time.time()
    # -----重构精度start---------
    # 利用新方法做出的估计

    hat_Y = tl.cp_to_tensor((hat_weight, hat_factor))
    new_RE = tl.norm(hat_Y - ori_Y) / tl.norm(ori_Y)

    # CP分解结果
    ori_start = time.time()
    cp_weights, cp_factors = parafac(Y, rank=para['rank'], normalize_factors=True, n_iter_max=1000)
    cp_weights, cp_factors = tool.scaling_plus_minus(para, cp_weights, cp_factors)
    cp_weights, cp_factors = tool.col_permutation(cp_weights, cp_factors)
    ori_end = time.time()

    newY = tl.cp_to_tensor((cp_weights, cp_factors))
    RE = tl.norm(newY - ori_Y) / tl.norm(ori_Y)

    # 保留4位小数
    new_RE = np.around(new_RE, decimals=5)
    RE = np.around(RE, decimals=5)

    # -----重构精度end---------

    # -----因子矩阵精度start---------
    # 利用新方法做出的估计
    pro_fac_RE = [np.around(tl.norm(factors[i] - hat_factor[i]) / tl.norm(factors[i]), decimals=5) for i in
                  range(len(factors))]
    ori_fac_RE = [np.around(tl.norm(factors[i] - cp_factors[i]) / tl.norm(factors[i]), decimals=5) for i in
                  range(len(factors))]

    # -----因子矩阵精度end---------

    # -----G(X)精度start---------
    """
    TODO：在做这个估计精度时，单独去写一个实验，因为本实验未加入Gamma，参考下如何加入Gamma
          这个实验好处在于不用去和CP比较，CP没有GX
          初期生成数据的时候，就用投影矩阵把最初的GX计算出来
          写在了simulation_v3中

    hatGX = [project[i].dot(cp_factors[i]) for i in range(len(factors))]
    GX_RE = [tl.norm(hatGX[i]-GX[i])/tl.norm(GX[i]) for i in range(len(factors))]
    print('GX_RE of projected CP = {}'.format(GX_RE))
    """
    # -----G(X)精度end---------
    return (pro_end-pro_start, ori_end-ori_start)


if __name__ == '__main__':
    tl.set_backend('numpy')
    para = {'I1': 20, 'I2': 20, 'I3': 20, 'rank': 3, 'q': 3, 'SNR': 100, 'dup': 100}

    for size in [20,50,100]:
        para['I1'], para['I2'], para['I3'] = size, size, size

        time1, time2 = 0,0
        for i in range(para['dup']):
            tmp = simulation1(para)
            time1 += tmp[0]
            time2 += tmp[1]
        print('The size of tensor is {}'.format(size))
        print('avg time of pro_CP is {}, avg time of ori_CP is {}'.format(time1/para['dup'], time2/para['dup']))
        print(' ')
    print('Done!')

