'''
@File    : simulation_v3.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020/11/25 上午9:47    Jichen Yeung    1.0       重构精度simulation，加入可识别性条件，Gamma不为0
                                                生成Gamma的方式：随机数生成，向X空间投影，只保留垂直的信息
'''

import tensorly as tl
import numpy as np
import tool
import pandas as pd
from tensorly.decomposition import parafac

"""
TODO：为什么加上gamma后重构精度降低了，因子矩阵ABC估计也不够准，projection到底是在提升什么
"""

import tensorly as tl
import numpy as np
import tool
import pandas as pd
from tensorly.decomposition import parafac


# --------数据生成------------

def simulation1(para):
    X1, X2, X3 = np.random.uniform(-1, 1, size=(para['I1'], para['q'])), \
                 np.random.uniform(-1, 1, size=(para['I2'], para['q'])), \
                 np.random.uniform(-1, 1, size=(para['I3'], para['q']))

    B1, B2, B3 = np.random.randn(para['q'], para['rank']), \
                 np.random.randn(para['q'], para['rank']), \
                 np.random.randn(para['q'], para['rank'])
    Gamma1, Gamma2, Gamma3 = np.random.randn(para['I1'], para['rank']), \
                             np.random.randn(para['I2'], para['rank']), \
                             np.random.randn(para['I3'], para['rank'])
    corvariate = [X1, X2, X3]
    project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate] # 需要对随机数Gamma做投影，所以得先计算投影矩阵

    #------通过投影，消除Gamma中与X无关的项------
    Gamma1 = Gamma1 - project[0].dot(Gamma1)
    Gamma2 = Gamma2 - project[0].dot(Gamma2)
    Gamma3 = Gamma3 - project[0].dot(Gamma3)

    # 归一化后CP组合得到的Y与非归一化结果相同
    A, B, C = X1.dot(B1) + Gamma1, X2.dot(B2) + Gamma2, X3.dot(B3) + Gamma3

    lambda1, lambda2, lambda3 = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0), np.linalg.norm(C, axis=0)

    weights = lambda1 * lambda2 * lambda3
    factors = [A / lambda1, B / lambda2, C / lambda3]

    # Step4调整符号
    weights, factors = tool.scaling_plus_minus(para, weights, factors)
    # Step5整矩阵列的顺序：计算降序排列后每列的位置
    weights, factors = tool.col_permutation(weights, factors)
    # -------option：需要做GX估计的时候再加入这个步骤---------
    # 根据投影矩阵和调整后的因子矩阵来得到符合可识别性条件的G(X)
    GX = [project[i].dot(factors[i]) for i in range(len(factors))]
    # ------------------option：end---------------------
    # Step5整矩阵列的顺序：计算降序排列后每列的位置

    ori_Y = tl.cp_to_tensor((weights, factors))

    sigma = tool.SNR_sigma(ori_Y, para['SNR'])  #
    error = np.random.normal(0, sigma, (para['I1'], para['I2'], para['I3']))  # 生成特定信噪比error

    Y = ori_Y + error
    print('Y均值为{}'.format(np.around(Y.mean(), decimals=2)))
    SNR1 = np.around(tl.norm(ori_Y) ** 2 / tl.norm(Y - ori_Y) ** 2, decimals=2)
    SNR2 = np.around(tl.norm(Y - ori_Y) ** 2 / tl.norm(ori_Y) ** 2, decimals=2)
    print('信号噪音比为:{}'.format(SNR1))  # 查看信噪比
    print('噪音信号比为:{}'.format(SNR2))  # 查看信噪比

    """
    算法实现
    """
    # 部分参数
    modes_list = [mode for mode in range(tl.ndim(Y))]

    # 初始化
    # 首先计算投影矩阵
    project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate]

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
    hat_GX = [project[i].dot(hat_factor[i]) for i in range(len(factors))]

    # -------G(X)精度start---------
    GX_RE = [np.around(tl.norm(hat_GX[i] - GX[i]) / tl.norm(GX[i]),decimals=5) for i in range(len(factors))]
    print('RE of GX = {}'.format(GX_RE))
    # -----G(X)精度end---------

    return (GX_RE)


if __name__ == '__main__':
    tl.set_backend('numpy')
    para = {'I1': 100, 'I2': 100, 'I3': 100, 'rank': 3, 'q': 3, 'SNR': 100, 'dup': 100}

    for size in [20,50,100]:
        para['I1'],para['I2'],para['I3'] = size, size, size
        res = []  # GX估计值的RE

        for i in range(para['dup']):
            print('-------start {}-------'.format(i))
            res.append(simulation1(para))
            print('-------end {}-------'.format(i))
            print(' ')
        """
        统计数据所用，保存成csv
        """
        res = pd.DataFrame(res, columns=['GX1', 'GX2', 'GX3'])
        print(res.mean())

        # ------创建文件夹并保存------
        save_path = 'result/GX_RE/'
        tool.mkdir(save_path)
        res.to_csv(save_path + 'size{}.csv'.format(para['I1']), sep=',', index=False)
        # ------创建文件夹并保存end------
        print('Done!')

