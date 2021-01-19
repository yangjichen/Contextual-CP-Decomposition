'''
@File    : simulation2.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020-11-20 11:10    Jichen Yeung    1.0         论文实验1：重构精度simulation，加入可识别性条件，但Gamma为0
'''
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
    weights, factors = tool.col_permutation(weights,factors)

    # Step5整矩阵列的顺序：计算降序排列后每列的位置

    ori_Y = tl.cp_to_tensor((weights, factors))

    sigma = tool.SNR_sigma(ori_Y, para['SNR']) #
    error = np.random.normal(0, sigma, (para['I1'], para['I2'], para['I3'])) # 生成特定信噪比error

    Y = ori_Y +error
    print('Y均值为{}'.format(np.around(Y.mean(), decimals = 2)))
    SNR1 = np.around(tl.norm(ori_Y)**2/tl.norm(Y - ori_Y)**2, decimals=2)
    SNR2 = np.around(tl.norm(Y - ori_Y)**2 / tl.norm(ori_Y)**2, decimals=2)
    print('信号噪音比为:{}'.format(SNR1)) #查看信噪比
    print('噪音信号比为:{}'.format(SNR2) ) # 查看信噪比

    """
    算法实现
    """
    # 部分参数
    modes_list = [mode for mode in range(tl.ndim(Y))]
    corvariate = [X1, X2, X3]

    # 初始化
    # 首先计算投影矩阵
    project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate]

    # -------option：需要做GX估计的时候再加入这个步骤---------
    # 根据投影矩阵和调整后的因子矩阵来得到符合可识别性条件的G(X)
    GX = [project[i].dot(factors[i]) for i in range(len(factors)) ]
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


    # -----重构精度start---------
    # 利用新方法做出的估计
    hat_Y = tl.cp_to_tensor((hat_weight, hat_factor))
    new_RE = tl.norm(hat_Y - ori_Y) / tl.norm(ori_Y)

    # CP分解结果
    cp_weights, cp_factors = parafac(Y, rank=para['rank'], normalize_factors=True, n_iter_max=1000)

    cp_weights, cp_factors = tool.scaling_plus_minus(para, cp_weights, cp_factors)
    cp_weights, cp_factors = tool.col_permutation(cp_weights, cp_factors)
    newY = tl.cp_to_tensor((cp_weights, cp_factors))
    RE = tl.norm(newY - ori_Y) / tl.norm(ori_Y)

    #保留4位小数
    new_RE = np.around(new_RE, decimals=5)
    RE = np.around(RE, decimals=5)
    print('RE of projected CP = {}, RE of original CP = {},'.format(new_RE, RE))
    # -----重构精度end---------


    # -----因子矩阵精度start---------
    # 利用新方法做出的估计
    pro_fac_RE = [np.around(tl.norm(factors[i]-hat_factor[i])/tl.norm(factors[i]),decimals=5)  for i in range(len(factors))]
    ori_fac_RE = [np.around(tl.norm(factors[i] - cp_factors[i]) / tl.norm(factors[i]),decimals=5) for i in range(len(factors))]
    print('factor_RE of projected CP = {}'.format(pro_fac_RE))
    print('factor_RE of original CP = {}'.format(ori_fac_RE))
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
    return (new_RE, RE, pro_fac_RE, ori_fac_RE)


if __name__=='__main__':
    tl.set_backend('numpy')
    para = {'I1': 100, 'I2': 100, 'I3': 100, 'rank': 3, 'q': 3, 'SNR': '待输入', 'dup':100}

    for ratio in [100,10,1,0.1]:
        para['SNR'] = ratio
        res1, res2 = [],[] #res1 projected_CP重构RE；res2 ori_CP重构RE
        res3, res4 = [],[] #res1 projected_CP因子RE；res2 ori_CP因子RE

        for i in range(para['dup']):
            print('-------start {}-------'.format(i))
            tmp = simulation1(para)

            res1.append(tmp[0])
            res2.append(tmp[1])
            res3.append(tmp[2])
            res4.append(tmp[3])

            print('-------end {}-------'.format(i))
            print(' ')

        """
        统计数据所用，保存成csv
        """
        res1 = np.array(res1).reshape(-1, 1)
        res2 = np.array(res2).reshape(-1, 1)
        res3 = np.array(res3)
        res4 = np.array(res4)

        res = np.hstack((res1, res2,res3,res4))
        res = pd.DataFrame(res,columns=['pro_RE', 'ori_RE', 'pro_A','pro_B','pro_C','ori_A','ori_B','ori_C' ])
        print(res.mean())
        #
        # save_path = 'result/size{}/'.format(para['I1'])
        # tool.mkdir(save_path)
        #
        # res.to_csv(save_path+'SNR{}_res.csv'.format(para['SNR']), sep = ',', index=False)
        # print('Done!')

