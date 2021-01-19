'''
@File    : create_data.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020-10-28 10:23    Jichen Yeung    1.0         预备实验：重构精度simulation，暂未考虑可识别性，未在论文出现
'''
import tensorly as tl
import numpy as np

from tensorly.decomposition import parafac

"""
数据生成
"""

tl.set_backend('numpy')

kkkRE1 = []
kkkRE2 = []
for kkkkk in range(10):
    para = {'I1':100, 'I2':100, 'I3':100, 'rank':5, 'q':10}

    X1,X2,X3 = np.random.uniform(0,1,size=(para['I1'],para['q'])), \
               np.random.uniform(0,1,size=(para['I2'],para['q'])), \
               np.random.uniform(0,1,size=(para['I3'],para['q']))

    B1, B2, B3 = np.random.randn(para['q'], para['rank']),\
                 np.random.randn(para['q'], para['rank']),\
                 np.random.randn(para['q'], para['rank'])
    # #非归一化结果
    # A, B, C = X1.dot(B1), X2.dot(B2), X3.dot(B3)
    # factors = [A, B, C]
    # Y1 = tl.cp_to_tensor((None,factors))

    #归一化后看是否相同
    A, B, C = X1.dot(B1), X2.dot(B2), X3.dot(B3)
    lambda1, lambda2, lambda3 = np.linalg.norm(A,axis=0),np.linalg.norm(B,axis=0),np.linalg.norm(C,axis=0)
    weights = lambda1*lambda2*lambda3
    factors = [A/lambda1, B/lambda2, C/lambda3]

    ori_Y = tl.cp_to_tensor((weights, factors))
    error = np.random.randn(para['I1'], para['I2'], para['I3']) #生成error
    Y = ori_Y+error


    """
    算法实现
    """
    # 部分参数
    modes_list = [mode for mode in range(tl.ndim(Y))]
    corvariate = [X1,X2,X3]


    # 初始化
    # 首先计算投影矩阵
    project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate]
    # 然后将Y张量投影
    project_Y = tl.tenalg.multi_mode_dot(Y, project)
    #Step1:计算G1G2G3，不需要归一化
    weights, Gs = parafac(project_Y, rank=para['rank'],normalize_factors = False,n_iter_max=100)
    #Step2借助G1G2G3计算ABC
    hat_factor = []
    for mode in modes_list:
        tildeY = tl.tenalg.multi_mode_dot(Y, project, skip=mode)
        tempGs = Gs.copy()
        tempGs[mode] = np.eye(para['rank'])
        Q = tl.unfold(tl.cp_to_tensor((None, tempGs)), mode = mode) #推导中的Q矩阵
        factor = tl.unfold(tildeY, mode=mode).dot(Q.T).dot( np.linalg.inv(Q.dot(Q.T)) ) #推导中因子矩阵的计算方式
        hat_factor.append(factor) #[A,B,C]的估计值
    # Step3-normalize
    hat_weight = np.ones(para['rank'])
    for mode in modes_list:
        tempweight = np.linalg.norm(hat_factor[mode], axis=0)
        hat_weight =hat_weight*tempweight
        hat_factor[mode] = hat_factor[mode]/tempweight

    #利用新方法做出的估计
    hat_Y = tl.cp_to_tensor((hat_weight, hat_factor))
    new_RE = tl.norm(hat_Y-ori_Y)/tl.norm(ori_Y)

    #CP分解结果
    cp_weights, cp_factors = parafac(Y, rank=para['rank'], normalize_factors = True, n_iter_max=100)
    newY = tl.cp_to_tensor((cp_weights, cp_factors))
    RE = tl.norm(newY-ori_Y)/tl.norm(ori_Y)

    kkkRE1.append(RE)
    kkkRE2.append(new_RE)
    print(kkkkk)
print('CPALS error = {}, Project_CPALS error = {}, upgrade = {}'.format(np.mean(kkkRE1), np.mean(kkkRE2), np.mean(kkkRE1)-np.mean(kkkRE2)))


import pandas as pd
import matplotlib.pyplot as plt
data = [kkkRE1[i]-kkkRE2[i] for i in range(len(kkkRE1))]
df = pd.DataFrame(data)
df.plot.box(title="Improvement")
plt.grid(linestyle="--", alpha=0.3)
plt.show()





