'''
@File    : GDELT3.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020/12/3 下午9:08    Jichen Yeung    1.0        对之前整理好的张量进行ProCP的分析
'''

import tensorly as tl
import numpy as np
import tool
import pandas as pd
from tensorly.decomposition import parafac
import matplotlib.pyplot as plt



def BIC(Y,Yhat,rank):
    M = np.prod(Y.shape)
    pe = rank*(np.sum(Y.shape)-3+1)
    res = 2*M*np.log(tl.norm(Y - Yhat))+pe*np.log(M)
    return res


# ------读入数据并作预处理--------
path = 'DATA/step2/'
Y = np.load(path+'tensor.npy')
Y = Y/1000 #将Y的数值单位缩放为 千次
geolist = np.load(path+'geolist.npy')
timelist = np.load(path+'timelist.npy')
data = pd.read_csv('Geopolitical power.csv')




# ------如果顺序对的上，则确定协变量的数值并归一化-----
if list(data.iloc[0].index) == list(geolist):
    X = data.iloc[0].values.astype(float)
    X = X/1000 #X的单位是mM，代表原有粒度乘了1000，我们这里回归到原始单位mir
    X = X.reshape((Y.shape[0],1))
else:
    print('Order of geolist is wrong' )


# ------模型参数设定，其中para['rank']结合CPALS和BIC准则来确定-----
para = {'I1': Y.shape[0], 'I2': Y.shape[1], 'I3': Y.shape[2]}
# bic = {}
# for rank in range(1,50):
#     para['rank'] = rank
#     cp_weights, cp_factors = parafac(Y, rank=para['rank'], normalize_factors=True, n_iter_max=1000)
#     Yhat = tl.cp_to_tensor((cp_weights, cp_factors))
#     bic[rank] = BIC(Y, Yhat, para['rank'])
#     print('---rank = {}, BIC = {}'.format(rank, bic[rank]))
# print('The best rank is {}'.format(min(bic,key=bic.get)))
# para['rank'] = min(bic,key=bic.get)
para['rank'] = 3




# -------正式算法--------
# 部分参数
modes_list = [mode for mode in range(tl.ndim(Y))]
corvariate = [X, X, np.identity(Y.shape[2])]

# 初始化
# 首先计算投影矩阵
project = [i.dot(np.linalg.inv(i.T.dot(i))).dot(i.T) for i in corvariate]
# 然后将Y张量投影
project_Y = tl.tenalg.multi_mode_dot(Y, project)

# Step1&2:计算G1G2G3，不需要归一化, 这一步初始化使用SVD的话会warning
weights, Gs = parafac(project_Y, rank=para['rank'], normalize_factors=False, n_iter_max=1000,verbose=1,init='random')
hat_proY = tl.cp_to_tensor((weights, Gs))
print('Error of reconstruction project_Y is: {}'.format(tl.norm(hat_proY - project_Y) / tl.norm(project_Y)))

# Step3借助G1G2G3计算ABC
hat_factor = []
for mode in modes_list:
    tildeY = tl.tenalg.multi_mode_dot(Y, project, skip=mode)
    tempGs = Gs.copy()
    tempGs[mode] = np.eye(para['rank'])
    Q = tl.unfold(tl.cp_to_tensor((None, tempGs)), mode=mode)  # 推导中的Q矩阵
    factor = tl.unfold(tildeY, mode=mode).dot(Q.T).dot(np.linalg.inv(Q.dot(Q.T)+1e-10))  # 推导中因子矩阵的计算方式
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
"""
TODO:目前来看，投影方法遇到了一点bug，导致分解很差
问题所在：tensorly源码的锅，不能使用svd做初始化，且有一定概率在原始CP分解中遇到求逆出现奇异阵的情况，
解决方案：不建议修改源码，只需要做好自己算法的分析。采用random初始化

问题2：Error of reconstruction project_Y一直很低，Error of reconstruction Y有时会很高，
      所以在选定rank后，多跑几组，人为选取reconstruction Y结果好的呈现
"""
# 利用新方法做出的估计
hat_Y = tl.cp_to_tensor((hat_weight, hat_factor))
hat_Y_RE = tl.norm(hat_Y - Y) / tl.norm(Y)
print('Error of reconstruction Y is: {}'.format(hat_Y_RE))
# -----重构精度end---------

# ------如果重构精度满足我们的需要，则进行下一步，系数B2的估计--------
# ------且和(Chen, 2020)不同的点在于，我认为Gamma2同样也需要估计，就像线性回归，不能扔掉截距项-------
G2 = project[1].dot(hat_factor[1])
B2 = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(G2)
Gamma2 = hat_factor[1]-G2




"""
如下完成可视化
"""
# ------下面来将X从0到200等值取，看不同国家均值的情况--------
data = []
for i in range(20,101,10):
    newX = np.ones(para['I1']).reshape((para['I1'],1))*i
    # ------按照第二个维度求均值压缩，等价于去看特定几个国家对于geopolitical power为[0,20,40,...200]这些国家的外交------
    newfactor = [hat_factor[0], newX.dot(B2)+Gamma2, hat_factor[2]]
    newY = tl.cp_to_tensor((hat_weight, newfactor))
    newY = newY.mean(axis = 1).mean(axis = 1)
    # ------选择了 USA,India,China,Russia,Brazil 5个国家来研究-------
    data.append(newY[[0,1,3,8,9]])
data = np.array(data).T

# 恢复原有量级，并取log
x_axis = list(range(20,101,10))
plt.plot(x_axis, np.log(data[0]*1000), color='green', label='USA')
plt.plot(x_axis, np.log(data[1]*1000), color='red', label='India')
plt.plot(x_axis, np.log(data[2]*1000),  color='skyblue', label='UK')
plt.plot(x_axis, np.log(data[3]*1000), color='blue', label='China')
plt.plot(x_axis, np.log(data[4]*1000), color='black', label='Russia')
plt.legend() # 显示图例

plt.xlabel('Geopolitical power')
plt.ylabel('Total counts (log)')
plt.savefig('scatter.png',dpi=600,format='png')
plt.show()
