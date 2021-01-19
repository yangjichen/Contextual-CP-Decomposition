'''
@File    : projected_CP.py

@Modify Time        @Author         @Version    @Desciption
------------        ------------    --------    -----------
2020-10-26 16:38    Jichen Yeung    1.0         None
'''
import tensorly as tl
import numpy as np
from tensorly.decomposition._cp import initialize_cp,error_calc
from tensorly.cp_tensor import unfolding_dot_khatri_rao,CPTensor



tl.set_backend('numpy')
tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)), dtype=tl.float32)

# 部分参数
l2_reg = 0
fixed_modes = []
n_iter_max = 1000
verbose = 1
modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]
rank=2
normalize_factors = True
linesearch = False
tol=1e-8
return_errors=False
mask = None
sparsity = None
cvg_criterion = 'abs_rec_error'




weights, factors = initialize_cp(tensor, rank=2, init='svd', svd='numpy_svd')

rec_errors = []
norm_tensor = tl.norm(tensor, 2)
Id = tl.eye(rank, **tl.context(tensor)) * l2_reg #暂时没看明白这一步是在做什么，默认l2_reg=0，会得到一个全0矩阵，看下面操作好像是正则项的意思

for iteration in range(n_iter_max):
    if verbose > 1:
        print("Starting iteration", iteration + 1)
    for mode in modes_list: #每个mode依次更新
        if verbose > 1:
            print("Mode", mode, "of", tl.ndim(tensor))

        pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor)) #全1的矩阵，ALS显式解的第三项
        for i, factor in enumerate(factors):
            if i != mode:
                pseudo_inverse = pseudo_inverse * tl.dot(tl.conj(tl.transpose(factor)), factor)
        pseudo_inverse += Id #看起来Id的作用是正则项

        if not iteration and weights is not None:
            # Take into account init weights
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode) #ALS显式解的前两项
        else:
            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

        factor = tl.transpose(tl.solve(tl.conj(tl.transpose(pseudo_inverse)), tl.transpose(mttkrp))) #这里之所以有这么多转置是因为solve(a,b)是解ax = b中的x
        #如果需要列单位化
        if normalize_factors:
            scales = tl.norm(factor, 2, axis=0)
            weights = tl.where(scales == 0, tl.ones(tl.shape(scales), **tl.context(factor)), scales)
            factor = factor / tl.reshape(weights, (1, -1)) #这里做归一化要小心，有numpy不熟悉的除法

        factors[mode] = factor

    # Will we be performing a line search iteration
    line_iter = False

    # Calculate the current unnormalized error if we need it
    if (tol or return_errors) and line_iter is False:
        unnorml_rec_error, tensor, norm_tensor = error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp) #The unnormalized reconstruction error
    else:
        if mask is not None:
            tensor = tensor * mask + tl.cp_to_tensor((weights, factors), mask=1 - mask)

    rec_error = unnorml_rec_error / norm_tensor #这个其实就是relative error，分母是原始tensor的norm
    rec_errors.append(rec_error)

    #收敛定义为：前后两轮的relative error的差值极小时说明收敛，abs(rec_error_decrease) < tol
    if tol:
        if iteration >= 1:
            rec_error_decrease = rec_errors[-2] - rec_errors[-1]

            if verbose:
                print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration,rec_error,rec_error_decrease,unnorml_rec_error))
            if cvg_criterion == 'abs_rec_error':
                stop_flag = abs(rec_error_decrease) < tol
            elif cvg_criterion == 'rec_error':
                stop_flag = rec_error_decrease < tol
            else:
                raise TypeError("Unknown convergence criterion")

            if stop_flag:
                if verbose:
                    print("PARAFAC converged after {} iterations".format(iteration))
                break
        else:
            if verbose:
                print('reconstruction error={}'.format(rec_errors[-1]))
cp_tensor = CPTensor((weights, factors))
# if return_errors:
#     return cp_tensor, rec_errors
# else:
#     return cp_tensor

# weights2, factors2 = tl.decomposition.parafac(tensor, rank=rank,normalize_factors = True)