# -*- coding: utf-8 -*-
"""
Truncate a matrix with mpo in a truncate number.
Date: 2020.11.16
@author: zfgao
"""
import numpy as np
import random
import torch.nn as nn
import torch
import pandas as pd
seed = 1234
random.seed(seed)
np.random.seed(seed)


class MPO:
    def __init__(self, mpo_input_shape, mpo_output_shape, truncate_num, fix_rank=None):
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.truncate_num = truncate_num
        self.num_dim = len(mpo_input_shape)
        self.mpo_ranks = self.compute_rank(truncate_num=None)
        if fix_rank:
            self.mpo_truncate_ranks = fix_rank
        else:
            self.mpo_truncate_ranks = self.compute_rank(truncate_num=self.truncate_num)

    def compute_rank_position(self, s, truncate_num=None):
        #s表示位置，返回对应位置的键维数
        """
        Calculate the rank position in MPO bond dimension
        :param s: target bond ,type = int, range in [1:len(mpo_input_shape-1)], r_0 = r_n = 1.
        :return:  target bond 's' real bond dimension.
        """
        rank_left = 1  # ranks_left: all the shape multiply in left of 's'.
        rank_right = 1  # ranks_right: all the shape multiply in right of 's'.
        for i in range(0, s):
            rank_left = rank_left * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        for i in range(s, self.num_dim):
            rank_right = rank_right * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        if truncate_num == None:
            min_rank = min(rank_left, rank_right)
        else:
            min_rank = min(int(self.truncate_num), rank_left, rank_right)
        return min_rank

    def compute_rank(self, truncate_num):
        #返回列表，每个地方的键维数
        """
        :param mpo_input_shape: the input mpo shape, type = list. [i0,i1,i2,...,i_(n-1)]
        :param truncate_num: the truncate number of mpo, type = int.
        :return:max bond dimension in every bond position, type = list, [r0,r1,r2,...,r_n],r0=r_n=1
        """
        bond_dims = [1 for i in range(self.num_dim + 1)]
        for i in range(1, self.num_dim):
            bond_dims[i] = self.compute_rank_position(i, truncate_num)
        return bond_dims

    def get_tensor_set(self, inp_matrix):
        """
        Calculate the left canonical of input matrix with a given mpo_input_shape
        :param inp_matrix: the input matrix
        :param mpo_input_shape:
        :return: a tensor with left canonical in input matrix
        """
        tensor_set = []
        res = inp_matrix
        #################################################################################
        
        res = res.reshape(tuple(self.mpo_input_shape[:]) + tuple(self.mpo_output_shape[:]))
        self.index_permute = np.transpose(
            np.array(range(len(self.mpo_input_shape) + len(self.mpo_output_shape))).reshape((2, -1))).flatten()
        res = np.transpose(res, self.index_permute)
        #################################################################################
        for i in range(self.num_dim - 1):
            # Do the SVD operator
            res = res.reshape([self.mpo_ranks[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i], -1])
            u, lamda, v = np.linalg.svd(res, full_matrices=False)
            # The first tensor should be T1(r_i+1, m_i, n_i, r_i)
            u = u.reshape([self.mpo_ranks[i], self.mpo_input_shape[i], self.mpo_output_shape[i], self.mpo_ranks[i+1]])
            tensor_set.append(u)
            res = np.dot(np.diag(lamda), v)
        #最后一个A
        res = res.reshape([self.mpo_ranks[self.num_dim-1], self.mpo_input_shape[self.num_dim-1],
                           self.mpo_output_shape[self.num_dim-1], self.mpo_ranks[self.num_dim]])
        tensor_set.append(res)
        return tensor_set
        #W=A1*A2*...*An
    def left_canonical(self,tensor_set):
        left_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        mat = tensor_set[0]
        mat = mat.reshape(-1, mat.shape[3])
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        left_canonical_tensor[1] = np.dot(np.diag(lamda), v)
        for i in range(1,self.num_dim-1):
            mat = np.tensordot(left_canonical_tensor[i], tensor_set[i],[1,0])
            mat = mat.reshape(-1, mat.shape[-1])
            u,lamda,v = np.linalg.svd(mat, full_matrices=False)#截断
            left_canonical_tensor[i+1] = np.dot(np.diag(lamda), v)
        return left_canonical_tensor

    def right_canonical(self, tensor_set):
        """
        Calculate the right tensor canonical for MPO format required
        :param left_tensor: the tensor_set output from function: left_canonical
        :return: the right_tensor_canonical format for calculate the mpo decomposition
        """
        right_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        # print(tensor_set.shape)
        mat = tensor_set[self.num_dim - 1]
        mat = mat.reshape(mat.shape[0], -1)
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        right_canonical_tensor[self.num_dim - 1] = np.dot(u, np.diag(lamda))

        for i in range(self.num_dim - 2, 0, -1):
            mat = np.tensordot(tensor_set[i], right_canonical_tensor[i + 1], [3, 0])
            mat = mat.reshape(mat.shape[0], -1)
            u, lamda, v = np.linalg.svd(mat, full_matrices=False)
            right_canonical_tensor[i] = np.dot(u, np.diag(lamda))
        return right_canonical_tensor

    def expectrum_normalization(self, lamda):
        """
        Do the lamda normalization for calculate the needed rank for MPO structure
        :param lamda: lamda parameter from left canonical
        :return:
        """
        norm_para = np.sum(lamda ** 2) ** (0.5)
        lamda_n = lamda / norm_para
        lamda_12 = lamda ** (-0.5)
        return lamda_n, np.diag(lamda_12)

    def gauge_aux_p_q(self, left_canonical_tensor, right_canonical_tensor):
        p = [0 for i in range(self.num_dim + 1)]
        q = [0 for i in range(self.num_dim + 1)]
        lamda_set = [0 for i in range(self.num_dim + 1)]
        lamda_set_value = [0 for i in range(self.num_dim + 1)]
        lamda_set[0] = np.ones([1,1])
        lamda_set[-1] = np.ones([1,1])
        for i in range(1, self.num_dim):
            mat = np.dot(left_canonical_tensor[i],right_canonical_tensor[i])
            # mat = right_canonical_tensor[i]
            u, lamda, v = np.linalg.svd(mat)
            lamda_n, lamda_l2 = self.expectrum_normalization(lamda)
            lamda_set[i] = lamda_n
            lamda_set_value[i] = lamda
            p[i] = np.dot(right_canonical_tensor[i], v.T)
            p[i] = np.dot(p[i],lamda_l2)
            q[i] = np.dot(lamda_l2,u.T)
            q[i] = np.dot(q[i], left_canonical_tensor[i])
        return p, q, lamda_set, lamda_set_value

    def mpo_canonical(self, tensor_set, p, q):
        tensor_set[0] = np.tensordot(tensor_set[0], p[1], [3,0])
        tensor_set[-1] = np.tensordot(q[self.num_dim-1], tensor_set[-1], [1,0])
        for i in range(1, self.num_dim-1):
            tensor_set[i] = np.tensordot(q[i],tensor_set[i],[1,0])
            tensor_set[i] = np.tensordot(tensor_set[i],p[i+1], [3,0])
        return tensor_set


    def truncated_tensor(self, tensor_set, step_train=False):
        """
        Get a untruncated tensor by mpo
        :param tensor_set: the input weight
        :return: a untruncated tensor_set by mpo
        """    
        if step_train:
            tensor_set_tmp = [i.detach().cpu().numpy() for i in tensor_set]
            cano_tensor_set = self.bi_canonical(tensor_set_tmp)
            tensor_set = torch.nn.ParameterList(
            [nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in cano_tensor_set])
            tensor_set[2].requires_grad = False

        mpo_trunc = self.mpo_truncate_ranks[:]
        for i in range(self.num_dim):
            if step_train:
                mask_noise = torch.ones_like(tensor_set[i])
            t = tensor_set[i]
            r_l = mpo_trunc[i]
            r_r = mpo_trunc[i + 1]
            if isinstance(tensor_set[i], nn.parameter.Parameter):
                if step_train:
                    
                    mask_noise[r_l:, :, :, :] = 0.0
                    mask_noise[:r_l, :, :, r_r:] = 0.0
                    tensor_set[i].data = tensor_set[i].data * mask_noise
                else:
                    tensor_set[i].data = t[:r_l, :, :, :r_r]
            else:
                tensor_set[i] = t[:r_l, :, :, :r_r]
                assert "Check! tensor_set is not nn.parameter.Parameter"
        return tensor_set

    def matrix2mpo(self, inp_matrix, cutoff=True):
        """
        Utilize the matrix to mpo format with or without cutoff
        :param inp_matrix: the input matrix, type=list
        :param cutoff: weather cut of not, type = bool
        :return: the truncated of not mps format of input matrix
        """
        tensor_set = self.get_tensor_set(inp_matrix)
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,lamda_set, lamda_set_value = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)
        if cutoff != False:
            tensor_set = self.truncated_tensor(tensor_set)
        return tensor_set,lamda_set, lamda_set_value
    def bi_canonical(self, tensor_set):
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,_, _ = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)

        return tensor_set
    def mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
            #t的最后一个维度和tensor_set[i]的第一个维度是一样的，通过求和来完成
        # Squeeze the first and the last 1 dimension
        t = t.squeeze(0)
        t = t.squeeze(-1)
        #t.shape (m1,n1,m2,n2,...,mk,nk)
        # Caculate the new index for mpo
        tmp1 = torch.tensor(range(len(self.mpo_output_shape))) * 2
        tmp2 = tmp1 + 1
        new_index = torch.cat((tmp1, tmp2), 0)#先输出维度，后输入维度
        # Transpose and reshape to output
        t = t.permute(tuple(new_index))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t

    def calculate_total_mpo_param(self, cutoff=True):
        # print("use cutoff: ", cutoff)
        total_size = 0
        if cutoff:
            rank = self.mpo_truncate_ranks
        else:
            rank = self.mpo_ranks
        for i in range(len(self.mpo_input_shape)):
            total_size += rank[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i] * rank[i + 1]

        return total_size
    def new_mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t
    @staticmethod
    def test_difference(matrix1, matrix2):
        """
        we input an matrix , return the difference between those two matrix
        :param matrix:
        :return:
        """
        v = matrix1 - matrix2
        error = np.linalg.norm(v)
        return error
    
    def extract_features(self, spatial_field):
        """
        从空间场中提取MPO特征
        spatial_field: [101, 101, 2] 单个时间步的流场
        返回: 压缩后的特征向量
        """
        # 将空间场reshape为适合MPO的形式
        if len(spatial_field.shape) == 3:
            # 如果是[101, 101, 2]，可以视为矩阵进行分解
            matrix_form = spatial_field.reshape(-1, spatial_field.shape[-1])
        else:
            matrix_form = spatial_field
            
        # 进行MPO分解
        tensor_set, lamda_set, _ = self.matrix2mpo(matrix_form, cutoff=True)
        
        # 使用MPO张量或奇异值作为特征
        features = []
        for i, tensor in enumerate(tensor_set):
            # 提取每个张量的特征（展平或取统计量）
            features.append(tensor.flatten())
        
        # 或者使用奇异值作为特征
        features.append(lamda_set[2].flatten())  # 中心位置的奇异值
        
        return torch.cat(features)
        
def FixAuxilaryTensorCalculateCentralTensor(tensor_set,New_matrix,New_central_in,New_central_out):
    
    #只更新中心向量
    """
    In put tensor set product by matrix2MPO, and New_matrix.
    return the central tensor when auxiliary tensor was fixed.
    We assumes n = 5
    """
    numpy_type = type(np.random.rand(2,2))
    if type(New_matrix) == numpy_type:
        New_matrix = torch.from_numpy(New_matrix).cuda()
    else:
        New_matrix = New_matrix.cuda()
    if type(tensor_set[0]) == numpy_type:
        a = torch.from_numpy(tensor_set[0])
        b = torch.from_numpy(tensor_set[1])
        Ori_CentralTensor = torch.from_numpy(tensor_set[2])
        d = torch.from_numpy(tensor_set[3])
        e = torch.from_numpy(tensor_set[4])
    else:
        a = tensor_set[0]
        b = tensor_set[1]
        Ori_CentralTensor = tensor_set[2]
        d = tensor_set[3]
        e = tensor_set[4]
    left_basis = torch.tensordot(a,b, ([3],[0])).reshape(-1,Ori_CentralTensor.shape[0])
    right_basis = torch.tensordot(d,e,([3],[0])).reshape(Ori_CentralTensor.shape[-1],-1)
    left_basis_inv = torch.inverse(left_basis)
    right_basis_inv = torch.inverse(right_basis)
    CentralTensor = torch.reshape(New_matrix, [Ori_CentralTensor.shape[0],New_central_in,New_central_out,Ori_CentralTensor.shape[3]])
    M_C = torch.tensordot(left_basis_inv,CentralTensor,([1],[0]))
    M_C = torch.tensordot(M_C,right_basis_inv,([3],[0]))
    return M_C


if __name__ == "__main__":
    # 修改MPO形状设置
    mpo_input_shape = [1, 101, 101, 2]    # 输入：单个时间步的流场
    mpo_output_shape = [1, 101, 101, 2]   # 输出：单个时间步的流场
    
    # 读取数据
    df = pd.read_csv("Burgers_data1.csv", comment='%', header=None)

    # 数据预处理
    columns_per_timestep = 5
    num_timesteps = (df.shape[1] - 2) // columns_per_timestep

    X = df.iloc[:, 0].values
    Y = df.iloc[:, 1].values

    x_unique = np.sort(np.unique(X))
    y_unique = np.sort(np.unique(Y))
    nx = len(x_unique)
    ny = len(y_unique)
    
    result = np.zeros((num_timesteps, nx, ny, 2))

    x_to_idx = {x_val: idx for idx, x_val in enumerate(x_unique)}
    y_to_idx = {y_val: idx for idx, y_val in enumerate(y_unique)}
    
    for t_idx in range(num_timesteps):
        start_col = 2 + t_idx * columns_per_timestep
        t_data = df.iloc[:, start_col:start_col+5].values
        for row in t_data:
            x_val, y_val, u, v = row[1], row[2], row[3], row[4]
            i = x_to_idx[x_val]
            j = y_to_idx[y_val]
            result[t_idx, i, j, 0] = u
            result[t_idx, i, j, 1] = v
    
    result_tensor = torch.from_numpy(result).float()
    print(f"数据张量形状: {result_tensor.shape}")
    
    # 创建用于特征提取的权重矩阵
    input_dim = np.prod(mpo_input_shape)  # 1*101*101*2 = 20402
    output_dim = np.prod(mpo_output_shape) # 1*101*101*2 = 20402
    
    # 创建一个虚拟的权重矩阵进行MPO分解测试
    # 在实际应用中，这个权重矩阵应该是可训练的参数
    weight_matrix = torch.randn(input_dim, output_dim)
    print(f"权重矩阵形状: {weight_matrix.shape}")
    
    mpo = MPO(mpo_input_shape=mpo_input_shape, mpo_output_shape=mpo_output_shape, truncate_num=100)
    print('input_modes is: ', mpo.mpo_input_shape)
    print('output_modes is: ', mpo.mpo_output_shape)
    print('max_bond_dims is: ', mpo.mpo_ranks)
    print('truncate_bond_dims is:', mpo.mpo_truncate_ranks)

    # 对权重矩阵进行MPO分解，而不是直接对数据分解
    mpo_set, lamda_set, lamda_set_value = mpo.matrix2mpo(weight_matrix.numpy(), cutoff=True)
    
    # 重建权重矩阵并计算误差
    reconstructed_weight = mpo.mpo2matrix(mpo_set)
    diff = mpo.test_difference(weight_matrix.numpy(), reconstructed_weight.numpy())
    
    print(f"MPO重建误差: {diff}")
    print(f"MPO参数数量: {mpo.calculate_total_mpo_param(cutoff=True)}")
    print(f"原始参数数量: {input_dim * output_dim}")
    print(f"压缩比: {(input_dim * output_dim) / mpo.calculate_total_mpo_param(cutoff=True):.2f}x")