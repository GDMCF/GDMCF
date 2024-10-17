import numpy as np
from fileinput import filename
import random
import torch as th
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def top_k_indices(tensor, k):
    """返回tensor中最大的k个值的索引"""
    _, indices = tensor.flatten().topk(k)
    return indices

def set_top_k_to_one(tensor, k=25000):
    """
    将tensor中值最大的k个位置设为1，其余设为0。
    参数:
    tensor: 输入的tensor，形状为[a,b]
    k: 需要设置为1的元素数量
    """
    # 获取整个tensor中最大的k个值的索引
    topk_indices = top_k_indices(tensor, k)
    
    # 初始化一个全零的tensor，形状与原tensor相同
    result_tensor = th.zeros_like(tensor)
    
    # 根据索引设置对应位置为1
    result_tensor.view(-1)[topk_indices] = 1
    
    return result_tensor

def topk_set(tensor, k=25000):
    print('')
    # 获取张量中每行（假设a为行数）的最大k个值及其索引
    values, indices = th.topk(tensor, k, dim=1)
    
    # 初始化一个与原tensor相同形状的零张量
    result_tensor = th.zeros_like(tensor)
    
    # 根据索引将结果张量中对应的位置设为1
    result_tensor.scatter_(1, indices, 1)
    
    return result_tensor


def adjacency_to_edge(X, index, a=5949):
    edge_list = []
    loca_list = th.nonzero(X)
    for k in range(loca_list.shape[0]):
        i = loca_list[k][0]
        j = loca_list[k][1]
        edge_list.append([index[i], a+j])
        # if i < a and j < a and i != j:  
        # Y[i, a+j] = 1  
        # Y[a+j, i] = 1  
    edge = th.tensor(edge_list)
    edge = edge.permute(1, 0)
    # print('edge:', edge.shape)    
    # import pdb
    # pdb.set_trace()
    return edge

def edge_to_adjacency(edge, index, a=5949, b=2810, bs=400):
    """
    将形状为[a+b, a+b]的one-hot编码矩阵Y转换回形状为[a, b]的邻接矩阵X。
    
    参数:
    - a: 节点数
    - Y: 形状为[a+b, a+b]的one-hot编码矩阵
    
    返回:
    - X: 形状为[a, b]的邻接矩阵
    """
    # 初始化邻接矩阵为0
    X = th.zeros((bs, b))
    rindex = th.zeros(a).long()
    for i in range(index.shape[0]):
        rindex[index[i]] = i
    
    # 根据one-hot编码逻辑恢复邻接矩阵
    # for i in range(a):  # 遍历所有可能的起始节点
    #     for j in range(a, a+b):  # 遍历可能的目标"边"索引对应的one-hot编码位置
    #         if Y[i, j] == 1:  # 如果在one-hot编码矩阵中找到对应边
    #             # 解码one-hot信息，还原邻接关系
    #             target_node_index = j - a  # 转换索引以匹配原始节点索引
    #             X[i, target_node_index] = 1  # 设置边的存在性
    for k in range(edge.shape[1]):
        i = edge[0][k]
        j = edge[1][k]
        target_node_index = j - a
        # print('i:{} j:{} tar:{}'.format(i, j, target_node_index))
        # import pdb
        # pdb.set_trace()
        #if j >= a:
        X[rindex[i], target_node_index] = 1
    # loca_list = th.nonzero(Y)
    # print('one_hot_to_adjacency loca_list:', loca_list.shape)
    # for k in range(loca_list.shape[0]):
    #     i = loca_list[k][0]
    #     j = loca_list[k][1]
    #     target_node_index = j - a
    #     # if i >= X.shape[0] or j >= X.shape[1] or i < 0 or target_node_index < 0:
    #     #     print('i:{} j:{} target_node_index:{}'.format(i, j, target_node_index))
    #     #     import pdb
    #     #     pdb.set_trace()
    #     if i < a and j >= a:
    #         X[i, target_node_index] = Y[i][j]

    return X

def pred_to_adjacency(edge, index, a=5949, b=2810, bs=400, pred=None):
    """
    将形状为[a+b, a+b]的one-hot编码矩阵Y转换回形状为[a, b]的邻接矩阵X。
    
    参数:
    - a: 节点数
    - Y: 形状为[a+b, a+b]的one-hot编码矩阵
    
    返回:
    - X: 形状为[a, b]的邻接矩阵
    """
    # 初始化邻接矩阵为0
    X = th.zeros((bs, b))
    rindex = th.zeros(a).long()
    for i in range(index.shape[0]):
        rindex[index[i]] = i
    
    # 根据one-hot编码逻辑恢复邻接矩阵
    # for i in range(a):  # 遍历所有可能的起始节点
    #     for j in range(a, a+b):  # 遍历可能的目标"边"索引对应的one-hot编码位置
    #         if Y[i, j] == 1:  # 如果在one-hot编码矩阵中找到对应边
    #             # 解码one-hot信息，还原邻接关系
    #             target_node_index = j - a  # 转换索引以匹配原始节点索引
    #             X[i, target_node_index] = 1  # 设置边的存在性
    for k in range(edge.shape[1]):
        if pred[k] != 1:
            continue
        i = edge[0][k]
        j = edge[1][k]
        target_node_index = j - a
        # print('i:{} j:{} tar:{}'.format(i, j, target_node_index))
        # import pdb
        # pdb.set_trace()
        #if j >= a:
        X[rindex[i], target_node_index] = 1
    # loca_list = th.nonzero(Y)
    # print('one_hot_to_adjacency loca_list:', loca_list.shape)
    # for k in range(loca_list.shape[0]):
    #     i = loca_list[k][0]
    #     j = loca_list[k][1]
    #     target_node_index = j - a
    #     # if i >= X.shape[0] or j >= X.shape[1] or i < 0 or target_node_index < 0:
    #     #     print('i:{} j:{} target_node_index:{}'.format(i, j, target_node_index))
    #     #     import pdb
    #     #     pdb.set_trace()
    #     if i < a and j >= a:
    #         X[i, target_node_index] = Y[i][j]

    return X


def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    # idm = torch.zeros(n_user, 1)
    # for i in range(idm.shape[0]):
    #     idm[i] = i
    # idm = idm.numpy()
    # print('idm:', idm.shape)
    # #train_data = np.array(train_data)
    # print('train_data:', train_data.shape)
    # #train_data = torch.from_numpy(train_data)
    # #train_data = torch.cat([train_data, idm], dim=-1)
    # train_data = np.concatenate([train_data, idm], axis=0)
    # # test_y_data = np.concatenate([test_y_data, idm], axis=-1)
    # # valid_y_data = np.concatenate([valid_y_data, idm], axis=-1)
    # print('train_data:{} test_y_data:{} valid_y_data:{}'.format(train_data.shape, test_y_data.shape, valid_y_data.shape))
    # import pdb
    # pdb.set_trace()
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
        # print('data:', data.shape)
        # import pdb
        # pdb.set_trace()
    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    def __len__(self):
        return len(self.data)
    

