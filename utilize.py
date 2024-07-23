import scipy.sparse as sp
import torch
import dgl
import numpy as np
import random
from math import *

def get_position(matrix):
    association = sp.coo_matrix(matrix)
    id_1 = torch.from_numpy(association.row)
    id_2 = torch.from_numpy(association.col)
    return id_1,id_2

def get_m_hop1_rw(g,m_number):
    # Perform a random walk on a heterogeneous graph
    mmrw1 = dgl.sampling.random_walk(
        g, [i for i in range(m_number)], metapath=['md_association',
                                                   'dm_association',
                                                   ], restart_prob=0.5
    )
    sample, ntype = mmrw1
    # print(ntype.shape[0])
    for n in range(ntype.shape[0]):
        if ntype[n] == 1:
            type = n
    sample = torch.index_select(sample, 1, torch.tensor([0, type]))
    list = []
    for i in range(m_number-1):
        if sample[i][1] != -1:
            list = list + [i]

    sample_mm1 = torch.index_select(sample, 0, torch.tensor(list))
    return sample_mm1

def get_m_hop2_rw(g,m_number):
    # Perform a random walk on a heterogeneous graph
    mmrw2 = dgl.sampling.random_walk(
        g, [i for i in range(m_number)], metapath=['md_association',
                                                   'dm_association',
                                                   'md_association',
                                                   'dm_association',
                                                   ], restart_prob=0.5
    )
    sample, ntype = mmrw2
    for n in range(ntype.shape[0]):
        if ntype[n] == 1:
            type = n
    sample = torch.index_select(sample, 1, torch.tensor([0, type]))
    list = []
    for i in range(m_number-1):
        if sample[i][1] != -1:
            list = list + [i]
    sample_mm2 = torch.index_select(sample, 0, torch.tensor(list))
    return sample_mm2

def get_d_hop1_rw(g,d_number):
    # Perform a random walk on a heterogeneous graph
    ddrw1 = dgl.sampling.random_walk(
        g, [i for i in range(d_number)], metapath=['dm_association',
                                                   'md_association',
                                                   ], restart_prob=0.5
    )
    sample, ntype = ddrw1
    for n in range(ntype.shape[0]):
        if ntype[n] == 0:
            type = n
    sample = torch.index_select(sample, 1, torch.tensor([0, type]))
    list = []
    for i in range(d_number-1):
        if sample[i][1] != -1:
            list = list + [i]
    sample_dd1 = torch.index_select(sample, 0, torch.tensor(list))
    return sample_dd1

def get_d_hop2_rw(g,d_number):
    # Perform a random walk on a heterogeneous graph
    ddrw1 = dgl.sampling.random_walk(
        g, [i for i in range(d_number)], metapath=['dm_association',
                                                   'md_association',
                                                   'dm_association',
                                                   'md_association',
                                                   ], restart_prob=0.5
    )
    sample, ntype = ddrw1
    for n in range(ntype.shape[0]):
        if ntype[n] == 0:
            type = n
    sample = torch.index_select(sample, 1, torch.tensor([0, type]))
    list = []
    for i in range(d_number-1):
        if sample[i][1] != -1:
            list = list + [i]
    sample_dd2 = torch.index_select(sample, 0, torch.tensor(list))
    return sample_dd2

def get_isom_adj(n_number,sample):
    n_adj = np.zeros((n_number, n_number))
    start_n = torch.index_select(sample, 1, torch.tensor([0]))
    start_n = start_n.view([1, -1])
    start_n = start_n.squeeze()
    start_n_list = start_n.numpy().tolist()
    end_n = torch.index_select(sample, 1, torch.tensor([1]))
    end_n = end_n.view([1, -1])
    end_n = torch.where(end_n < 0, 0, end_n)
    end_n = end_n.squeeze()
    end_n_list = end_n.numpy().tolist()
    # Construct an adjacency matrix for a homogeneous graph
    for i in range(len(start_n_list)):
        n_adj[start_n_list[i]][end_n_list[i]] = 1
    n_adj = n_adj + np.eye(n_number)
    return n_adj

def get_all_the_samples(A):
    m,n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i,j] ==1:
                pos.append([i,j,1])
            else:
                neg.append([i,j,0])
    n = len(pos)
    neg_new = random.sample(neg, n) # Negative samples equal to the number of positive samples are randomly selected
    tep_samples = pos + neg_new
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    return samples

def get_all_samples(A):
    m, n = A.shape
    pos = []
    neg = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                pos.append([i, j, 1])
            else:
                neg.append([i, j, 0])
    n = len(pos)
    neg_new = random.sample(neg, n)  # Negative samples equal to the number of positive samples are randomly selected
    # One-tenth of the positive sample is randomly selected as the positive sample of the independent validation set
    pos_ind, pos_train = data_split(list(pos), 0.1, True)
    # pos_ind = np.array(pos_ind)
    neg_ind, neg_train = data_split(list(neg_new), 0.1, True)
    # neg_ind = np.array(neg_ind)
    ind_tep_sample = pos_ind + neg_ind
    ind_sample = random.sample(ind_tep_sample, len(ind_tep_sample))
    ind_sample = random.sample(ind_sample, len(ind_sample))
    # The remaining positive and negative samples are composed of the training samples
    train_tep_sample = pos_train + neg_train
    train_sample = random.sample(train_tep_sample, len(train_tep_sample))
    train_sample = random.sample(train_sample, len(train_sample))
    tep_samples = pos + neg_new  # The total number of samples, where the number of positive and negative samples is the same
    samples = random.sample(tep_samples, len(tep_samples))
    samples = random.sample(samples, len(samples))
    samples = np.array(samples)
    ind_sample = np.array(ind_sample)
    train_sample = np.array(train_sample)
    return samples,ind_sample,train_sample

import random
def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def set_digo_zero(sim, z):
    sim_new = sim.copy()
    n = sim.shape[0]
    for i in range(n):
        sim_new[i][i] = z
    return sim_new

def preprocess_feature(adj,number):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    if not sp.isspmatrix_coo(adj_normalized):
        adj_normalized = adj_normalized.tocoo()
    # coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = adj_normalized.data
    similarity = values.reshape(-1, number)
    return similarity

def get_adj(row_number,col_number,sample):
    n_adj = np.zeros((row_number, col_number))
    sample = torch.tensor(sample)
    m_node = torch.index_select(sample, 1, torch.tensor([0]))
    m_node = m_node.view([1, -1])
    m_node = m_node.squeeze()
    m_node_list = m_node.numpy().tolist()
    d_node = torch.index_select(sample, 1, torch.tensor([1]))
    d_node = d_node.view([1, -1])
    d_node = d_node.squeeze()
    d_node_list = d_node.numpy().tolist()
    value = torch.index_select(sample, 1, torch.tensor([2]))
    value = value.view([1, -1])

    value = value.squeeze()
    value = value.numpy().tolist()
    for i in range(len(sample)):
        n_adj[m_node_list[i]][d_node_list[i]] = value[i]
    return n_adj,value

def choose_adj_value(adj,sample):
    sample = torch.tensor(sample)
    start_n = torch.index_select(sample, 1, torch.tensor([0]))
    start_n = start_n.view([1, -1])
    start_n = start_n.squeeze()
    start_n_list = start_n.numpy().tolist()
    end_n = torch.index_select(sample, 1, torch.tensor([1]))
    end_n = end_n.view([1, -1])
    end_n = torch.where(end_n < 0, 0, end_n)
    end_n = end_n.squeeze()
    end_n_list = end_n.numpy().tolist()
    value = []
    for i in range(len(start_n_list)):
        value.append(adj[start_n_list[i], end_n_list[i]])
    return value

def matrix_factorization(Y, rank, aplha, beta, steps):
    '''
    :param Y: label matrix m*n
    :param U: Linear features of miRNAs m*k
    :param V: Linear features of diseases m*k
    :param K: The dimension of the linear feature
    :param aplha: Learning rate
    :param beta: Regularization parameters
    :param steps:
    :return:
    '''
    print('Begin to decompose the original matrix: \n')

    Y = np.array(Y)

    # Number of rows of label matrix Y
    rows_Y = len(Y)

    # Number of columns of label matrix Y
    columns_Y = len(Y[0])  # The number of columns of the original matrix R

    # Random initialization matrix. [0 1]
    U = np.random.rand(rows_Y, rank)
    print(U)
    print(U.shape)
    V = np.random.rand(columns_Y, rank)
    # Transpose
    V = V.T

    result = []

    # update parameters using gradient descent method
    print('Start training: \n')
    for step in range(steps):
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                eij = Y[i][j] - np.dot(U[i, :], V[:, j])
                for k in range(rank):
                    if Y[i][j] > 0:
                        # update parameters
                        U[i][k] = U[i][k] + aplha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] = V[k][j] + aplha * (2 * eij * U[i][k] - beta * V[k][j])

        # loss
        e = 0
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] > 0:
                    e = e + pow(Y[i][j] - np.dot(U[i, :], V[:, j]), 2)  # loss
                    for k in range(rank):
                        e = e + (beta / 2) * (pow(U[i][k], 2) + pow(V[k][j], 2))  # loss with regularization
        result.append(e)
        if e < 0.001:
            break
    print('training Finshed...')
    return U, V.T, result
