import random

import dgl
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
from utilize import *
from sklearn.model_selection import KFold
from torch import nn
import torch.nn.functional as F
from model import GAT,GAT_MF
from MF import matrix_factorization
from sklearn.metrics import roc_curve,auc



#先导入邻阶矩阵
import torch

A = np.load('D:\\研究生文件\\论文\\IGLMDA-demo\\data\\HMDD v3.2\\miRNA-disease association.npy')
m_number,d_number = A.shape
print('miRNA和disease的数量',A.shape)
print('miRNA和disease的关系数',sum(sum(A)))

#再导入miRNA和disease的相似度矩阵(下面的两个相似度都是从图自动编码器那里获得，
# 这里为了减少运行的时间成本，直接使用处理过后的相似度)
sim_m = np.loadtxt('D:\\研究生文件\\论文\\IGLMDA-demo\\data\\HMDD v3.2\\sim_m')
print(sim_m.shape)

sim_d = np.loadtxt('D:\\研究生文件\\论文\\IGLMDA-demo\\data\\HMDD v3.2\\sim_d')
print(sim_d.shape)

#对特征矩阵进行处理
sim_m_0 = set_digo_zero(sim_m,0)
sim_d_0 = set_digo_zero(sim_d,0)
print(sim_m_0)

m_similarity = preprocess_feature(sim_m_0,m_number)
d_similarity = preprocess_feature(sim_d_0,d_number)

# print(m_similarity)
m_similarity = set_digo_zero(m_similarity,0)
d_similarity = set_digo_zero(d_similarity,0)
m_similarity = m_similarity + np.eye(788)
d_similarity = d_similarity + np.eye(374)

#获得miRNA-disease异构矩阵中有关联的位置
miRNA_id,disease_id = get_position(A)
print(miRNA_id.size())
print(miRNA_id.type())


#构建出异构图
g = dgl.heterograph({
    ('miRNA','md_association','disease'):(miRNA_id,disease_id),
    ('disease','dm_association','miRNA'):(disease_id,miRNA_id)
})
print(g)
print(g.ntypes)

#开始构建同构图(目的是生成不同的步长的同构图)

#设置同构图采样的步长
p = 1
#对异构图进行随机游走
sample_mm_hop1 = get_m_hop1_rw(g,m_number)
print(sample_mm_hop1)
#至此对于miRNA的一阶元路径邻居的位置信息已经获得
#构建miRNA同构图的邻接矩阵
m_adj = get_isom_adj(m_number,sample_mm_hop1)
print(m_adj)
#再获得miRNA和miRNA关系的位置
m_start, m_end = get_position(m_adj)
# 使用dgl去构建miRNA-miRNA同构图
g_m = dgl.DGLGraph()
g_m.add_nodes(m_number)
m_start = m_start.type(torch.int64)
m_end = m_end.type(torch.int64)
g_m.add_edges(m_end,m_start)
print(g_m)
# nx.draw(g_m.to_networkx(),with_labels=True)
# plt.show()
#添加顶点属性
print(sim_m_0.dtype)
m_similarity = torch.from_numpy(m_similarity)
g_m.ndata['feature'] = m_similarity
print(g_m)
#对1跳miRNA同构图进行消息传递
import dgl.function as fn
g_m.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
m_similarity1 = g_m.ndata['feature']
new_feature_m = m_similarity1.numpy()
print(g_m.ndata['feature'])




#获得1_hop的disease同构图特征
sample_dd_hop1 = get_d_hop1_rw(g,d_number)
d_adj = get_isom_adj(d_number,sample_dd_hop1)
d_start, d_end = get_position(d_adj)
g_d = dgl.DGLGraph()
g_d.add_nodes(d_number)
d_start = d_start.type(torch.int64)
d_end = d_end.type(torch.int64)
g_d.add_edges(d_end,d_start)
print(g_d)
# nx.draw(g_d.to_networkx(),with_labels=True)
# plt.show()
#添加顶点属性
d_similarity = torch.from_numpy(d_similarity)
print(d_similarity)
g_d.ndata['feature'] = d_similarity
print(g_d)
g_d.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
d_similarity1 = g_d.ndata['feature']
new_feature_d = d_similarity1.numpy()
print(g_d.ndata['feature'])


#开始构建同构图(目的是生成不同的步长的同构图)

#设置同构图采样的步长
p = 2
sample_mm_hop2 = get_m_hop2_rw(g,m_number)
print(sample_mm_hop2)
#至此对于miRNA的一阶元路径邻居的位置信息已经获得
#构建miRNA同构图的邻接矩阵
m_adj_2 = get_isom_adj(m_number,sample_mm_hop2)
print(m_adj)
#再获得miRNA和miRNA关系的位置
m_start_2, m_end_2 = get_position(m_adj_2)
# 使用dgl去构建miRNA-miRNA同构图
g_m_2 = dgl.DGLGraph()
g_m_2.add_nodes(m_number)
m_start_2 = m_start_2.type(torch.int64)
m_end_2 = m_end_2.type(torch.int64)
g_m_2.add_edges(m_end_2,m_start_2)
print(g_m_2)
# nx.draw(g_m.to_networkx(),with_labels=True)
# plt.show()
#添加顶点属性
print(sim_m_0.dtype)
g_m_2.ndata['feature'] = m_similarity
print(g_m_2)
#对1跳miRNA同构图进行消息传递
g_m_2.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
m_similarity2 = g_m_2.ndata['feature']
new_feature_m_2 = m_similarity2.numpy()
print(g_m_2.ndata['feature'])



sample_dd_hop2 = get_d_hop2_rw(g,d_number)
print(sample_dd_hop2)
#至此对于miRNA的一阶元路径邻居的位置信息已经获得
#构建miRNA同构图的邻接矩阵
d_adj_2 = get_isom_adj(d_number,sample_dd_hop2)
print(d_adj_2)
#再获得miRNA和miRNA关系的位置
d_start_2, d_end_2 = get_position(d_adj_2)
# 使用dgl去构建miRNA-miRNA同构图
g_d_2 = dgl.DGLGraph()
g_d_2.add_nodes(d_number)
d_start_2 = d_start_2.type(torch.int64)
d_end_2 = d_end_2.type(torch.int64)
g_d_2.add_edges(d_end_2,d_start_2)
print(g_d_2)
# nx.draw(g_m.to_networkx(),with_labels=True)
# plt.show()
#添加顶点属性
print(sim_d_0.dtype)
g_d_2.ndata['feature'] = d_similarity
print(g_d_2)
#对1跳miRNA同构图进行消息传递
g_d_2.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
d_similarity2 = g_d_2.ndata['feature']
new_feature_d_2 = d_similarity2.numpy()
print(g_d_2.ndata['feature'])

#目前的
#开始设置训练集和测试集
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


samples = get_all_the_samples(A)#选择和正样本数量一致的负样本作为总的样本
print(samples)
print(len(samples))


import pandas as pd

data1 = pd.DataFrame(samples)
# data1.to_csv('./N_mgatmf10/samples150.csv')
sample_label = choose_adj_value(A,samples)
# print(sample_label)

c, label = get_adj(A.shape[0],A.shape[1],samples)
print(c.shape)
print(label)
print(len(label))
# np.savetxt("./N_mgatmf10/label150.txt", np.array(label))

kf = KFold(n_splits=10,shuffle=True)
#对邻接矩阵进行归一化

model1 = GAT(in_features_m=torch.tensor(788),
            in_features_d=torch.tensor(374),
            out_features=torch.tensor(100),
            n_channels=torch.tensor(3),
            dropout=0.3,
            nheads =3)
model1 = model1.cuda()
123
#利用MF获得隐向量
A = np.array(A)
#U, V, result = matrix_factorization(A, 50, aplha=0.006, beta=0.00002, steps=100)
U = np.load('D:\\研究生文件\\论文\\IGLMDA-demo\\data\\U.npy')
print(U.shape)
V = np.load('D:\\研究生文件\\论文\\IGLMDA-demo\\data\\V.npy')
print(V)
print(V.shape)

model2 = GAT_MF(in_features_m=torch.tensor(788),
            in_features_d=torch.tensor(374),
            out_features=torch.tensor(50),
            h_features= torch.tensor(50),
            n_channels=torch.tensor(3),
            dropout=0.1,
            nheads=3)

model2 = model2.cuda()

#选择模型
model = model1
loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0015)
train_loss = []
test_loss = []
pre = []


for train_index, test_index in kf.split(samples):
    train_set = samples[train_index,:]

    print(train_set)
    print(len(train_set))
    test_set = samples[test_index,:]
    print(test_set)
    print(len(test_set))
    #print(train_samples.shape[0])  16143/16142
    train_set_adj,train_set_label = get_adj(A.shape[0],A.shape[1],train_set)
    print('miRNA和disease的关系数', sum(sum(train_set_adj)))
    print(device)
    train_set_adj = torch.tensor(train_set_adj)
    train_set_adj = train_set_adj.to(device)
    A = torch.tensor(A)
    m_similarity = m_similarity.to(device)
    d_similarity = d_similarity.to(device)
    m_similarity1 = m_similarity1.to(device)
    d_similarity1 = d_similarity1.to(device)
    m_similarity2 = m_similarity2.to(device)
    d_similarity2 = d_similarity2.to(device)
    #U = U.to(device)
    #V = V.to(device)
    m_similarity = torch.tensor(m_similarity).float()
    d_similarity = torch.tensor(d_similarity).float()
    m_similarity1 = torch.tensor(m_similarity1).float()
    d_similarity1 = torch.tensor(d_similarity1).float()
    m_similarity2 = torch.tensor(m_similarity2).float()
    d_similarity2 = torch.tensor(d_similarity2).float()
    U = torch.tensor(U).float()
    U = U.to(device)
    V = torch.tensor(V).float()
    V = V.to(device)

    #到此可以进行训练了
    epoch = 150
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(m_similarity,
                       m_similarity1,
                       m_similarity2,
                       # U,
                       d_similarity,
                       d_similarity1,
                       d_similarity2,
                       # V,
                       train_set_adj)
        print(output)

        print(output.shape)

        pred_value = choose_adj_value(output,train_set)
        pred_value = torch.stack(pred_value)
        pred_value = pred_value.to(torch.float32)
        pred_value = pred_value.to(device)
        train_set_label = torch.tensor(train_set_label)
        train_set_label = train_set_label.to(torch.float32)
        train_set_label = train_set_label.to(device)
        loss = loss_fn(pred_value,train_set_label)
        print(loss)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print(train_loss)
#训练完成，在验证集上验证
    print(train_loss)
    output = output.cpu()
    test_set_adj,test_set_label = get_adj(A.shape[0],A.shape[1],test_set)
    print(test_set_label)
    print(len(test_set_label))
    test_sample_score = choose_adj_value(output.detach().numpy(), test_set)
    # print(test_sample_score)
    # print(len(test_sample_score))

    fp_test, tp_test, threshold_test = roc_curve(test_set_label, test_sample_score)
    print('IGLMDA在交叉验证集上的AUC为：', auc(fp_test, tp_test))
    pre.append(auc(fp_test,tp_test))
print('整体准确率')
print(pre)
print(np.mean(pre))
# np.save('./N_mgat10/train_loss150.npy',arr=train_loss)
# np.savetxt('./N_mgat10/model_output150.csv', output.detach().numpy(), fmt='%.2f', delimiter=',')
# print('运行完成')
