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
from model import IGMDA,IGLMDA
from MF import matrix_factorization
from sklearn.metrics import roc_curve,auc

import torch

A = np.load('.\data\HMDD v3.2\miRNA-disease association.npy')
m_number,d_number = A.shape
print('The number of miRNA and disease: ',A.shape)
print('The number of relationships between miRNA and disease: ',sum(sum(A)))

sim_m = np.loadtxt('.\data\HMDD v3.2\sim_m')
sim_d = np.loadtxt('.\data\HMDD v3.2\sim_d')

sim_m_0 = set_digo_zero(sim_m,0)
sim_d_0 = set_digo_zero(sim_d,0)

m_similarity = preprocess_feature(sim_m_0,m_number)
d_similarity = preprocess_feature(sim_d_0,d_number)

m_similarity = set_digo_zero(m_similarity,0)
d_similarity = set_digo_zero(d_similarity,0)
m_similarity = m_similarity + np.eye(788)
d_similarity = d_similarity + np.eye(374)

miRNA_id,disease_id = get_position(A)
print(miRNA_id.size())
print(miRNA_id.type())


# Construct a heterogeneous map
g = dgl.heterograph({
    ('miRNA','md_association','disease'):(miRNA_id,disease_id),
    ('disease','dm_association','miRNA'):(disease_id,miRNA_id)
})
print(g)
print(g.ntypes)

# Construct isomorphic graph (the aim is to generate isomorphic graph with different step sizes)

# Perform a random walk on a heterogeneous graph
# Generate 1-hop nodes of miRNA
sample_mm_hop1 = get_m_hop1_rw(g,m_number)
m_adj = get_isom_adj(m_number,sample_mm_hop1)
m_start, m_end = get_position(m_adj)
g_m = dgl.DGLGraph()
g_m.add_nodes(m_number)
m_start = m_start.type(torch.int64)
m_end = m_end.type(torch.int64)
g_m.add_edges(m_end,m_start)
m_similarity = torch.from_numpy(m_similarity)
g_m.ndata['feature'] = m_similarity
print(g_m)
# Messaging on 1-hop miRNA isomorphism graph
import dgl.function as fn
g_m.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
m_similarity1 = g_m.ndata['feature']
new_feature_m = m_similarity1.numpy()


# Generate 1-hop nodes of disease
sample_dd_hop1 = get_d_hop1_rw(g,d_number)
d_adj = get_isom_adj(d_number,sample_dd_hop1)
d_start, d_end = get_position(d_adj)
g_d = dgl.DGLGraph()
g_d.add_nodes(d_number)
d_start = d_start.type(torch.int64)
d_end = d_end.type(torch.int64)
g_d.add_edges(d_end,d_start)
d_similarity = torch.from_numpy(d_similarity)
g_d.ndata['feature'] = d_similarity
# Messaging on 1-hop disease isomorphism graph
g_d.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
d_similarity1 = g_d.ndata['feature']
new_feature_d = d_similarity1.numpy()

# Generate 2-hop nodes of miRNA
sample_mm_hop2 = get_m_hop2_rw(g,m_number)
m_adj_2 = get_isom_adj(m_number,sample_mm_hop2)
m_start_2, m_end_2 = get_position(m_adj_2)
g_m_2 = dgl.DGLGraph()
g_m_2.add_nodes(m_number)
m_start_2 = m_start_2.type(torch.int64)
m_end_2 = m_end_2.type(torch.int64)
g_m_2.add_edges(m_end_2,m_start_2)
g_m_2.ndata['feature'] = m_similarity
# Messaging on 2-hop miRNA isomorphism graph
g_m_2.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
m_similarity2 = g_m_2.ndata['feature']
new_feature_m_2 = m_similarity2.numpy()

# Generate 2-hop nodes of disease
sample_dd_hop2 = get_d_hop2_rw(g,d_number)
d_adj_2 = get_isom_adj(d_number,sample_dd_hop2)
d_start_2, d_end_2 = get_position(d_adj_2)
g_d_2 = dgl.DGLGraph()
g_d_2.add_nodes(d_number)
d_start_2 = d_start_2.type(torch.int64)
d_end_2 = d_end_2.type(torch.int64)
g_d_2.add_edges(d_end_2,d_start_2)
g_d_2.ndata['feature'] = d_similarity
# Messaging on 2-hop disease isomorphism graph
g_d_2.update_all(fn.copy_u('feature','m'),
               fn.mean('m','feature'))
d_similarity2 = g_d_2.ndata['feature']
new_feature_d_2 = d_similarity2.numpy()

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Get the training set and the test set
samples = get_all_the_samples(A) # Obtain a total sample
sample_label = choose_adj_value(A,samples)
c, label = get_adj(A.shape[0],A.shape[1],samples)
kf = KFold(n_splits=10,shuffle=True)

model1 = IGMDA(in_features_m=torch.tensor(788),
            in_features_d=torch.tensor(374),
            out_features=torch.tensor(100),
            n_channels=torch.tensor(3),
            dropout=0.3,
            nheads =3)
model1 = model1.cuda()
# Get a linear representation
A = np.array(A)
#U, V, result = matrix_factorization(A, 50, aplha=0.006, beta=0.00002, steps=100)
U, V,  = matrix_factorization(A, 50, aplha=0.021, beta=0.00002, steps=100)

model2 = IGLMDA(in_features_m=torch.tensor(788),
            in_features_d=torch.tensor(374),
            out_features=torch.tensor(50),
            h_features= torch.tensor(50),
            n_channels=torch.tensor(3),
            dropout=0.1,
            nheads=3)

model2 = model2.cuda()

# Choose model
model = model2
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
    print('The number of relationships between miRNA and Disease :', sum(sum(train_set_adj)))
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
    U = U.to(device)
    V = V.to(device)
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

    # Training
    epoch = 150
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(m_similarity,
                       m_similarity1,
                       m_similarity2,
                       U,
                       d_similarity,
                       d_similarity1,
                       d_similarity2,
                       V,
                       train_set_adj)

        pred_value = choose_adj_value(output,train_set)
        pred_value = torch.stack(pred_value)
        pred_value = pred_value.to(torch.float32)
        pred_value = pred_value.to(device)
        train_set_label = torch.tensor(train_set_label)
        train_set_label = train_set_label.to(torch.float32)
        train_set_label = train_set_label.to(device)
        loss = loss_fn(pred_value,train_set_label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    print(train_loss)
    output = output.cpu()
    test_set_adj,test_set_label = get_adj(A.shape[0],A.shape[1],test_set) #The training is complete and validated on the test set
    print(test_set_label)
    print(len(test_set_label))
    test_sample_score = choose_adj_value(output.detach().numpy(), test_set)

    fp_test, tp_test, threshold_test = roc_curve(test_set_label, test_sample_score)
    print('The AUC of IGLMDA on the cross-validation set is :', auc(fp_test, tp_test))
    pre.append(auc(fp_test,tp_test))
np.save('Train_loss.npy',arr=train_loss)
np.savetxt('Output.csv', output.detach().numpy(), fmt='%.2f', delimiter=',')
print(pre)
print('Overall accuracy = ', np.mean(pre))
