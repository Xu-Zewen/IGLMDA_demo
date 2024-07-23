from torch import nn
import torch.nn.functional as F
import torch

def mish(x):
    return x * (torch.tanh(F.softplus(x)))

class IGMDALayer(nn.Module):
    def __init__(self,
                 in_features_m,
                 in_features_d,
                 out_features,
                 channels,
                 dropout):
        super(IGMDALayer, self).__init__()
        self.dropout = dropout
        self.in_features_m = in_features_m
        self.in_features_d = in_features_d
        self.out_features = out_features
        self.channels = channels
        self.w_m= nn.Parameter(torch.zeros(size=(self.in_features_m,
                                                 self.out_features)))
        nn.init.xavier_uniform_(self.w_m)
        self.w_d = nn.Parameter(torch.zeros(size=(self.in_features_d,
                                                  self.out_features)))
        nn.init.xavier_uniform_(self.w_d)
        self.a_m = nn.Parameter(torch.zeros(size=(1,self.channels)))
        nn.init.xavier_uniform_(self.a_m)
        self.a_d = nn.Parameter(torch.zeros(size=(1,self.channels)))
        nn.init.xavier_uniform_(self.a_d)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_features,1)))
        nn.init.xavier_uniform_(self.a)

    def forward(self,m,m1,m2,d,d1,d2,adj):
        m_combine = torch.cat((m, m1, m2), 0).view(self.channels, -1)
        h_m = torch.mm(self.a_m, m_combine)
        d_combine = torch.cat((d, d1, d2), 0).view(self.channels, -1)
        h_d = torch.mm(self.a_d, d_combine)
        h1qq = torch.mm(h_m.view(-1,self.in_features_m),self.w_m)
        h2qq = torch.mm(h_d.view(-1,self.in_features_d),self.w_d)
        N1 = h1qq.size()[0]
        N2 = h2qq.size()[0]
        a_input = torch.cat([h1qq.repeat(1,N2).view(N1*N2,-1),h2qq.repeat(N1,1)],dim=1).view(N1,-1,2*self.out_features)
        e = mish(torch.matmul(a_input,self.a).squeeze(2))
        return e

class IGMDA(nn.Module):
    def __init__(self,in_features_m,in_features_d,out_features,n_channels,dropout,nheads):
        super(IGMDA, self).__init__()
        self.nheads = nheads
        self.attentions = [IGMDALayer(in_features_m,in_features_d,out_features,n_channels,dropout)for _ in range(nheads)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

    def forward(self, x,x_1,x_2,y,y_1,y_2,adj):

        col = adj.size()[0]
        row = adj.size()[1]
        x = torch.cat([att(x,x_1,x_2,y,y_1,y_2,adj) for att in self.attentions], dim=0).reshape(self.nheads,col,row)
        output = x.mean(0)
        return output


class IGLMDALayer(nn.Module):
    def __init__(self,
                 in_features_m,
                 in_features_d,
                 out_features,
                 h_features,
                 channels,
                 dropout=0.1):
        super(IGLMDALayer, self).__init__()
        self.dropout = dropout
        self.in_features_m = in_features_m
        self.in_features_d = in_features_d
        self.out_features = out_features
        self.h_features = h_features
        self.channels = channels
        self.w_m= nn.Parameter(torch.zeros(size=(self.in_features_m,
                                                 self.out_features)))
        nn.init.xavier_uniform_(self.w_m)
        self.w_d = nn.Parameter(torch.zeros(size=(self.in_features_d,
                                                  self.out_features)))
        nn.init.xavier_uniform_(self.w_d)
        self.a_m = nn.Parameter(torch.zeros(size=(1,self.channels)))
        nn.init.xavier_uniform_(self.a_m)
        self.a_d = nn.Parameter(torch.zeros(size=(1,self.channels)))
        nn.init.xavier_uniform_(self.a_d)
        self.a = nn.Parameter(torch.zeros(size=(2*(self.out_features + self.h_features),1)))
        nn.init.xavier_uniform_(self.a)

    def forward(self,m,m1,m2,m_h,d,d1,d2,d_h):
        m_combine = torch.cat((m, m1, m2), 0).view(self.channels, -1)
        h_m = torch.mm(self.a_m, m_combine)
        d_combine = torch.cat((d, d1, d2), 0).view(self.channels, -1)
        h_d = torch.mm(self.a_d, d_combine)
        h1qq = torch.mm(h_m.view(-1,self.in_features_m),self.w_m)
        h2qq = torch.mm(h_d.view(-1,self.in_features_d),self.w_d)
        h1qq = torch.cat((h1qq,m_h),1)
        h2qq = torch.cat((h2qq,d_h),1)
        N1 = h1qq.size()[0]
        N2 = h2qq.size()[0]
        a_input = torch.cat([h1qq.repeat(1,N2).view(N1*N2,-1),
                             h2qq.repeat(N1,1)],
                            dim=1).view(N1,-1,2*(self.out_features + self.h_features))
        s = mish(torch.matmul(a_input,self.a).squeeze(2))
        return s

class IGLMDA(nn.Module):
    def __init__(self,in_features_m,in_features_d,out_features,h_features,n_channels,dropout,nheads):
        super(IGLMDA, self).__init__()
        self.nheads = nheads
        self.attentions = [IGLMDALayer(in_features_m,in_features_d,out_features,h_features,n_channels,dropout)for _ in range(nheads)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

    def forward(self, x, x_1, x_2, U, y, y_1, y_2, V, adj):

        col = adj.size()[0]
        row = adj.size()[1]
        x = torch.cat([att(x,x_1,x_2,U,y,y_1,y_2,V,adj) for att in self.attentions], dim=0).reshape(self.nheads,col,row)
        output = x.mean(0)
        return output
