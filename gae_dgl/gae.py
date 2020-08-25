import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.inputs = in_feats
        self.outputs=  out_feats

    def __str__(self):
        return "NodeApply with {inp} inputs, {out} outputs, {act} activation".format(inp = self.inputs,out = self.outputs, act = self.activation)

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')  # sum aggregation
GCN = GraphConv

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super(GAE, self).__init__()
        layers = [GCN(in_dim, hidden_dims[0], activation =F.relu)]
        if len(hidden_dims)>=2:
            layers = [GCN(in_dim, hidden_dims[0], activation =F.relu)]
            for i in range(1,len(hidden_dims)):
                if i != len(hidden_dims)-1:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], activation = F.relu))
                else:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], activation =lambda x:x))
        else:
            layers = [GCN(in_dim, hidden_dims[0], lambda x:x)]
            
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)

        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        return h



class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dims,zdim):
        super(VGAE, self).__init__()
        layers = [GCN(in_dim, hidden_dims[0], activation =F.relu)]
        if len(hidden_dims)>=2:
            layers = [GCN(in_dim, hidden_dims[0], activation =F.relu)]
            for i in range(1,len(hidden_dims)):
                if i != len(hidden_dims)-1:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], activation = F.relu))
                else:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], activation =lambda x:x))
        else:
            layers = [GCN(in_dim, hidden_dims[0], lambda x:x)]
        
        self.zdim = zdim
        self.layers = nn.ModuleList(layers)
        self.z_mean = GCN(hidden_dims[-1],zdim)
        self.z_log_std = GCN(hidden_dims[-1],zdim)

        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)

        z_mean = self.z_mean(g,h) 
        z_log_std = self.z_log_std(g,h)
        
        z = z_mean + torch.normal(torch.zeros(self.zdim),torch.ones(self.zdim)) * z_log_std

        g.ndata['h'] = z
        adj_rec = self.decoder(z)
        return (z_mean,z_log_std),adj_rec

    def encode(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)

        z = self.z_mean(g,h) + torch.normal(0,1) * self.z_log_std(g,h)
        return z


    def _KLDiv(self,z_mean,z_log_std):
        print(z_mean.shape)
        exit()
        return 0.5 * torch.sum(torch.exp(z_log_std)**2 +z_mean**2 -1 -z_log_std)

    def vgae_loss(self,z,output,target, norm, pos_weight):
        
        z_mean,z_log_std = z
        
        loss = norm*BCELoss(output,target, reduction ="mean", pos_weight = pos_weight) - self._KLDiv(*z)
        return loss



class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj

  

