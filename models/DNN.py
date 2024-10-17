import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import copy, time
from torch_geometric.nn import LightGCN
from data_utils import adjacency_to_edge, edge_to_adjacency, pred_to_adjacency
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h

class DNN_conti(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949, args=None):
        super(DNN_conti, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_indices_user = torch.arange(user_num)
        hidden_dim = 512
        output_dim = embedding_size_item
        print('emb:{} hid:{} out:{}'.format(embedding_size_item, hidden_dim, output_dim))
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h
    


class DNNCat(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, cat_dim=2):
        super(DNNCat, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.cat_layer = nn.Linear(cat_dim + 1, 1)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, x_U):
        #print('x:{} x_U:{}'.format(x.shape, x_U.shape))
        x = torch.cat([x.unsqueeze(-1), x_U], dim=2)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        #print('xIn:', x.shape)
        x = self.cat_layer(x)
        x = x.squeeze()
        #print('xO:', x.shape)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        #print('x3:{} emb:{}'.format(x.shape, emb.shape))
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h
    

class DNNCat2(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, cat_dim=2):
        super(DNNCat2, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        self.cat_layer = nn.Linear((cat_dim + 1) * in_dims[0], 1 * in_dims[0])
        #print('in_dims:{} cat:{}'.format(self.in_dims, self.cat_layer))

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, x_U):
        #print('x:{} x_U:{}'.format(x.shape, x_U.shape))
        x = torch.cat([x.unsqueeze(-1), x_U], dim=2)
        #print('xCat:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        x = x.view(x.shape[0], -1)
        #print('xIn:', x.shape)
        x = self.cat_layer(x)
        x = torch.tanh(x)
        x = x.squeeze()
        #print('xO:', x.shape)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        #print('x3:{} emb:{}'.format(x.shape, emb.shape))
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.relu(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.relu(h)
        
        return h

class DNNOneHot(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNNOneHot, self).__init__()
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, x_U):
        # print('x_U:', x_U.shape)
        x_U = x_U.reshape(x_U.shape[0], -1)
        # print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        h = torch.cat([x, emb], dim=-1)
        # import pdb
        # pdb.set_trace()
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        #print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        h = torch.cat([h, h_U], dim=1)
        #h = h_U
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
                #h = self.lrelu(h)
        
        return h

def nt_xent_loss(z1, z2, temperature=0.1, eps=1e-5):
    """
    z1, z2: 特征矩阵，形状均为[n, m]
    temperature: 温度参数，用于缩放softmax
    """
    n = z1.size(0)
    device = z1.device
    
    # 计算两组特征间的点积相似度矩阵
    sim_matrix = torch.mm(z1, z2.t()) / temperature
    
    # 对角线上的元素是相同样本在不同表示下的相似度，应设为-inf，避免自我比较
    mask = torch.eye(n, device=device).bool()
    #sim_matrix.masked_fill_(mask, float('-inf'))
    
    # 计算softmax，得到每个样本与其他所有样本的相似度分布
    sim_distribution = F.softmax(sim_matrix, dim=-1)
    
    # 只取非对角元素的平均，作为每对样本的负对数似然损失
    # 注意，这里实际上我们只关心非对角元，因为对角元已经被设为-inf，不会影响softmax后的结果
    loss = -torch.log(sim_distribution.diag() + eps).mean()

    negatives = sim_distribution.masked_select(~mask).view(n, -1)
    loss2 = -torch.log((torch.diag(sim_distribution) + eps) / negatives.sum(dim=1)).mean()
    # print('loss:', loss)
    # print('loss2:', loss2)
    #print('loss:{} loss2:{} neg_sum:{}'.format(loss, loss2, (negatives.sum(dim=1)).mean()))
    # import pdb
    # pdb.set_trace()
    return loss2

class DNNOneHotEmbedding(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949):
        super(DNNOneHotEmbedding, self).__init__()
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        embedding_size_item = embedding_size_item + in_dims_temp2[-1]
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_indices_user = torch.arange(user_num)
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        nn.init.xavier_uniform_(self.embedding_user.weight)
    
    def forward(self, x, timesteps, x_U, index=None, graph=None, RCloss=False):
        # print('x_U:', x_U.shape)
        x_U = x_U.reshape(x_U.shape[0], -1)
        # print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        h = torch.cat([x, emb], dim=-1)
        # import pdb
        # pdb.set_trace()
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        if RCloss:
            
            closs = nt_xent_loss(h, h_U)
        #print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        
        # import pdb
        # pdb.set_trace()
        if self.all_indices_item.device != x.device:
            self.all_indices_item = self.all_indices_item.to(x.device)
        
        all_embeddings_item = self.embedding_item(self.all_indices_item)
        index = index.to(x.device)
        all_embeddings_user = self.embedding_user(index)
        # print('user:', all_embeddings_user.shape)
        # # print('all_e:', all_embeddings.shape)
        
        h = torch.cat([h, h_U, all_embeddings_user], dim=1)
        #h = torch.matmul(h, all_embeddings.t())
        h = self.cosine_similarity_cuda(h, all_embeddings_item)
        #h = h_U
        
        # for i, layer in enumerate(self.out_layers):
        #     h = layer(h)
        #     if i != len(self.out_layers) - 1:
        #         h = torch.tanh(h)
                #h = self.lrelu(h)
        if RCloss:
            return h, closs
        return h
    
    def cosine_similarity_cuda(self, user_embeddings, item_embeddings):
        """
        计算用户embedding和物品embedding之间的余弦相似度。
        
        参数:
        - user_embeddings: 形状为[a, m]的Tensor，表示a个用户的m维embedding。
        - item_embeddings: 形状为[b, m]的Tensor，表示b个物品的m维embedding。
        
        返回:
        - similarity_matrix: 形状为[a, b]的Tensor，表示每个用户与每个物品之间的余弦相似度。
        """
        # 确保计算在CUDA设备上进行
        # user_embeddings = user_embeddings.cuda()
        # item_embeddings = item_embeddings.cuda()

        # 计算用户和物品向量的L2范数
        user_norms = torch.norm(user_embeddings, dim=1, keepdim=True)
        item_norms = torch.norm(item_embeddings, dim=1)

        # 计算余弦相似度
        dot_product = torch.mm(user_embeddings, item_embeddings.t())
        similarity_matrix = dot_product / (user_norms * item_norms.t())

        return similarity_matrix
    

class DNNOneHotEmbedding_conti(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949):
        super(DNNOneHotEmbedding_conti, self).__init__()
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        embedding_size_item = embedding_size_item + in_dims_temp2[-1]
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_indices_user = torch.arange(user_num)
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        nn.init.xavier_uniform_(self.embedding_user.weight)
    
    def forward(self, x, timesteps, x_U, index=None, graph=None, RCloss=False):
        # print('x_U:', x_U.shape)
        x_U = x_U.reshape(x_U.shape[0], -1)
        # print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        h = torch.cat([x, emb], dim=-1)
        # import pdb
        # pdb.set_trace()
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        if RCloss:
            
            closs = nt_xent_loss(h, h_U)
        #print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        
        # import pdb
        # pdb.set_trace()
        if self.all_indices_item.device != x.device:
            self.all_indices_item = self.all_indices_item.to(x.device)
        
        all_embeddings_item = self.embedding_item(self.all_indices_item)
        index = index.to(x.device)
        all_embeddings_user = self.embedding_user(index)
        # print('user:', all_embeddings_user.shape)
        # # print('all_e:', all_embeddings.shape)
        
        h = torch.cat([h_U, h_U, all_embeddings_user], dim=1)
        #h = torch.matmul(h, all_embeddings.t())
        h = self.cosine_similarity_cuda(h, all_embeddings_item)
        #h = h_U
        
        # for i, layer in enumerate(self.out_layers):
        #     h = layer(h)
        #     if i != len(self.out_layers) - 1:
        #         h = torch.tanh(h)
                #h = self.lrelu(h)
        if RCloss:
            return h, closs
        return h
    
    def cosine_similarity_cuda(self, user_embeddings, item_embeddings):
        """
        计算用户embedding和物品embedding之间的余弦相似度。
        
        参数:
        - user_embeddings: 形状为[a, m]的Tensor，表示a个用户的m维embedding。
        - item_embeddings: 形状为[b, m]的Tensor，表示b个物品的m维embedding。
        
        返回:
        - similarity_matrix: 形状为[a, b]的Tensor，表示每个用户与每个物品之间的余弦相似度。
        """
        # 确保计算在CUDA设备上进行
        # user_embeddings = user_embeddings.cuda()
        # item_embeddings = item_embeddings.cuda()

        # 计算用户和物品向量的L2范数
        user_norms = torch.norm(user_embeddings, dim=1, keepdim=True)
        item_norms = torch.norm(item_embeddings, dim=1)

        # 计算余弦相似度
        dot_product = torch.mm(user_embeddings, item_embeddings.t())
        similarity_matrix = dot_product / (user_norms * item_norms.t())

        return similarity_matrix
    
class DNNOneHotEmbeddingGCN_conti(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949, args=None):
        super(DNNOneHotEmbeddingGCN_conti, self).__init__()
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        self.args = args
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        embedding_size_item = embedding_size_item + in_dims_temp2[-1]
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_indices_user = torch.arange(user_num)
        hidden_dim = 512
        output_dim = embedding_size_item
        self.gcn_model = LayerGCN(embedding_size_item, hidden_dim, output_dim, residual=False, args=args).cuda()
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
        self.sumW = 1
        self.sumW = nn.Parameter(torch.tensor(1.0))
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        nn.init.xavier_uniform_(self.embedding_user.weight)
    
    def forward(self, x, timesteps, x_U, index=None, graph=None, RCloss=False):
        # print('x_U:', x_U.shape)
        ct = graph.argmax(dim=2)
        edge_index = torch.nonzero(ct).t().contiguous()
        edge_index[1, :] += ct.shape[0]
        x_U = x_U.reshape(x_U.shape[0], -1)
        # print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        h = torch.cat([x, emb], dim=-1)
        # import pdb
        # pdb.set_trace()
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        if RCloss:
            
            closs = nt_xent_loss(h, h_U)
        #print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        
        # import pdb
        # pdb.set_trace()
        if self.all_indices_item.device != x.device:
            self.all_indices_item = self.all_indices_item.to(x.device)
        
        all_embeddings_item = self.embedding_item(self.all_indices_item)
        index = index.to(x.device)
        all_embeddings_user = self.embedding_user(index)
        # print('user:', all_embeddings_user.shape)
        # # print('all_e:', all_embeddings.shape)
        
        #h = torch.cat([h_U, h_U, all_embeddings_user], dim=1)
        hc = torch.cat([h_U, h_U, all_embeddings_user], dim=1)
        #h = torch.matmul(h, all_embeddings.t())
        #print('emb_usr:{} emb_item:{}'.format(h.shape, all_embeddings_item.shape))
        all_embeddings = torch.cat([hc, all_embeddings_item], dim=0)
        #print('all_emb0:', all_embeddings.shape)
        if self.args.gcnLayerNum > 0:
            all_embeddings = self.gcn_model(all_embeddings, edge_index)
        hc = hc * self.sumW + all_embeddings[:hc.shape[0]] * (1 - self.sumW)
        hc = self.cosine_similarity_cuda(hc, all_embeddings_item)
        #h = torch.matmul(h, all_embeddings.t())
        #h = self.cosine_similarity_cuda(h, all_embeddings_item)
        #h = h_U
        
        # for i, layer in enumerate(self.out_layers):
        #     h = layer(h)
        #     if i != len(self.out_layers) - 1:
        #         h = torch.tanh(h)
                #h = self.lrelu(h)
        if RCloss:
            return hc, closs
        return hc
    
    def cosine_similarity_cuda(self, user_embeddings, item_embeddings):
        """
        计算用户embedding和物品embedding之间的余弦相似度。
        
        参数:
        - user_embeddings: 形状为[a, m]的Tensor，表示a个用户的m维embedding。
        - item_embeddings: 形状为[b, m]的Tensor，表示b个物品的m维embedding。
        
        返回:
        - similarity_matrix: 形状为[a, b]的Tensor，表示每个用户与每个物品之间的余弦相似度。
        """
        # 确保计算在CUDA设备上进行
        # user_embeddings = user_embeddings.cuda()
        # item_embeddings = item_embeddings.cuda()

        # 计算用户和物品向量的L2范数
        user_norms = torch.norm(user_embeddings, dim=1, keepdim=True)
        item_norms = torch.norm(item_embeddings, dim=1)

        # 计算余弦相似度
        dot_product = torch.mm(user_embeddings, item_embeddings.t())
        similarity_matrix = dot_product / (user_norms * item_norms.t())

        return similarity_matrix



from torch_geometric.nn import GCNConv, MessagePassing
class AggregationLayer(MessagePassing):
    def __init__(self, aggr='add'):
        super(AggregationLayer, self).__init__(aggr=aggr)
        # 无需定义可学习参数
        
    def forward(self, x, edge_index):
        # x 是节点特征，edge_index 是边的连接信息
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):  # x_j 是邻居节点的特征
        # 这里直接返回邻居节点特征，即实现平均聚合，因为默认的"add"聚合后会除以边的入度
        return x_j

class LightGCN(nn.Module):
    def __init__(self):
        super(LightGCN, self).__init__()
        self.conv1 = AggregationLayer()
        self.conv2 = AggregationLayer()
        
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)
        
    def forward(self, x, edge_index):
        # x 是节点特征，edge_index 是边的连接信息..
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x   

class LayerGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, residual=False,args=None):
        super(LayerGCN, self).__init__()
        self.args = args
        if self.args.gcnLayerNum == 1:
            self.conv1 = GCNConv(in_channels, out_channels)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            #print('in:{} hid:{} out:{}'.format(in_channels, hidden_channels, out_channels))
        self.lrelu = torch.nn.LeakyReLU(0.1)
        self.residual = residual
        
        # nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_uniform_(self.conv2.weight)
        
    def forward(self, x, edge_index):
        # x 是节点特征，edge_index 是边的连接信息..
        out = self.conv1(x, edge_index)
        if self.args.gcnLayerNum==2:
            out = torch.relu(out)
            out = self.lrelu(out)
            
            out = self.conv2(out, edge_index)
        if self.residual:
            out = out + x
        return out   

class DNNOneHotEmbeddingGCN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949,args=None):
        super(DNNOneHotEmbeddingGCN, self).__init__()
        self.args = args
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        embedding_size_item = embedding_size_item + in_dims_temp2[-1] # concat one hot
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_indices_user = torch.arange(user_num)
        hidden_dim = 512
        output_dim = embedding_size_item
        print('emb:{} hid:{} out:{}'.format(embedding_size_item, hidden_dim, output_dim))
        # import pdb
        # pdb.set_trace()
        self.gcn_model = LayerGCN(embedding_size_item, hidden_dim, output_dim, residual=False, args=args).cuda()
        #self.gcn_model = LightGCN().cuda()
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
        self.sumW = 1
        self.sumW = nn.Parameter(torch.tensor(1.0))
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        nn.init.xavier_uniform_(self.embedding_user.weight)
    
    def forward(self, x, timesteps, x_U, index=None, graph=None, RCloss=False):
        # print('x_U:', x_U.shape)
        # print('x:{} x_U:{}'.format(x.shape, x_U.shape))
        # if self.args.noise_type == 1:
        #     x=x_U[:,:,0] #去掉连续
        # elif self.args.noise_type == 2:
        #     x_U[:,:,0]=x #去掉离散 消融1
        #     x_U[:,:,1]=x

        #graph = x_U.long()
        ct = graph.argmax(dim=2)
        edge_index = torch.nonzero(ct).t().contiguous()
        edge_index[1, :] += ct.shape[0]
        #print('edge_index:', edge_index.shape)
        #print('ct:{} edge_index:{}'.format(ct.shape, edge_index.shape))
        # import pdb
        # pdb.set_trace()
        x_U = x_U.reshape(x_U.shape[0], -1)
        #print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        #y = x_U[:,:]
        #print(x.shape, x_U.shape)
        if self.args.noise_type == 1:
            h = torch.cat([x_U[:,:x.shape[1]], emb], dim=-1)
        else:
            h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        
        if self.args.noise_type == 2:
            h_U = torch.cat([x, x, emb], dim=-1)
        else:
            h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        # print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        
        if RCloss:
            closs = nt_xent_loss(h, h_U)
            if self.args.noise_type != 0:
                closs *= 0
        if self.all_indices_item.device != x.device:
            self.all_indices_item = self.all_indices_item.to(x.device)
        
        all_embeddings_item = self.embedding_item(self.all_indices_item)
        index = index.to(x.device)
        all_embeddings_user = self.embedding_user(index)
        # print('user:', all_embeddings_user.shape)
        # # print('all_e:', all_embeddings.shape)
        # if self.args.noise_type == 1:
        #     hc = torch.cat([h, h, all_embeddings_user], dim=1)
        # elif self.args.noise_type == 2:  
        #     hc = torch.cat([h_U, h_U, all_embeddings_user], dim=1)
        # else:
        #     hc = torch.cat([h, h_U, all_embeddings_user], dim=1)
        hc = torch.cat([h, h_U, all_embeddings_user], dim=1)
        #h = torch.matmul(h, all_embeddings.t())
        #print('emb_usr:{} emb_item:{}'.format(h.shape, all_embeddings_item.shape))
        all_embeddings = torch.cat([hc, all_embeddings_item], dim=0)
        #print('all_emb0:', all_embeddings.shape)
        if self.args.gcnLayerNum > 0:
            all_embeddings = self.gcn_model(all_embeddings, edge_index)
        #print('all_emb1:', all_embeddings.shape)
        # import pdb
        # pdb.set_trace()
        #
        
        #h = self.cosine_similarity_cuda(all_embeddings[:h.shape[0]], all_embeddings_item)

        hc = hc * self.sumW + all_embeddings[:hc.shape[0]] * (1 - self.sumW)
        hc = self.cosine_similarity_cuda(hc, all_embeddings_item)

        #h = self.cosine_similarity_cuda(h, all_embeddings[h.shape[0]:])
        #h = self.cosine_similarity_cuda(all_embeddings[:h.shape[0]], all_embeddings[h.shape[0]:])
        #h = h_U
        
        # for i, layer in enumerate(self.out_layers):
        #     h = layer(h)
        #     if i != len(self.out_layers) - 1:
        #         h = torch.tanh(h)
                #h = self.lrelu(h)
        if RCloss:
            return hc, closs
        return hc
    
    def cosine_similarity_cuda(self, user_embeddings, item_embeddings):
        """
        计算用户embedding和物品embedding之间的余弦相似度。
        
        参数:
        - user_embeddings: 形状为[a, m]的Tensor，表示a个用户的m维embedding。
        - item_embeddings: 形状为[b, m]的Tensor，表示b个物品的m维embedding。
        
        返回:
        - similarity_matrix: 形状为[a, b]的Tensor，表示每个用户与每个物品之间的余弦相似度。
        """
        # 确保计算在CUDA设备上进行
        # user_embeddings = user_embeddings.cuda()
        # item_embeddings = item_embeddings.cuda()

        # 计算用户和物品向量的L2范数
        user_norms = torch.norm(user_embeddings, dim=1, keepdim=True)
        item_norms = torch.norm(item_embeddings, dim=1)

        # 计算余弦相似度
        dot_product = torch.mm(user_embeddings, item_embeddings.t())
        similarity_matrix = dot_product / (user_norms * item_norms.t())

        return similarity_matrix
 

class DNNOneHotEmbeddingGCN_time(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, item_num=2810, user_num=5949,args=None):
        super(DNNOneHotEmbeddingGCN, self).__init__()
        self.args = args
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] += self.in_dims2[-1]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        print('in_dims_temp:', in_dims_temp)
        embedding_size_user = in_dims_temp[-1]
        embedding_size_item = in_dims_temp[-1] + embedding_size_user
        embedding_size_item = embedding_size_item + in_dims_temp2[-1] # concat one hot
        self.embedding_item = nn.Embedding(item_num, embedding_size_item)
        self.embedding_user = nn.Embedding(user_num, embedding_size_user)
        self.all_indices_item = torch.arange(item_num)
        self.all_embeddings_item = self.embedding_item(self.all_indices_item)
        self.all_embeddings_item = self.all_embeddings_item.cuda()
        self.all_indices_user = torch.arange(user_num)
        hidden_dim = 512
        output_dim = embedding_size_item
        print('emb:{} hid:{} out:{}'.format(embedding_size_item, hidden_dim, output_dim))
        # import pdb
        # pdb.set_trace()
        self.gcn_model = LayerGCN(embedding_size_item, hidden_dim, output_dim, residual=False, args=args).cuda()
        #self.gcn_model = LightGCN().cuda()
        self.init_weights()
        self.lrelu = torch.nn.LeakyReLU(0.1)
        self.sumW = 1
        self.sumW = nn.Parameter(torch.tensor(1.0))
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        nn.init.xavier_uniform_(self.embedding_item.weight)
        nn.init.xavier_uniform_(self.embedding_user.weight)
    
    def forward(self, x, timesteps, x_U, index=None, graph=None, RCloss=False):
        # print('x_U:', x_U.shape)
        # print('x:{} x_U:{}'.format(x.shape, x_U.shape))
        # if self.args.noise_type == 1:
        #     x=x_U[:,:,0] #去掉连续
        # elif self.args.noise_type == 2:
        #     x_U[:,:,0]=x #去掉离散 消融1
        #     x_U[:,:,1]=x
        all_embeddings_user = self.embedding_user(index)
        #graph = x_U.long()
        log_time = False
        if log_time:
            time_start = time.time()
        ct = graph.argmax(dim=2)
        edge_index = torch.nonzero(ct).t().contiguous()
        edge_index[1, :] += ct.shape[0]
        if log_time:
            time_graph = time.time()
            print('time_graph:', time_graph - time_start)
        #print('edge_index:', edge_index.shape)
        #print('ct:{} edge_index:{}'.format(ct.shape, edge_index.shape))
        # import pdb
        # pdb.set_trace()
        x_U = x_U.reshape(x_U.shape[0], -1)
        #print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        if log_time:
            time_temb = time.time()
            print('time_temb:', time_temb - time_graph)
        #y = x_U[:,:]
        #print(x.shape, x_U.shape)
        if self.args.noise_type == 1:
            h = torch.cat([x_U[:,:x.shape[1]], emb], dim=-1)
        else:
            h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            #h = self.lrelu(h)

        
        if self.args.noise_type == 2:
            h_U = torch.cat([x, x, emb], dim=-1)
        else:
            h_U = torch.cat([x_U, emb], dim=-1)
        for i, layer in enumerate(self.in_layers2):
            h_U = layer(h_U)
            h_U = torch.tanh(h_U)
            #h_U = self.lrelu(h_U)
        # print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        if log_time:
            time_mlp = time.time()
            print('time_mlp:', time_mlp - time_temb)
        if RCloss:
            closs = nt_xent_loss(h, h_U)
            if self.args.noise_type != 0:
                closs *= 0
        if log_time:
            time_RC = time.time()
            print('time_RC:', time_RC - time_mlp)
        # if self.all_indices_item.device != x.device:
        #     self.all_indices_item = self.all_indices_item.to(x.device)
        
        #all_embeddings_item = self.embedding_item(self.all_indices_item)
        all_embeddings_item = self.all_embeddings_item
        if log_time:
            time_sel1 = time.time()
            print('time_sel1:', time_sel1 - time_RC)
        #index = index.to(x.device)
        time_sel2 = time.time()
        if log_time:
            
            print('time_sel2:', time_sel2 - time_sel1)
        # print('user:', all_embeddings_user.shape)
        # # print('all_e:', all_embeddings.shape)
        # if self.args.noise_type == 1:
        #     hc = torch.cat([h, h, all_embeddings_user], dim=1)
        # elif self.args.noise_type == 2:  
        #     hc = torch.cat([h_U, h_U, all_embeddings_user], dim=1)
        # else:
        #     hc = torch.cat([h, h_U, all_embeddings_user], dim=1)
        hc = torch.cat([h, h_U, all_embeddings_user], dim=1)
        #h = torch.matmul(h, all_embeddings.t())
        #print('emb_usr:{} emb_item:{}'.format(h.shape, all_embeddings_item.shape))
        all_embeddings = torch.cat([hc, all_embeddings_item], dim=0)
        #print('all_emb0:', all_embeddings.shape)
        if self.args.gcnLayerNum > 0:
            all_embeddings = self.gcn_model(all_embeddings, edge_index)
        #if log_time:
        time_gcn = time.time()
        #print('time gcn:', time_gcn - time_sel2)
        #print('all_emb1:', all_embeddings.shape)
        # import pdb
        # pdb.set_trace()
        #
        
        #h = self.cosine_similarity_cuda(all_embeddings[:h.shape[0]], all_embeddings_item)

        hc = hc * self.sumW + all_embeddings[:hc.shape[0]] * (1 - self.sumW)
        hc = self.cosine_similarity_cuda(hc, all_embeddings_item)
        if log_time:
            time_sim = time.time()
            print('time sim:', time_sim - time_gcn)
            print('time model:', time_sim - time_start)
        #h = self.cosine_similarity_cuda(h, all_embeddings[h.shape[0]:])
        #h = self.cosine_similarity_cuda(all_embeddings[:h.shape[0]], all_embeddings[h.shape[0]:])
        #h = h_U
        
        # for i, layer in enumerate(self.out_layers):
        #     h = layer(h)
        #     if i != len(self.out_layers) - 1:
        #         h = torch.tanh(h)
                #h = self.lrelu(h)
        if RCloss:
            return hc, closs
        return hc
    
    def cosine_similarity_cuda(self, user_embeddings, item_embeddings):
        """
        计算用户embedding和物品embedding之间的余弦相似度。
        
        参数:
        - user_embeddings: 形状为[a, m]的Tensor，表示a个用户的m维embedding。
        - item_embeddings: 形状为[b, m]的Tensor，表示b个物品的m维embedding。
        
        返回:
        - similarity_matrix: 形状为[a, b]的Tensor，表示每个用户与每个物品之间的余弦相似度。
        """
        # 确保计算在CUDA设备上进行
        # user_embeddings = user_embeddings.cuda()
        # item_embeddings = item_embeddings.cuda()

        # 计算用户和物品向量的L2范数
        user_norms = torch.norm(user_embeddings, dim=1, keepdim=True)
        item_norms = torch.norm(item_embeddings, dim=1)

        # 计算余弦相似度
        dot_product = torch.mm(user_embeddings, item_embeddings.t())
        similarity_matrix = dot_product / (user_norms * item_norms.t())

        return similarity_matrix
    
  

class DNNOneHotTransformer(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNNOneHotTransformer, self).__init__()
        self.in_dims = in_dims
        self.in_dims2 = copy.deepcopy(in_dims)
        self.in_dims2[0] *= 2
        #self.in_dims2[-1] = 100
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        
        #self.fc_out = nn.Linear(in_dims[0], input_dim)

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            in_dims_temp2 = [self.in_dims2[0] + self.time_emb_dim] + self.in_dims2[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        out_dims_temp[0] = in_dims_temp2[0] + in_dims_temp[0]
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        
        self.in_layers2 = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp2[:-1], in_dims_temp2[1:])])

        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.encoder_layer = TransformerEncoderLayer(d_model=in_dims_temp[0], nhead=2, dropout=dropout, dim_feedforward=in_dims_temp[-1])
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=2)
        self.encoder_layer2 = TransformerEncoderLayer(d_model=in_dims_temp2[0], nhead=2, dropout=dropout, dim_feedforward=in_dims_temp2[-1])
        self.transformer_encoder2 = TransformerEncoder(encoder_layer=self.encoder_layer2, num_layers=2)
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.in_layers2:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, x_U):
        # print('x_U:', x_U.shape)
        x_U = x_U.reshape(x_U.shape[0], -1)
        # print('x_U2:', x_U.shape)
        # print('x:', x.shape)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
            x_U = F.normalize(x_U)
        x = self.drop(x)
        x_U = self.drop(x_U)
        h = torch.cat([x, emb], dim=-1)
        
        h = self.transformer_encoder(h)
        # for i, layer in enumerate(self.in_layers):
        #     h = layer(h)
        #     h = torch.tanh(h)

        h_U = torch.cat([x_U, emb], dim=-1)
        h_U = self.transformer_encoder2(h_U)

        # for i, layer in enumerate(self.in_layers2):
        #     h_U = layer(h_U)
        #     h_U = torch.tanh(h_U)
        #print('h:{} h_U:{}'.format(h.shape, h_U.shape))
        h = torch.cat([h, h_U], dim=1)
        #h = h_U
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h

class DNNlightGCN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, num_nodes=5949+2810,e_f=None):
        super(DNNlightGCN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.gcn = LightGCN(
            num_nodes=num_nodes,
            embedding_dim=64,
            num_layers=2,
        )
        self.e_f = e_f

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        # import pdb
        # pdb.set_trace()
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, index):
        e_t = adjacency_to_edge(x, index)
        # print('e_t:', e_t.shape)
        # convert_e = edge_to_adjacency(e_t, index)
        # print('x_t:{} e_t:{} convert_e:{}'.format(x.shape, e_t.shape, convert_e.shape))
        # print('equ:', (x.cpu() == convert_e).sum())
        predl = self.gcn.predict_link(edge_index=self.e_f.to(x.device), edge_label_index=e_t.to(x.device))
        xp = pred_to_adjacency(e_t, index, pred=predl).to(x.device)
        x = xp
        # print('predl:', predl.shape)
        # print('xp:', xp.shape)
        # import pdb
        # pdb.set_trace()
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h
    
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
