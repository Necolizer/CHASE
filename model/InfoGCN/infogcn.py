import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .modules import import_class, bn_init, EncodingBlock


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, num_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=1, config=None):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        self.in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.num_class = num_class
        self.num_point = num_point
        self.noise_ratio = noise_ratio

        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * num_point)
        self.A_vector = self.get_A(graph, k)
        
        self.to_joint_embedding = nn.Linear(num_channels, self.in_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.in_channels))

        self.blocks = nn.ModuleList()
        for _, (blk_in_channels, blk_out_channels, blk_stride) in enumerate(config):
            self.blocks.append(EncodingBlock(blk_in_channels, blk_out_channels, A, stride=blk_stride))

        self.fc = nn.Linear(self.out_channels, self.out_channels)
        self.fc_mu = nn.Linear(self.out_channels, self.out_channels)
        self.fc_logvar = nn.Linear(self.out_channels, self.out_channels)
        self.decoder = nn.Linear(self.out_channels, self.num_class)

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node, dtype=np.float32)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        for i, block in enumerate(self.blocks):
            x = block(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return (y_hat, z) if self.training else y_hat
        

class InfoGCN_GROUP(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, num_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=1, config=None, num_class_person=5):
        super(InfoGCN_GROUP, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        self.in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.num_class = num_class
        self.num_point = num_point
        self.noise_ratio = noise_ratio

        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * num_point)
        self.A_vector = self.get_A(graph, k)
        
        self.to_joint_embedding = nn.Linear(num_channels, self.in_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.in_channels))

        self.blocks = nn.ModuleList()
        for _, (blk_in_channels, blk_out_channels, blk_stride) in enumerate(config):
            self.blocks.append(EncodingBlock(blk_in_channels, blk_out_channels, A, stride=blk_stride))

        self.fc = nn.Linear(self.out_channels, self.out_channels)
        self.fc_mu = nn.Linear(self.out_channels, self.out_channels)
        self.fc_logvar = nn.Linear(self.out_channels, self.out_channels)
        self.decoder = nn.Linear(self.out_channels, self.num_class)

        self.fc_person = nn.Linear(self.out_channels, num_class_person)
        nn.init.normal_(self.fc_person.weight, 0, math.sqrt(2. / num_class_person))

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node, dtype=np.float32)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        for i, block in enumerate(self.blocks):
            x = block(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3)
        xm = x
        x = x.mean(1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return (y_hat, z, self.fc_person(xm)) if self.training else y_hat


class InfoGCN_GROUP_VOL(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, num_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=1, config=None, num_class_person=5):
        super(InfoGCN_GROUP_VOL, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        self.in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.num_class = num_class
        self.num_point = num_point
        self.noise_ratio = noise_ratio

        self.data_bn = nn.BatchNorm1d(num_person * self.in_channels * num_point)
        self.A_vector = self.get_A(graph, k)
        
        self.to_joint_embedding = nn.Linear(num_channels, self.in_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.in_channels))

        self.blocks = nn.ModuleList()
        for _, (blk_in_channels, blk_out_channels, blk_stride) in enumerate(config):
            self.blocks.append(EncodingBlock(blk_in_channels, blk_out_channels, A, stride=blk_stride))

        self.fc = nn.Linear(self.out_channels*2, self.out_channels)
        self.fc_mu = nn.Linear(self.out_channels, self.out_channels)
        self.fc_logvar = nn.Linear(self.out_channels, self.out_channels)
        self.decoder = nn.Linear(self.out_channels, self.num_class)

        self.fc_person = nn.Linear(self.out_channels, num_class_person)
        nn.init.normal_(self.fc_person.weight, 0, math.sqrt(2. / num_class_person))

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / self.num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node, dtype=np.float32)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x

        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        x = self.data_bn(x)
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()

        for i, block in enumerate(self.blocks):
            x = block(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        xm = x.mean(3)
        x_left = xm[:, :N//2, :]
        x_right = xm[:, N//2:, :]
        x_left = x_left.mean(1)
        x_right = x_right.mean(1)
        x = torch.cat([x_left, x_right], dim=-1)
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return (y_hat, z, self.fc_person(xm)) if self.training else y_hat