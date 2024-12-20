from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn import MessagePassing
from attention_conv import my_conv


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class HMSL(MessagePassing):
    def __init__(self, hidden, out_channels, data, concat, target_node):
        super().__init__(aggr='sum')
        self.concat_num = concat
        self.target = target_node
        self.loss_co = TripletLoss(margin=0.3)

        self.lin_dict = torch.nn.ModuleDict()
        self.lin_dict_mean = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden)
            self.lin_dict_mean[node_type] = Linear(hidden, hidden)
        self.lin_out = Linear(hidden, out_channels)

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()

        for node_type in data.node_types:
            self.k_lin[node_type] = Linear(hidden, hidden)
            self.q_lin[node_type] = Linear(hidden, hidden)
            self.v_lin[node_type] = Linear(hidden, hidden)

        self.conv = my_conv(hidden, hidden, metadata=data.metadata())
        self.conv1 = my_conv(hidden, hidden, metadata=data.metadata())

    def reset_parameters(self):
        reset(self.lin_dict)
        reset(self.lin_dict_mean)
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.conv)
        reset(self.conv1)
        self.lin_out.reset_parameters()

    def forward(self, x_dict, edge_index):
        CA_hidden_ls = []
        EOA_hidden_ls = []

        ca_out_ls = []
        eoa_out_ls = []
        out_dict = {}
        for k in range(self.concat_num):
            x_dict_ = {
                node_type: F.tanh(self.lin_dict[node_type](x[k]))
                for node_type, x in x_dict.items()
            }
            CA_hidden_ls.append(x_dict_['CA'])
            EOA_hidden_ls.append(x_dict_['EOA'])
            CA_h = torch.mean(CA_hidden_ls[k], dim=0)
            CA_h = F.tanh(self.lin_dict_mean['CA'](CA_h))
            EOA_h = torch.mean(EOA_hidden_ls[k], dim=0)
            EOA_h = F.tanh(self.lin_dict_mean['EOA'](EOA_h))
            ca_out_ls.append(CA_h)
            eoa_out_ls.append(EOA_h)

        out_dict['CA'] = ca_out_ls
        out_dict['EOA'] = eoa_out_ls

        for node_type, x in out_dict.items():
            out_dict[node_type] = self.attention(x, node_type)

        out_dict = self.conv(out_dict, edge_index)
        out_dict = self.conv1(out_dict, edge_index)

        out = self.lin_out(out_dict[self.target])
        loss = self.contrast_module(CA_hidden_ls, EOA_hidden_ls)

        return out, loss

    def contrast_module(self, CA_hidden_ls, EOA_hidden_ls):
        anchors = []
        pos = []
        neg = []
        for i, z in enumerate(CA_hidden_ls):
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])

        out_anchors = self.attention(anchors, 'CA')
        out_pos = self.attention(pos, 'CA')
        out_neg = self.attention(neg, 'CA')
        loss_ca = self.loss_co(out_anchors, out_pos, out_neg)

        anchors = []
        pos = []
        neg = []
        for i, z in enumerate(EOA_hidden_ls):
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])

        out_anchors = self.attention(anchors, 'EOA')
        out_pos = self.attention(pos, 'EOA')
        out_neg = self.attention(neg, 'EOA')
        loss_eoa = self.loss_co(out_anchors, out_pos, out_neg)
        loss = loss_ca + loss_eoa
        return loss

    def attention(self, input, node_type):
        input = torch.stack(input, dim=0).permute(1, 0, 2)
        q = self.q_lin[node_type](input)
        k = self.k_lin[node_type](input)
        v = self.v_lin[node_type](input)
        att = torch.matmul(k, q.transpose(1, 2))
        att = torch.nn.functional.softmax(att, dim=-1)
        v = att @ v
        v = torch.mean(v, dim=1)
        return v
