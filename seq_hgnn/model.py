import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from einops import rearrange


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None or aggr == 'cat':
        if len(xs[0].shape) == 2:
            return torch.stack(xs, dim=1)
        else:
            return torch.cat(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class SeqHGNNConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)
        
        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.k_lin)
        # reset(self.q_lin)
        # reset(self.v_lin)
        # reset(self.a_lin)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]],  # Support both.
        rel_encoding:  Dict[Tuple ,Tensor]
    ) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(x.shape[:-1] + (H, D))
            q_dict[node_type] = self.q_lin[node_type](x).view(x.shape[:-1] + (H, D))
            v_dict[node_type] = self.v_lin[node_type](x).view(x.shape[:-1] + (H, D))
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            a_rel = self.a_rel[edge_type]
            k = torch.einsum('... h d , h d r -> ... h r', k_dict[src_type], a_rel)

            m_rel = self.m_rel[edge_type]
            v = torch.einsum('... h d , h d r -> ... h r', v_dict[src_type], m_rel)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 rel=self.p_rel[edge_type], size=None)
            out = out + rel_encoding[edge_type].expand_as(out)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, 'cat')

            if out is None:
                out_dict[node_type] = None
                continue
            out = self.a_lin[node_type](F.gelu(out))
            out = self.dropout(out)
            
            old = x_dict[node_type]
            if len(x_dict[node_type].shape) == 2:
                old = torch.unsqueeze(old, dim=1)
            out = torch.cat([old, out], dim=1)
            out = self.dropout(out)
            out_dict[node_type] = out
        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        if len(q_i.shape) == 3:
            alpha = (q_i * k_j).sum(dim=-1) * rel
            alpha = alpha / math.sqrt(q_i.size(-1))
            alpha = softmax(alpha, index, ptr, size_i)
            out = v_j * alpha.view(-1, self.heads, 1)
        else:
            alpha = torch.einsum('b i h d , b j h d -> b h i j', q_i,k_j)
            alpha = torch.einsum('b h i j , h -> b h i j', alpha, rel)
            alpha = alpha / math.sqrt(q_i.size(-1))
            alpha = softmax(alpha, index, ptr, size_i)
            out = torch.einsum('b j h d, b h i j -> b i h d', v_j, alpha)
        return rearrange(out, '... h d -> ... (h d)')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')


class SeqHGNN(torch.nn.Module):
    def __init__(self, graph_meta, targe_node_type, hidden_channels, out_channels, num_heads, num_layers, agg=['sum'], dropout=0.5):
        super().__init__()
        self.targe_node_type = targe_node_type
        self.graph_meta = graph_meta
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.graph_meta[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        self.rel_encoding = torch.nn.ParameterDict()
        for relation_type in self.graph_meta[1]:
            self.rel_encoding['__'.join(relation_type)] = None
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SeqHGNNConv(hidden_channels, hidden_channels, self.graph_meta,
                           num_heads, dropout=dropout)
            self.convs.append(conv)
        self.dropout = dropout
        self.flatten = True
        # self.lin = Linear(-1, out_channels)
        self.lin = None

        # self.flatten = False
        # self.final_encoder = torch.nn.TransformerDecoderLayer(d_model=self.hidden_channels, nhead=8)
        # self.lin = torch.nn.Linear(hidden_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.lin_dict)
        gain = torch.nn.init.calculate_gain('relu')
        for rel in self.rel_encoding.keys():
            v = torch.FloatTensor(self.num_heads, self.hidden_channels // self.num_heads)
            torch.nn.init.xavier_uniform_(v, gain=gain)
            self.rel_encoding[rel] = torch.nn.Parameter(v.reshape(-1))
    
    def dropout_channel(self, feats):
        if self.training:
            num_samples = int(feats.shape[1] * self.dropout)# self.dropout
            selected_idx = torch.randperm(feats.shape[1], dtype=torch.int64, device=feats.device)[:num_samples]
            feats[:,selected_idx,:] = 0
        return feats

    
    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, self.rel_encoding)
        
        out = x_dict[self.targe_node_type]
        if self.flatten:
            out = self.dropout_channel(out)
            out = out.reshape(out.shape[0], -1)

        if not self.lin:
            device = out.device
            num_dim = out.shape[-1]
            self.lin = torch.nn.Linear(num_dim, self.out_channels, device=device)
            # reset(self.lin)

        return self.lin(out)

    
class SeqHGNN_LP(torch.nn.Module):
    def __init__(self, graph_meta, targe_node_type, hidden_channels, out_channels, num_heads, num_layers, agg=['sum'], dropout=0.5):
        super().__init__()
        self.targe_node_type = targe_node_type
        self.graph_meta = graph_meta
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.lin_dict = torch.nn.ModuleDict()
        
        self.rel_encoding = torch.nn.ParameterDict()
        for relation_type in self.graph_meta[1]:
            self.rel_encoding['__'.join(relation_type)] = None


        if len(agg) < num_layers:
            for _ in range(num_layers - len(agg)):
                agg.append(agg[-1])
        elif len(agg) > num_layers:
            agg = agg[:num_layers]
        self.agg = agg
        
        self.convs = torch.nn.ModuleList()
        for _, l_agg in zip(range(num_layers), self.agg):
            conv = SeqHGNNConv(hidden_channels, hidden_channels, self.graph_meta,
                           num_heads, dropout=dropout) # num_cross_layer=1
            self.convs.append(conv)
        self.dropout = dropout
        self.out_channels = out_channels

        self.flatten = True
        self.lin = None
        self.label_embedding = torch.nn.Embedding(self.out_channels, self.hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain('relu')
        for rel in self.rel_encoding.keys():
            v = torch.FloatTensor(self.num_heads, self.hidden_channels // self.num_heads)
            torch.nn.init.xavier_uniform_(v, gain=gain)
            self.rel_encoding[rel] = torch.nn.Parameter(v.reshape(-1))

    def dropout_channel(self, feats):
        if self.training:
            num_samples = int(feats.shape[1] * self.dropout)# self.dropout
            selected_idx = torch.randperm(feats.shape[1], dtype=torch.int64).to(feats.device)[:num_samples]
            feats[:,selected_idx,:] = 0
        return feats

    def forward(self, x_dict, edge_index_dict, batch_y, batch_train_mask, batch_size):
        
        y_embs = self.label_embedding(batch_y)
        no_train_idxs = (batch_train_mask==0).nonzero(as_tuple=True)[0]
        y_embs[no_train_idxs] = 0
        y_embs[:batch_size] = 0

        x_input = {}
        device = x_dict[self.targe_node_type].device
        for t, x_t in x_dict.items():
            if len(x_t.shape) == 2:
                x_t = x_t.unsqueeze(1)
            x_t_inp = []
            
            if t == self.targe_node_type:
                y_embs_exp = y_embs.expand(x_t.shape[1], -1, -1).transpose(0, 1)
                x_t = torch.cat([x_t, y_embs_exp], dim=-1)

            for idx, x_t_i in enumerate(torch.split(x_t, 1, dim=1)):
                key = '{}_{}'.format(t, idx)
                if key not in self.lin_dict.keys():
                    self.lin_dict[key] = Linear(-1, self.hidden_channels).to(device)
                x_t_inp.append(self.lin_dict[key](x_t_i))
            x_t_inp = torch.cat(x_t_inp, dim=1)
            x_input[t] = x_t_inp
        
        
            
        for conv in self.convs:
            x_input = conv(x_input, edge_index_dict, self.rel_encoding)
        
        out = x_input[self.targe_node_type]
        
        if self.flatten:
            out = self.dropout_channel(out)
            out = out.reshape(out.shape[0], -1)

        if not self.lin:
            device = out.device
            num_dim = out.shape[-1]
            self.lin = torch.nn.Linear(num_dim, self.out_channels).to(device)
        return self.lin(out)
