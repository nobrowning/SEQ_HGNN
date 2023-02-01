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


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = torch.transpose(x, 0, 1)
        return x


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
        group: str = "sum",
        dropout: float = 0.2,
        cross_att: bool = False,
        num_cross_layer: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group
        self.dropout = torch.nn.Dropout(dropout)
        self.cross_att = cross_att
        
        if self.cross_att:
            self.pe = PositionalEncoding(out_channels, dropout=0, max_len=16)
        
        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        self.cross_att_net = torch.nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))
            if self.cross_att:
                cross_att_layer = torch.nn.TransformerEncoderLayer(out_channels, self.heads, \
                    dim_feedforward=out_channels*2, dropout=dropout, batch_first=True, norm_first=True)
                self.cross_att_net[node_type] = torch.nn.TransformerEncoder(cross_att_layer, num_layers=num_cross_layer)

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
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]],
        rel_encoding:  Dict[Tuple ,Tensor]
    ) -> Dict[NodeType, Optional[Tensor]]:

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(x.shape[:-1] + (H, D))
            q_dict[node_type] = self.q_lin[node_type](x).view(x.shape[:-1] + (H, D))
            v_dict[node_type] = self.v_lin[node_type](x).view(x.shape[:-1] + (H, D))
            out_dict[node_type] = []

        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            a_rel = self.a_rel[edge_type]
            k = torch.einsum('... h d , h d r -> ... h r', k_dict[src_type], a_rel)

            m_rel = self.m_rel[edge_type]
            v = torch.einsum('... h d , h d r -> ... h r', v_dict[src_type], m_rel)

            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 rel=self.p_rel[edge_type], size=None)
            out = out + rel_encoding[edge_type].expand_as(out)
            out_dict[dst_type].append(out)

        for node_type, outs in out_dict.items():
            out = group(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue
            out = self.a_lin[node_type](F.gelu(out))
            out = self.dropout(out)

            if self.group != 'cat':
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            else:
                old = x_dict[node_type]
                if len(x_dict[node_type].shape) == 2:
                    old = torch.unsqueeze(old, dim=1)
                out = torch.cat([old, out], dim=1)
            if self.cross_att:
                out = self.pe(out)
                out = self.cross_att_net[node_type](out)
            else:
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
