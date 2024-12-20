from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import reset, ones
from torch_geometric.typing import EdgeType, Metadata, NodeType, SparseTensor
from torch.nn import Parameter

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


class my_conv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            group: str = "sum",
            **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group

        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(in_channels + out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))


    def reset_parameters(self):
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Union[Dict[EdgeType, Tensor],
            Dict[EdgeType, SparseTensor]]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        D = self.out_channels

        v_dict, out_dict = {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            v_src = v_dict[src_type]
            v_dst = v_dict[dst_type]

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, v=(v_src, v_dst), size=None)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.group)
            out = torch.cat([out, x_dict[node_type]], dim=1)
            out = self.a_lin[node_type](F.tanh(out))
            out_dict[node_type] = out

        return out_dict

    def message(self, v_j: Tensor, index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        out = v_j
        return out.view(-1, self.out_channels)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}')

