import logging
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from gene_graph.data_loaders import Culley
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import (GCN, BatchNorm, GCNConv, Sequential,
                                dense_diff_pool, dense_mincut_pool,
                                global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils import (dense_to_sparse, erdos_renyi_graph,
                                   to_dense_adj, to_dense_batch, unbatch)

log = logging.getLogger(__name__)


class DiffPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        max_num_nodes: int = 5845,
        **kwargs,
    ):
        super().__init__()
        # kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        # TODO remove... linspace leads to many more parameters since clusters decays slower than geometric spacing.
        # num_nodes = np.linspace(
        #     max_num_nodes, 1, self.num_pool_layers + 1, dtype=int
        # ).astype(int)[1:]
        num_nodes = np.flip(
            np.geomspace(1, max_num_nodes, self.num_pool_layers + 1, dtype=int)
        )[1:]

        self.pool = torch.nn.ModuleList()
        self.embed = torch.nn.ModuleList()
        for i, num_node in enumerate(num_nodes):
            self.pool.append(
                GCN(
                    in_channels,
                    self.hidden_channels,
                    self.num_layers,
                    out_channels=num_node,
                )
            )
            self.embed.append(
                GCN(
                    in_channels,
                    self.hidden_channels,
                    self.num_layers,
                    self.out_channels,
                )
            )
            # TODO This necks down network to out_channels after the first step... This could probably be more complicated if needed.
            in_channels = ceil(self.out_channels * self.out_channel_decay)
            self.hidden_channels = ceil(self.out_channels * self.out_channel_decay)
            self.out_channels = ceil(self.out_channels * self.out_channel_decay)

    def forward(self, x, edge_index, batch=None):
        # X in B x N x F makes this forward pass tricky... and also computationally intensive with all of the data transformations
        #TODO pass in device... should be only in main.py
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        edge_indices = []
        ls = []
        es = []
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        batch_size = len(batch.unique())
        # TODO see if max_num_nodes can be removed. In some cases it can be automatically removed. can use num_nodes from init
        # TODO check slack response on reverse to_dense_batch
        max_num_nodes = batch.bincount().max()
        for i in range(self.num_pool_layers):
            if i > 0:
                x, x_mask = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)
                s, _ = to_dense_batch(s, batch, max_num_nodes=max_num_nodes)
                max_num_nodes = None
                x, adj, l, e = dense_diff_pool(x, adj, s, x_mask, normalize=True)
                batch = (
                    torch.arange(0, batch_size, dtype=torch.long)
                    .repeat_interleave(x.size()[1])
                    .to(device)
                )
                x = x.view(-1, x.size()[2])
                edge_index, _ = dense_to_sparse(adj)
                ls.append(l)
                es.append(e)
                edge_indices.append(edge_index)
            s = self.pool[i](x, edge_index)
            x = self.embed[i](x, edge_index)
            adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes)
        x, _ = to_dense_batch(x, batch)
        s, _ = to_dense_batch(s, batch)
        x, adj, l, e = dense_diff_pool(x, adj, s)

        ls.append(l)
        es.append(e)
        # matching global_mean_pool size output
        x = x.squeeze(1)
        # Can pool to some final n clusters, then take mean to reduce params
        return x, ls, es


pools = {
    "mean": global_mean_pool,
    "sum": global_add_pool,
    "max": global_max_pool,
    "diffpool": DiffPool,
}


class Pool(nn.Module):
    def __init__(self, in_channels=None, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.name == "diffpool":
            self.pool = pools[self.name](in_channels, **kwargs)
            self.out_channels = self.out_channels
        else:
            self.pool = pools[self.name]
            self.out_channels = in_channels

    def forward(self, x, edge_index=None, batch=None):
        if self.name in ["sum", "mean", "max"]:
            u = self.pool(x, batch)
        elif self.name == "diffpool":
            u = self.pool(x, edge_index, batch)[0]
        return u


def main():
    data = Culley()[0]
    kwargs = {
        "name": "diffpool",
        "num_pool_layers": 2,
        "hidden_channels": 9,
        "num_layers": 1,
        "out_channels": 5,
        "out_channel_decay": 1.0,
    }
    data.x = data.x[:, 1].unsqueeze(1)
    pool = Pool(in_channels=1, **kwargs)
    u = pool(data.x, data.edge_index, None)


if __name__ == "__main__":
    main()
