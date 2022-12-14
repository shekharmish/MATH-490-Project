import logging

import hydra
import pandas as pd
import torch
from gene_graph.mp import MPs
from gene_graph.pool import Pool, pools
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as f
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, GCN, MLP, InnerProductDecoder

import wandb

log = logging.getLogger(__name__)

class MpIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        ## pre_mlp
        if kwargs["pre_mlp"].get("num_layers") > 0:
            self.pre_mlp = MLP(in_channels, **kwargs["pre_mlp"])
        else:
            self.pre_mlp = nn.Identity()
        ## mp - There is always a MP layer
        # pre_mlp exists
        if (
            kwargs["pre_mlp"].get("num_layers") > 0
            and kwargs["mp"].get("num_layers") > 0
        ):
            self.mp = MPs[kwargs["mp"].get("name")](
                in_channels=kwargs["pre_mlp"].get("out_channels"), **kwargs["mp"]
            )
        # pre_mlp DNE
        elif kwargs["mp"].get("num_layers") > 0:
            self.mp = MPs[kwargs["mp"].get("name")](
                in_channels=in_channels, **kwargs["mp"]
            )
        else:
            self.mp = MpIdentity()
        ## post_mlp
        # mp exists
        if (
            kwargs["mp"].get("num_layers") > 0
            and kwargs["post_mlp"].get("num_layers") > 0
        ):
            self.post_mlp = MLP(kwargs["mp"].get("out_channels"), **kwargs["post_mlp"])
        # mp DNE, pre_mlp exits
        elif (
            kwargs["pre_mlp"].get("num_layers") > 0
            and kwargs["mp"].get("num_layers") == 0
            and kwargs["post_mlp"].get("num_layers") > 0
        ):
            self.post_mlp = MLP(
                kwargs["pre_mlp"].get("out_channels"), **kwargs["post_mlp"]
            )
        # mp DNE, pre_mlp DNE
        elif (
            kwargs["pre_mlp"].get("num_layers") == 0
            and kwargs["mp"].get("num_layers") == 0
            and kwargs["post_mlp"].get("num_layers") > 0
        ):
            self.post_mlp = MLP(in_channels, **kwargs["post_mlp"])
        else:
            self.post_mlp = nn.Identity()

    def forward(self, x, edge_index):
        x = self.pre_mlp(x)
        x = self.mp(x, edge_index)
        x = self.post_mlp(x)
        return x

class Model(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.autoencoder = kwargs["autoencoder"].get("name")
        if self.autoencoder == "GAE":
            decoder = InnerProductDecoder()
            self.encoder = GAE(Encoder(in_channels, **kwargs), decoder)
            self.encoder.__setattr__(name="__name__", value="autoencoder")
        else:
            #TODO can probably generalize with an identity decoder... but then would have to change language to 'encode', 'decode' instead of forward maybe not worth it...
            self.encoder = Encoder(in_channels, **kwargs)
            self.encoder.__setattr__(name="__name__", value="encoder")
        ## Pool
        # post_mlp exits
        if kwargs["post_mlp"].get("num_layers") > 0:
            self.pool = Pool(
                in_channels=kwargs["post_mlp"].get("out_channels"), **kwargs["pool"]
            )
        # post_mlp DNE, mp exits
        elif (
            kwargs["post_mlp"].get("num_layers") == 0
            and kwargs["mp"].get("num_layers") > 0
        ):
            self.pool = Pool(
                in_channels=kwargs["mp"].get("out_channels"), **kwargs["pool"]
            )
        # post_mlp DNE, mp DNE, pre_mlp exits
        elif (
            kwargs["post_mlp"].get("num_layers") == 0
            and kwargs["mp"].get("num_layers") == 0
            and kwargs["pre_mlp"].get("num_layers") > 0
        ):
            self.pool = Pool(
                in_channels=kwargs["pre_mlp"].get("out_channels"), out_channels=self.out_channels, **kwargs["pool"]
            )
        # post_mlp DNE, mp DNE, pre_mlp DNE
        elif (
            kwargs["post_mlp"].get("num_layers") == 0
            and kwargs["mp"].get("num_layers") == 0
            and kwargs["pre_mlp"].get("num_layers") == 0
        ):
            self.pool = Pool(in_channels, **kwargs["pool"])
        ## global_mlp
        if kwargs["global_mlp"].get("num_layers") > 0:
            self.global_mlp = MLP(
                in_channels=self.pool.out_channels,
                out_channels=num_classes,
                **kwargs["global_mlp"],
            )
        else:
             self.global_mlp = nn.Identity()


    def forward(self, x, edge_index, batch=None, **kwargs):
        # Unpack kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Encoder
        if self.encoder.__name__ == "autoencoder":
            x = self.encoder.encode(x, edge_index)
        elif self.encoder.__name__ == "encoder":
            x = self.encoder.forward(x, edge_index)
        # Pool
        u = self.pool(x, edge_index, batch)
        # Top
        u_top = self.global_mlp(u)
        # probability readout
        u_top = f.softmax(u_top, dim=1)
        return x, u, u_top

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Load config
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(mode=cfg.wandb.mode, project=cfg.project, config=wandb_cfg)
    # data setup
    dataset = TUDataset(root="./tmp/PROTEINS", name="PROTEINS", use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model(in_channels=dataset.num_features, **wandb.config.model)
    num_params = count_parameters(model)
    num_params
    for i, batch in enumerate(loader):
        x, u, u_top = model(batch.x, batch.edge_index, batch.batch)
        x, u, u_top
        break


if __name__ == "__main__":
    main()
