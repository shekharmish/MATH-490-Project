###########################--Module Import--##################################
import logging

import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (add_self_loops, coalesce, mask_to_index,
                                   to_undirected)
from tqdm import tqdm

import wandb
from pyg_gym.metrics import Metric
from pyg_gym.models import Model, count_parameters

###
log = logging.getLogger(__name__)


class Runner:
    def __init__(self, device=None, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        # device
        self.device = device
        # kwargs
        self.optimizer = None
        self.data = None
        self.optimizer = None
        self.df_train_embed = None
        self.df_val_embed = None
        self.trainable_parameters = {}

    def load_data(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        dataset = PygGraphPropPredDataset("ogbg-ppa")
        self.dataset = PygGraphPropPredDataset("ogbg-ppa")
        # self.num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        if self.dev:
           split_idx = {"train": torch.linspace(0,80,81,dtype=torch.int64),"valid": torch.linspace(80,90,11,dtype=torch.int64), "test": torch.linspace(90,100,11,dtype=torch.int64) }
        self.train_loader = DataLoader(dataset[split_idx["train"]], batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset[split_idx["test"]], batch_size=self.batch_size, shuffle=False)
        self.num_features = 1

    def add_model(self, **kwargs):
        model = Model(in_channels=self.num_features, num_classes=self.num_classes, **kwargs).to(self.device)
        self.model = model
        # count trainable params
        model_trainable_parameters = {
            "model/trainable_parameters-total": count_parameters(model)
        }
        module_trainable_parameters = {
            f"model/trainable_parameters-{str(k).split('(')[0]}": count_parameters(v)
            for k, v in model._modules.items()
        }
        self.trainable_parameters = {
            **model_trainable_parameters,
            **module_trainable_parameters,
        }
        log.info("model added to runner")
        return model

    def set_optimizer(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.optimizer = optimizer

    def set_criterion(self, **kwargs):
        criterion = torch.nn.NLLLoss()
        self.criterion = criterion

    def train_single(self, batch):
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad()
        x = torch.ones(batch.num_nodes, 1).to(self.device)
        x, u, u_top = self.model(x, batch.edge_index, batch.batch)
        l = self.criterion(u_top, batch.y.view(-1))
        l.backward()
        self.optimizer.step()
        return l, x, u, u_top


    def test_single(self, batch):
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            x = torch.ones(batch.num_nodes, 1).to(self.device)
            x, u, u_top = self.model(x, batch.edge_index, batch.batch)
            l = self.criterion(u_top, batch.y.view(-1))
        return l, x, u, u_top

    def train(self):
        metric = Metric("train")
        for batch in tqdm(self.train_loader):
            l, x, u, u_top = self.train_single(batch)
            metric.record(l, batch.y, u_top.argmax(dim=1))

        metrics = metric.summarize()
        return metrics

    def valid(self):
        self.model.eval()
        metric = Metric("valid")
        for batch in tqdm(self.valid_loader):
            l, x, u, u_top = self.test_single(batch)
            metric.record(l, batch.y, u_top.argmax(dim=1))

        metrics = metric.summarize()
        return metrics

    def test(self):
        self.model.eval()
        metric = Metric("test")
        # sets size, but removed at end
        u_epoch = np.empty((0,self.num_classes))
        for batch in tqdm(self.test_loader):
            l, x, u, u_top = self.test_single(batch)
            metric.record(l, batch.y, u_top.argmax(dim=1))
            u_epoch = np.concatenate([u_epoch, u.detach().cpu().numpy()])
        df_embed = self.embedding_projection(u_epoch, metric.y_true_epoch, metric.y_pred_epoch)
        metrics = metric.summarize()
        return metrics, df_embed

    def embedding_projection(self, z, y_true, y_pred):
            # DataFrame for generating projector embedding
            df_embed = pd.DataFrame(z)
            df_embed.columns = df_embed.columns.astype(str)
            df_embed['y_pred'] = y_pred
            df_embed['y_true'] = y_true
            return df_embed

    def run(self, **kwargs):
        valid_metrics = {}
        # kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        for epoch in tqdm(range(1, self.max_epoch + 1)):
            train_metrics = self.train()
            train_metrics["epoch"] = epoch
            wandb.log(train_metrics, commit=False)
            if (epoch % self.valid_epoch == 0) and (epoch >= self.start_valid_epoch):
                valid_metrics = self.valid()
                valid_metrics["epoch"] = epoch
                wandb.log(valid_metrics, commit=False)
            wandb.log({}, commit=True)
        test_metrics, df_test_embed = self.test()
        wandb.log(test_metrics)
        wandb.log({"test/embed": df_test_embed})

        return train_metrics, valid_metrics, test_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")
    runner = Runner(device)

if __name__ == "__main__":
    main()
