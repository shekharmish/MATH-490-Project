import logging

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

log = logging.getLogger(__name__)
# For one epoch
class Metric:
    def __init__(self, name: str = "default", **kwargs):
        self.name = name
        self.y_true_epoch = np.empty(0)
        self.y_pred_epoch = np.empty(0)
        self.loss_all = np.empty(0)
        self.loss_autoencoder_all = np.empty(0)

    def record(self, loss, y_true, y_pred):
        self.y_true_epoch = np.concatenate(
            (self.y_true_epoch, y_true.cpu().numpy().reshape(-1))
        )
        self.y_pred_epoch = np.concatenate(
            (self.y_pred_epoch, y_pred.detach().cpu().numpy().reshape(-1))
        )
        self.loss_all = np.concatenate(
            (self.loss_all, loss.detach().cpu().numpy().reshape(-1))
        )


    def summarize(self, **kwargs):
        loss = np.mean(self.loss_all)
        try:
            accuracy = accuracy_score(self.y_true_epoch, self.y_pred_epoch)
        except ValueError as e:
            log.error(f"Error in computing Accuracy: {e}")
            accuracy = np.nan
        try:
            f1 = f1_score(self.y_true_epoch, self.y_pred_epoch, zero_division=1)
        except ValueError as e:
            log.error(f"Error in computing f1_score: {e}")
            f1 = np.nan
        try:
            precision = precision_score(self.y_true_epoch, self.y_pred_epoch, zero_division=1)
        except ValueError as e:
            log.error(f"Error in computing precision: {e}")
            precision = np.nan
        try:
            recall = recall_score(self.y_true_epoch, self.y_pred_epoch, zero_division=1)
        except ValueError as e:
            log.error(f"Error in computing recall: {e}")
            recall = np.nan

        metrics_temp = {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        # prepend self.type to all keys of metrics
        metrics = {}
        for k, v in metrics_temp.items():
            metrics[f"{self.name}/{k}"] = v
        return metrics

def main():
    pass

if __name__ == "__main__":
    main()
