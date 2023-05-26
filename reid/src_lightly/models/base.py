from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
def knn_predict(
    feature,
    feature_bank,
    feature_labels,
    classes: int,
    knn_k: int = 10,
    knn_t: float = 0.1,
):
    """Helper method to run kNN predictions on features based on a feature bank
    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: Temperature parameter used for kNN
    """
    # compute cos similarity between each feature vector and feature bank
    # [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback

    At the start of every validation epoch we create a feature bank by
    inferencing the model on the dataloader_suspect passed to the module.
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the
    suspect features.
    We can access the highest accuracy during a kNN prediction using the
    max_accuracy attribute.
    """

    def __init__(
        self,
        dataloader_suspect: torch.utils.data.DataLoader,
        suspect_labels: List[int],
        num_classes=10,
        knn_k=10,
        knn_t=0.1,
    ):
        super().__init__()
        self.max_accuracy = 0.0
        self.dataloader_suspect = dataloader_suspect
        self.suspect_map = (
            torch.zeros((num_classes,))
            .scatter_(0, torch.tensor(suspect_labels), 1)
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classes = num_classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        self.backbone = nn.Module()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called to encode features for suspect on validation start"""
        self.backbone.eval()
        self.suspect_features = []
        self.suspect_targets = []
        with torch.no_grad():
            for img, target, _ in self.dataloader_suspect:
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                suspect_feature = self.forward(img).squeeze()
                suspect_feature = F.normalize(suspect_feature, dim=1)
                is_suspect_label = torch.ones(
                    suspect_feature.size(0), device=self.device
                )
                self.suspect_features.append(suspect_feature)
                self.suspect_targets.append(is_suspect_label)
        self.suspect_features = (
            torch.cat(self.suspect_features, dim=0).t().contiguous()
        )
        self.suspect_targets = (
            torch.cat(self.suspect_targets, dim=0).t().contiguous()
        )

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, "suspect_features") and hasattr(
            self, "suspect_targets"
        ):
            views, targets, _ = batch
            # Select only first view
            images = views[0].to(self.device)
            targets = targets.to(self.device)
            feature = self.forward(images).squeeze()
            feature = F.normalize(feature, dim=1)
            is_suspect_label = torch.matmul(
                torch.eye(self.classes, device=self.device)[targets],
                self.suspect_map,
            )
            pred_labels = knn_predict(
                feature,
                self.suspect_features,
                self.suspect_targets,
                2,
                self.knn_k,
                self.knn_t,
            )
            batch_size_num = images.size(0)
            top1 = (pred_labels[:, 0] == is_suspect_label).float().sum().item()
            return (batch_size_num, top1)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.0
            for num, top1 in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            self.log("val/top1_acc", acc * 100.0, prog_bar=True)
