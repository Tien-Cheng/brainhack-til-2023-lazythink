from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# code for KNN prediction
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L143
def dino_knn(features, feature_bank, feature_labels, num_classes, knn_k, knn_t):
    batch_size = features.size(0)
    retrieval_one_hot = torch.zeros(knn_k, num_classes, device=features.device)

    # calculate the dot product and compute top-k neighbors
    similarity = torch.mm(features, feature_bank)
    distances, indices = similarity.topk(knn_k, largest=True, sorted=True)
    candidates = feature_labels.view(1, -1).expand(batch_size, -1)
    retrieved_neighbors = torch.gather(candidates, 1, indices)

    retrieval_one_hot.resize_(batch_size * knn_k, num_classes).zero_()
    retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
    distances_transform = distances.clone().div_(knn_t).exp_()
    probs = torch.sum(
        torch.mul(
            retrieval_one_hot.view(batch_size, -1, num_classes),
            distances_transform.view(batch_size, -1, 1),
        ),
        1,
    )
    _, predictions = probs.sort(1, True)

    return predictions

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
    print("sim_weight", sim_weight)

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    print("one_hot_label", one_hot_label)

    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    print("pred_scores", pred_scores)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    print("pred_labels", pred_labels)
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
        self.total_num = 0
        self.total_top1 = 0.0

    def on_validation_epoch_start(self, **kwargs):
        """Called to encode features for suspect on validation start"""
        suspect_features = []
        suspect_targets = []
        with torch.no_grad():
            for views, _, _ in self.dataloader_suspect:
                views = [view.to(self.device) for view in views]
                # Pick first view as img
                img = views[0]
                suspect_feature = self.forward(img).squeeze()
                suspect_feature = F.normalize(suspect_feature, dim=1)
                # Set all images as suspect
                is_suspect_label = torch.ones(
                    suspect_feature.size(0), dtype=int, device=self.device
                )
                suspect_features.append(suspect_feature)
                suspect_targets.append(is_suspect_label)
        self.suspect_features = (
            torch.cat(suspect_features, dim=0).t().contiguous()
        )
        self.suspect_targets = (
            torch.cat(suspect_targets, dim=0).t().contiguous()
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
            print("is_suspect_label", is_suspect_label)
            pred_labels = dino_knn(
                feature,
                self.suspect_features,
                self.suspect_targets,
                2,
                self.knn_k,
                self.knn_t,
            )
            print("pred_labels", pred_labels)
            batch_size = images.size(0)
            correct = pred_labels.eq(is_suspect_label.view(-1, 1))
            top1 = correct.narrow(1, 0, 1).sum().item()
            self.total_num += batch_size
            self.total_top1 += top1

    def on_validation_epoch_end(self, **kwargs):
        print(self.total_top1)
        print(self.total_num)
        acc = float(self.total_top1 / self.total_num)
        if acc > self.max_accuracy:
            self.max_accuracy = acc
        self.log("val/top1_acc", acc * 100.0, prog_bar=True)
        self.total_num = 0
        self.total_top1 = 0.0
