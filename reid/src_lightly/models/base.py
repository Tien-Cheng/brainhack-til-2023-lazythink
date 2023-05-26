from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# code for KNN prediction
# https://github.com/facebookresearch/dino/blob/main/eval_knn.py#L143
def dino_knn(
    features, feature_bank, feature_labels, num_classes, knn_k, knn_t
):
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
            pred_labels = dino_knn(
                feature,
                self.suspect_features,
                self.suspect_targets,
                2,
                self.knn_k,
                self.knn_t,
            )
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
