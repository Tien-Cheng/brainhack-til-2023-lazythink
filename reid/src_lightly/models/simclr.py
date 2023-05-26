from typing import Literal

import torch
import torchvision
from torch import nn
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import NTXentLoss

from src_lightly.models.base import BenchmarkModule


class SimCLR(BenchmarkModule):
    def __init__(
        self,
        dataloader_suspect,
        suspect_labels,
        num_classes,
        knn_k,
        knn_t,
        backbone: Literal[
            "resnet50", "efficientnet_v2_m", "convnext_base"
        ] = "resnet50",
    ):
        super().__init__(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            knn_k=knn_k,
            knn_t=knn_t,
        )

        if backbone == "resnet50":
            backbone = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            input_dim = 2048
        elif backbone == "efficientnet_v2_m":
            backbone = torchvision.models.efficientnet_v2_m(
                weights="IMAGENET1K_V1"
            )
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            input_dim = 1280
        elif backbone == "convnext_base":
            backbone = torchvision.models.convnext_base(
                weights="IMAGENET1K_V1"
            )
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            input_dim = 2048
        else:
            raise NotImplementedError()

        self.projection_head = SimCLRProjectionHead(input_dim, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
