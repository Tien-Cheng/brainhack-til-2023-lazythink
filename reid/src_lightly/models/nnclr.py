import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import (
    NNCLRPredictionHead,
    NNCLRProjectionHead,
    NNMemoryBankModule,
)

from src_lightly.models.base import BenchmarkModule


class NNCLR(BenchmarkModule):
    def __init__(
        self,
        dataloader_suspect,
        suspect_labels,
        num_classes,
        knn_k,
        knn_t,
    ):
        super().__init__(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            knn_k=knn_k,
            knn_t=knn_t,
        )
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)
        self.memory_bank = NNMemoryBankModule(size=4096)

        self.criterion = NTXentLoss()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=True)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        z0 = self.memory_bank(z0, update=False)
        z1 = self.memory_bank(z1, update=False)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
