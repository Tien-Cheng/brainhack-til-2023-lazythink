import copy
from typing import Literal

import torch
import torchvision
from torch import nn
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.utils.scheduler import cosine_schedule

from src_lightly.models.base import BenchmarkModule


class DINO(BenchmarkModule):
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
            input_dim = 1000
        elif backbone == "efficientnet_v2_m":
            backbone = torchvision.models.efficientnet_v2_m(
                weights="IMAGENET1K_V1"
            )
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            input_dim = 1000
        elif backbone == "convnext_base":
            backbone = torchvision.models.convnext_base(
                weights="IMAGENET1K_V1"
            )
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            input_dim = 1000
        else:
            raise NotImplementedError()

        # input_dim = 512
        # instead of a resnet you can also use a vision transformer backbone
        # as in the
        # original paper (you might have to reduce the batch size in this case)
        # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16'
        # , pretrained=False)
        # input_dim = self.backbone.embed_dim

        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(
            output_dim=2048, warmup_teacher_temp_epochs=5
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(
            self.student_backbone, self.teacher_backbone, m=momentum
        )
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(
            teacher_out, student_out, epoch=self.current_epoch
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(
            teacher_out, student_out, epoch=self.current_epoch
        )
        self.log("val/loss", loss)

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(
            current_epoch=self.current_epoch
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
