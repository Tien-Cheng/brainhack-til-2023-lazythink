from typing import Literal, Union, List

import click
import torch
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.smog_transform import SMoGTransform

from src_lightly.data import YoloPlushieDataset
from src_lightly.models.dino import DINO
from src_lightly.models.nnclr import NNCLR
from src_lightly.models.simclr import SimCLR
from src_lightly.models.smog import SMoGModel


def get_model(
    model_type: Literal["dino", "nnclr", "simclr", "smog"],
    backbone: Literal["resnet50", "efficientnet_v2_m", "convnext_base"],
) -> pl.LightningModule:
    if model_type == "dino":
        return DINO(backbone)
    elif model_type == "nnclr":
        return NNCLR(backbone)
    elif model_type == "simclr":
        return SimCLR(backbone)
    elif model_type == "smog":
        return SMoGModel(backbone)
    else:
        raise NotImplementedError()


def get_dataloader(
    root_dir: str,
    img_prefix: str,
    label_prefix: str,
    transform_type: Literal["dino", "nnclr", "simclr", "smog"],
    batch_size: int = 64,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    if transform_type == "dino":
        transform = DINOTransform()
    elif transform_type == "nnclr":
        transform = SimCLRTransform(input_size=64, gaussian_blur=0.5)
    elif transform_type == "simclr":
        transform = SimCLRTransform(input_size=64, gaussian_blur=0.5)
    elif transform_type == "smog":
        transform = SMoGTransform(
            crop_sizes=(64, 64),
            crop_counts=(1, 1),
            gaussian_blur_probs=(0.0, 0.0),
            crop_min_scales=(0.2, 0.2),
            crop_max_scales=(1.0, 1.0),
        )
    else:
        raise NotImplementedError()

    train_yolo_dataset = YoloPlushieDataset(root_dir, img_prefix, label_prefix)

    train_dataset = LightlyDataset.from_torch_dataset(
        train_yolo_dataset, transform=transform
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=MultiViewCollate(),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_dataloader


def get_trainer(
    max_epochs: int, devices: Union[str, int, List[int]]
) -> pl.Trainer:
    trainer = pl.Trainer(max_epochs=max_epochs, devices=devices)
    return trainer


def main(architechture, backbone, batch_size, max_epochs, devices):
    model = get_model(architechture, backbone)
    train_dataloader = get_dataloader(
        root_dir="data",
        img_prefix="images/train",
        label_prefix="labels/yolo/train_labels",
        transform_type=architechture,
        batch_size=batch_size,
        num_workers=2,
    )
    trainer = get_trainer(max_epochs=max_epochs, devices=devices)

    trainer.fit(model, train_dataloader)
