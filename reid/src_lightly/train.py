from typing import Literal, Union, List

import click
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
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
    architechture: Literal["dino", "nnclr", "simclr", "smog"],
    backbone: Literal["resnet50", "efficientnet_v2_m", "convnext_base"],
    dataloader_suspect: torch.utils.data.DataLoader,
    suspect_labels: List[int],
    num_classes: int,
    knn_k: int,
    knn_t: float,
) -> pl.LightningModule:
    if architechture == "dino":
        return DINO(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            backbone=backbone,
            knn_k=knn_k,
            knn_t=knn_t,
        )
    elif architechture == "nnclr":
        return NNCLR(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            backbone=backbone,
            knn_k=knn_k,
            knn_t=knn_t,
        )
    elif architechture == "simclr":
        return SimCLR(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            backbone=backbone,
            knn_k=knn_k,
            knn_t=knn_t,
        )
    elif architechture == "smog":
        return SMoGModel(
            dataloader_suspect=dataloader_suspect,
            suspect_labels=suspect_labels,
            num_classes=num_classes,
            backbone=backbone,
            knn_k=knn_k,
            knn_t=knn_t,
        )
    else:
        raise NotImplementedError()


def get_dataloader(
    root_dir: str,
    train_img_prefix: str,
    train_label_prefix: str,
    val_img_prefix: str,
    val_label_prefix: str,
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

    train_yolo_dataset = YoloPlushieDataset(
        root_dir, train_img_prefix, train_label_prefix
    )

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

    val_yolo_dataset = YoloPlushieDataset(
        root_dir, val_img_prefix, val_label_prefix
    )

    val_dataset = LightlyDataset.from_torch_dataset(
        val_yolo_dataset, transform=transform
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=MultiViewCollate(),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    suspect_yolo_dataset = YoloPlushieDataset(
        root_dir, val_img_prefix, val_label_prefix, suspect=True
    )

    suspect_dataset = LightlyDataset.from_torch_dataset(
        suspect_yolo_dataset, transform=transform
    )

    suspect_dataloader = torch.utils.data.DataLoader(
        suspect_dataset,
        batch_size=batch_size,
        collate_fn=MultiViewCollate(),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader, suspect_dataloader


def get_trainer(
    max_epochs: int, devices: Union[str, int, List[int]], architechture: str
) -> pl.Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{architechture}/",
        every_n_epochs=10,
        save_top_k=-1,  # save all checkpoints every 10 epochs
    )
    wandb_logger = WandbLogger(project="til-23-reid-lightly-runs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )
    return trainer


@click.command()
@click.option(
    "--architechture", type=click.Choice(["dino", "nnclr", "simclr", "smog"])
)
@click.option(
    "--backbone",
    type=click.Choice(["resnet50", "efficientnet_v2_m", "convnext_base"]),
)
@click.option("--batch_size", type=int, default=64)
@click.option("--max_epochs", type=int, default=100)
@click.option("--devices", type=str, default="0")
@click.option("--knn_k", type=int, default=10)
@click.option("--knn_t", type=float, default=0.1)
def main(
    architechture: Literal["dino", "nnclr", "simclr", "smog"],
    backbone: Literal["resnet50", "efficientnet_v2_m", "convnext_base"],
    batch_size: int,
    max_epochs: int,
    devices: str,
    knn_k: int,
    knn_t: float,
):
    train_dataloader, val_dataloader, dataloader_suspect = get_dataloader(
        root_dir="data",
        train_img_prefix="images/train",
        train_label_prefix="labels/yolo/train_labels",
        val_img_prefix="images/validation",
        val_label_prefix="labels/yolo/val_labels",
        transform_type=architechture,
        batch_size=batch_size,
        num_workers=16,
    )

    model = get_model(
        architechture=architechture,
        backbone=backbone,
        dataloader_suspect=dataloader_suspect,
        suspect_labels=[1, 0, 3, 7],  # validation label identified as suspect
        num_classes=10,  # Number of class in validation set
        knn_k=knn_k,
        knn_t=knn_t,
    )
    trainer = get_trainer(
        max_epochs=max_epochs, devices=devices, architechture=architechture
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    """
    Script to train the ReID model using lightly.ai and pytorch-lightning

    Example:
    PYTHONPATH=. python src_lightly/train.py --architechture dino \
      --backbone resnet50 \
      --batch_size 64 \
      --max_epochs 100 \
      --devices 0
    """
    main()
