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

    train_yolo_dataset = YoloPlushieDataset(root_dir, train_img_prefix, train_label_prefix)

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

    val_yolo_dataset = YoloPlushieDataset(root_dir, val_img_prefix, val_label_prefix)

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

    suspect_yolo_dataset = YoloPlushieDataset(root_dir, val_img_prefix, val_label_prefix, suspect=True)

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
        dirpath=f"checkpoints/{architechture}/", every_n_epochs=10
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
def main(architechture, backbone, batch_size, max_epochs, devices):
    model = get_model(architechture, backbone)
    train_dataloader, val_dataloader, suspect_dataloader = get_dataloader(
        root_dir="data",
        img_prefix="images/train",
        label_prefix="labels/yolo/train_labels",
        transform_type=architechture,
        batch_size=batch_size,
        num_workers=12,
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
