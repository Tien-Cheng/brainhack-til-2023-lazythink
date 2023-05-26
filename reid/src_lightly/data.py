import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class YoloPlushieDataset(Dataset):
    """Plushie dataset in yolo format."""

    def __init__(
        self,
        root_dir,
        img_prefix,
        label_prefix,
        transform=None,
        target_transform=None,
        suspect=False,
        suspect_path="filtered_val_data.json",
    ):
        """
        Arguments:
            root_dir {str} -- Root directory of dataset where directory
                ``img_prefix`` and ``label_prefix`` exists.
            img_prefix {str} -- Directory of images.
            label_prefix {str} -- Directory of labels.
            transform {callable, optional} -- A function/transform that takes
                in an PIL image and returns a transformed version.
            target_transform {callable, optional} -- A function/transform that
                takes in the target and transforms it.
            suspect {bool, optional} -- Whether to include suspect images.
            suspect_path {str, optional} -- Path to suspect images json file
                that contains list of suspect images file name.
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / img_prefix
        self.label_dir = self.root_dir / label_prefix
        self.transform = transform
        self.target_transform = target_transform
        self.plushie_labels = []
        self.suspect = suspect
        with open(suspect_path, 'r') as file:
            self.suspect_images = json.load(file)

        for label_path in self.label_dir.glob("*.txt"):
            if label_path.name in self.suspect_images:
                # Skip suspect images if not in suspect mode
                if not suspect:
                    continue

            with open(label_path) as f:
                for idx, label_infos in enumerate(f.readlines()):
                    [c, n_x, n_y, n_w, n_h] = label_infos.split(" ")
                    self.plushie_labels.append(
                        {
                            "index": "{}_{}".format(label_path.stem, idx),
                            "class": int(c),
                            "bbox": [
                                float(n_x),
                                float(n_y),
                                float(n_w),
                                float(n_h),
                            ],
                            "path": self.img_dir
                            / (str(label_path.stem) + ".png"),
                        }
                    )

    def __len__(self):
        return len(self.plushie_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        plushie_info = self.plushie_labels[idx]
        img = Image.open(plushie_info["path"]).convert("RGB")
        target = plushie_info["class"]

        # Convert yolo bbox to PIL bbox
        (x, y, w, h) = plushie_info["bbox"]
        left = (x - w / 2) * img.width
        top = (y - h / 2) * img.height
        right = (x + w / 2) * img.width
        bottom = (y + h / 2) * img.height

        # Crop plushie
        img = img.crop((left, top, right, bottom))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
