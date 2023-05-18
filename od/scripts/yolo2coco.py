import argparse
import os
import warnings
from pathlib import Path
from typing import List, Dict, Union

from mmengine.fileio import dump, list_from_file
from mmengine.utils import scandir, track_iter_progress
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert images to coco format from yolo"
    )
    parser.add_argument("img_path", help="The root path of images")
    parser.add_argument(
        "yolo_annotation_path", help="The root path of yolo annotations"
    )
    parser.add_argument(
        "classes", type=str, help="The text file name of storage class list"
    )
    parser.add_argument(
        "out",
        type=str,
        help="The output annotation json file name, The save dir is in the "
        "same directory as img_path",
    )
    parser.add_argument(
        "-e",
        "--exclude-extensions",
        type=str,
        nargs="+",
        help='The suffix of images to be excluded, such as "png" and "bmp"',
    )
    args = parser.parse_args()
    return args


def collect_image_infos(
    path, exclude_extensions=None
) -> List[Dict[str, Union[str, int]]]:
    print("Collecting image infos...")
    img_infos = []

    images_generator = scandir(path, recursive=True)
    for image_path in track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
            exclude_extensions is not None
            and not image_path.lower().endswith(exclude_extensions)
        ):
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)
            img_info = {
                "fileId": Path(image_path).stem,
                "filename": image_path,
                "width": img_pillow.width,
                "height": img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def collect_yolo_annotations(
    path,
) -> Dict[str, List[Dict[str, Union[str, int]]]]:
    print("Collecting yolo annotations...")
    annotation_infos = dict()
    annotations_generator = scandir(path, recursive=False)
    for annotation_path in track_iter_progress(list(annotations_generator)):
        annotations = []
        with open(Path(path)/annotation_path, "r") as f:
            for line in f.readlines():
                infos = line.strip().split()
                annotations.append(
                    {
                        "classes": int(infos[0]),
                        "n_x": float(infos[1]),
                        "n_y": float(infos[2]),
                        "n_w": float(infos[3]),
                        "n_h": float(infos[4]),
                    }
                )

        annotation_infos[Path(annotation_path).stem] = annotations
    return annotation_infos


def cvt_yolo_to_coco_json(img_infos, annotation_infos, classes):
    image_id = 0
    annotation_id = 0
    coco = dict(images=[], type="instance", categories=[], annotations=[])
    image_set = set()

    for category_id, name in enumerate(classes):
        category_item = dict(
            supercategory=str("none"), id=int(category_id), name=str(name)
        )
        coco["categories"].append(category_item)

    for img_dict in img_infos:
        file_id = img_dict["fileId"]
        file_name = img_dict["filename"]
        img_height = int(img_dict["height"])
        img_width = int(img_dict["width"])
        assert file_name not in image_set

        image_item = dict(
            id=image_id,
            file_name=file_name,
            height=img_height,
            width=img_width,
        )

        if file_id not in annotation_infos:
            warnings.warn(f"Cannot find annotation for image {file_name}")
            continue

        for annotation in annotation_infos[file_id]:
            n_x = float(annotation["n_x"])
            n_y = float(annotation["n_y"])
            n_w = float(annotation["n_w"])
            n_h = float(annotation["n_h"])

            x = float((n_x - n_w / 2) * img_width)
            y = float((n_y - n_h / 2) * img_height)
            w = float(n_w * img_width)
            h = float(n_h * img_height)

            annotation_item = dict(
                id=annotation_id,
                image_id=image_id,
                category_id=int(annotation["classes"]),
                bbox=[x, y, w, h],
                iscrowd=0,
                area=w * h,
            )
            coco["annotations"].append(annotation_item)

            annotation_id += 1

        coco["images"].append(image_item)
        image_set.add(file_name)

        image_id += 1
    return coco


def main():
    args = parse_args()
    assert args.out.endswith(
        "json"
    ), "The output file name must be json suffix"

    # 1 load image list info
    img_infos = collect_image_infos(args.img_path, args.exclude_extensions)

    # 2. load yolo annotation info
    annotation_infos = collect_yolo_annotations(args.yolo_annotation_path)

    # 3 convert to coco format data
    classes = list_from_file(args.classes)
    coco_info = cvt_yolo_to_coco_json(img_infos, annotation_infos, classes)

    # 4 dump
    dump(coco_info, args.out)
    print(f"save json file: {args.out}")


if __name__ == "__main__":
    main()
