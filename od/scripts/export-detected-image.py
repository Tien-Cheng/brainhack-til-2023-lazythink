import argparse
import json
import numpy as np
from pathlib import Path

import cv2
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export coco detection to individual images"
    )
    parser.add_argument("img_root", help="The root path of images", type=str)
    parser.add_argument(
        "ann_file", help="The empty coco annotation file", type=str
    )
    parser.add_argument(
        "det_json", help="The coco detection json file", type=str
    )
    parser.add_argument(
        "out_dir", help="The output directory of images", type=str
    )
    parser.add_argument(
        "--confidence_threshold",
        help="The confidence threshold",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()
    return args


def parse_json(ann_file):
    has_anno = True
    with open(ann_file) as f:
        data = json.load(f)

    category = [c["name"] for c in data["categories"]]  # 80 classes

    if "annotations" not in data.keys():
        has_anno = False

    if has_anno:
        annotations = data["annotations"]
    images = data["images"]

    category_dict = {c["id"]: c["name"] for c in data["categories"]}
    cat2idx = {c["id"]: i for i, c in enumerate(data["categories"])}
    max_category_id = max(category_dict.keys())

    # id to image mapping
    image_dict = {}
    img_list = list()
    img_id_list = list()

    for image in images:
        key = image["id"]
        image_dict[key] = [image["file_name"], image["width"], image["height"]]
        img_list.append(image["file_name"])
        img_id_list.append(key)

    img2idx = {x: i for i, x in enumerate(img_id_list)}
    category_count = [0 for _ in range(max_category_id + 1)]

    total_annotations = {}

    if has_anno:
        for a in annotations:
            image_name = image_dict[a["image_id"]][0].replace(".jpg", "")
            width = image_dict[a["image_id"]][1]
            height = image_dict[a["image_id"]][2]
            idx = a["category_id"]
            single_ann = []
            single_ann.append(category_dict[idx])
            single_ann.extend(list(map(int, a["bbox"])))
            single_ann.extend([width, height])

            if image_name not in total_annotations:
                total_annotations[image_name] = []

            category_count[idx] += 1
            total_annotations[image_name].append(single_ann)

        print(
            "\n==============[ {} json info ]==============".format(
                "CoCoDataset"
            )
        )
        print("Total Annotations: {}".format(len(annotations)))
        print("Total Image      : {}".format(len(images)))
        print("Annotated Image  : {}".format(len(total_annotations)))
        print("Total Category   : {}".format(len(category)))
        print("----------------------------")
        print("{:^20}| count".format("class"))
        print("----------------------------")
        for idx, cat_idx in enumerate(cat2idx):
            c = category[idx]
            cnt = category_count[cat_idx]
            if cnt != 0:
                print("{:^20}| {}".format(c, cnt))
        print()
    return category, img_list, total_annotations, cat2idx, img2idx


def get_det_results(det_file, img_list, img2idx, cat2idx):
    if det_file != "":
        if det_file.endswith(".pkl"):
            raise NotImplementedError("Could not load pkl yet")

        elif det_file.endswith(".json"):
            with open(det_file) as f:
                json_results = json.load(f)

            if "segmentation" in json_results[0]:
                det_results = [
                    [[np.empty((0, 5)), []] for _ in range(len(img_list))]
                    for _ in range(len(cat2idx))
                ]

                for res in json_results:
                    img_idx = img2idx[res["image_id"]]
                    cat_idx = cat2idx[res["category_id"]]
                    x, y, w, h = res["bbox"]

                    det_results[cat_idx][img_idx][0] = np.concatenate(
                        (
                            det_results[cat_idx][img_idx][0],
                            np.asarray(
                                [x, y, x + w, y + h, res["score"]]
                            ).reshape(1, -1),
                        )
                    )
                    det_results[cat_idx][img_idx][1].append(
                        res["segmentation"]
                    )

            elif "bbox" in json_results[0]:
                det_results = [
                    [np.empty((0, 5)) for _ in range(len(img_list))]
                    for _ in range(len(cat2idx))
                ]

                for res in json_results:
                    img_idx = img2idx[res["image_id"]]
                    cat_idx = cat2idx[res["category_id"]]
                    x, y, w, h = res["bbox"]

                    det_results[cat_idx][img_idx] = np.concatenate(
                        (
                            det_results[cat_idx][img_idx],
                            np.asarray(
                                [x, y, x + w, y + h, res["score"]]
                            ).reshape(1, -1),
                        )
                    )

            det_results = np.asarray(det_results, dtype=object)

        return det_results

    else:
        return None


def main():
    args = parse_args()
    assert args.ann_file.endswith(
        "json"
    ), "The annotation name must be json suffix"
    assert args.det_json.endswith(
        "json"
    ), "The detection name must be json suffix"

    # Convert path string to Path object
    img_root = Path(args.img_root)
    ann_file = Path(args.ann_file)
    det_json = Path(args.det_json)
    out_dir = Path(args.out_dir)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read annotations and parse detection results
    _, img_list, _, cat2idx, img2idx = parse_json(ann_file)

    results = get_det_results(str(det_json), img_list, img2idx, cat2idx)

    # Map each detection boxes to its image name
    img_det = {img_list[i]: results[:, i] for i in range(len(img_list))}

    # Export images
    print("Exporting images...")
    image_infos = []
    for image_name, cat_arr in tqdm.tqdm(img_det.items()):
        for img_arr in cat_arr:
            # Filter out detections with low confidence
            filtered_detections = img_arr[
                img_arr[:, 4] > args.confidence_threshold
            ]

            # Export each detection as an image
            if len(filtered_detections) > 0:
                img = cv2.imread(str(img_root / image_name))
                for idx, detection in enumerate(filtered_detections):
                    xmin, ymin, xmax, ymax = detection[:4].astype(np.int32)
                    confidence = detection[4]
                    cropped_image = img[ymin:ymax, xmin:xmax]

                    image_id = image_name.replace(".png", "")
                    export_image_name = f"{image_id}-{idx}.png"

                    cv2.imwrite(
                        str(out_dir / export_image_name),
                        cropped_image,
                    )

                    image_infos.append(
                        {
                            "Image_ID": image_id,
                            "Image_Name": export_image_name,
                            "confidence": confidence,
                            "ymin": ymin,
                            "xmin": xmin,
                            "ymax": ymax,
                            "xmax": xmax,
                        }
                    )

    # Export image infos as csv
    print("Exporting image infos...")
    with open(out_dir / "image_infos.csv", "w") as f:
        f.write("Image_ID,Image_Name,confidence,ymin,xmin,ymax,xmax\n")
        for image_info in image_infos:
            f.write(
                f"{image_info['Image_ID']},{image_info['Image_Name']},"
                f"{image_info['confidence']},{image_info['ymin']},"
                f"{image_info['xmin']},{image_info['ymax']},"
                f"{image_info['xmax']}\n"
            )

    print("All images exported successfully!")


if __name__ == "__main__":
    main()
