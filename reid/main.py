import argparse
import glob
from pathlib import Path

import pandas as pd
import torch
import open_clip
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Perform cosine similarity between suspect and cropped"
            "images using open_clip models"
        )
    )
    parser.add_argument(
        "suspect_dir",
        help="The directory of suspect images",
        type=str,
    )
    parser.add_argument(
        "cropped_dir",
        help="The directory of cropped images",
        type=str,
    )
    parser.add_argument(
        "output_csv",
        help="The output csv file",
        type=str,
    )
    parser.add_argument(
        "--similarity_threshold",
        help="The similarity threshold",
        type=float,
        default=0.91,
    )
    args = parser.parse_args()
    return args


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_model(config: dict):
    print("loading model")
    model, preprocess = open_clip.create_model_from_pretrained(**config)
    model.eval()
    return model, preprocess


def encode_image(model, preprocess, img_dir: str):
    img_paths = sorted(glob.glob(img_dir))
    print("num image found:", len(img_paths))
    with torch.inference_mode(), torch.cuda.amp.autocast():
        preprocesed_img = torch.stack(
            [preprocess(load_image(x)) for x in img_paths]
        ).to(DEVICE)
        vectors = model.encode_image(preprocesed_img)
        # normalize vectors prior
        vectors /= vectors.norm(dim=-1, keepdim=True)
    return vectors


def main():
    args = parse_args()
    assert Path(args.suspect_dir).is_dir()
    assert Path(args.cropped_dir).is_dir()

    suspect_dir = Path(args.suspect_dir)
    cropped_dir = Path(args.cropped_dir)
    output_csv = Path(args.output_csv)

    # load model
    model, preprocess = load_model(
        dict(
            model_name="ViT-H-14",
            pretrained="laion2b_s32b_b79k",
            #     jit=True,
            device=DEVICE,
        )
    )

    # encode suspect images
    suspect_vectors = encode_image(
        model, preprocess, str(suspect_dir / "*.png")
    )

    # encode cropped images
    cropped_vectors = encode_image(
        model, preprocess, str(cropped_dir / "*.png")
    )

    # compute cosine similarity
    # shape=(num_cropped, num_suspect)
    similarity_matrix = (cropped_vectors @ suspect_vectors.T).squeeze(0)

    # find the most similar image
    max_similarity = torch.max(similarity_matrix, dim=1)

    # check if similarity is greater than threshold
    is_suspect = torch.where(max_similarity > args.similarity_threshold)

    # save to csv
    pd.DataFrame(
        dict(
            suspect_file_name=list(suspect_dir.glob("*.png")),
            is_suspect=is_suspect.tolist(),
        )
    ).to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
