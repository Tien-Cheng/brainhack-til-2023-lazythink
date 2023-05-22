import argparse
from pathlib import Path

import pandas as pd
import torch
import open_clip
import tqdm
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
        "input_csv",
        help="The input csv file that contains key of Image_Name",
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
    print("model loaded")
    return model, preprocess


def encode_image(model, preprocess, img_paths: list[str]):
    print("num image found:", len(img_paths))
    with torch.inference_mode(), torch.cuda.amp.autocast():
        preprocesed_img = torch.stack(
            [preprocess(load_image(x)) for x in img_paths]
        )
        vectors = []
        for img in tqdm.tqdm(preprocesed_img):
            vector = model.encode_image(img.unsqueeze(0).to(DEVICE))
            vectors.append(vector.reshape(-1).to("cpu"))
        vectors = torch.stack(vectors)
        # normalize vectors prior
        vectors /= vectors.norm(dim=-1, keepdim=True)
    return vectors


def main():
    args = parse_args()
    assert Path(args.suspect_dir).is_dir()
    assert Path(args.cropped_dir).is_dir()
    assert Path(args.input_csv).is_file()

    suspect_dir = Path(args.suspect_dir)
    cropped_dir = Path(args.cropped_dir)
    output_csv = Path(args.output_csv)

    # load csv
    df = pd.read_csv(args.input_csv)
    df["Image_Path"] = df["Image_Name"].apply(lambda x: str(cropped_dir / x))

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
    print("Encoding suspect images")
    suspect_vectors = encode_image(
        model, preprocess, list(suspect_dir.glob("*.png"))
    )

    # encode cropped images
    print("Encoding cropped images")
    cropped_vectors = encode_image(
        model, preprocess, df["Image_Path"].values.tolist()
    )

    # compute cosine similarity
    # shape=(num_cropped, num_suspect)
    similarity_matrix = cropped_vectors.float() @ suspect_vectors.T.float()

    # find the most similar image
    max_similarity = torch.max(similarity_matrix, dim=1).values

    # check if similarity is greater than threshold
    is_suspect = torch.where(max_similarity > args.similarity_threshold, 1, 0)

    # save to csv
    df["class"] = is_suspect.tolist()
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
