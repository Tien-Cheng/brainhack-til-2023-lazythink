import torch
import numpy as np
from PIL import Image
import os
import glob
import open_clip

# module-level
from src.transforms import make_classification_eval_transform
from src.utils import load_image

# constants and configs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIMILARITY_THRESHOLD = 0.91

# load preprocessing module and dinov2 model
# preprocess = make_classification_eval_transform()
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(DEVICE)
config = dict(
    model_name='ViT-H-14',
    pretrained='laion2b_s32b_b79k',
#     jit=True,
    device=DEVICE
)
model, preprocess = open_clip.create_model_from_pretrained(**config)
model.eval()
print("model loaded")

# creating vectors for all suspect images
suspect_img_path = sorted(glob.glob("data/suspects/*.png"))
print("num image found:", len(suspect_img_path))
with torch.inference_mode(), torch.cuda.amp.autocast():
    suspect_preprocess = torch.stack([preprocess(load_image(x)) for x in suspect_img_path]).to(DEVICE)
    suspect_vectors = model.encode_image(suspect_preprocess)
    # normalize vectors prior
    suspect_vectors /= suspect_vectors.norm(dim=-1, keepdim=True)
    print("suspect vectors shape:", suspect_vectors.shape)

# creating vectors for all cropped images
cropped_img_path = sorted(glob.glob("data/test_detected/*.png"))
for img_fp in cropped_img_path:
    with torch.inference_mode(), torch.cuda.amp.autocast():
        img_preprocess = preprocess(load_image(img_fp)).unsqueeze(0).to(DEVICE)
        img_vector = model.encode_image(img_preprocess)
        # normalize vectors prior
        img_vector /= img_vector.norm(dim=-1, keepdim=True)
        print("img vector shape:", img_vector.shape)

        # compute cosine similarity
        similarity = (img_vector @ suspect_vectors.T).squeeze(0)
        print("similarity shape:", similarity.shape)
        print("similarity:", similarity)

        # find the most similar image
        max_similarity = torch.max(similarity)
        max_idx = torch.argmax(similarity)
        print("max similarity:", max_similarity)
        print("max idx:", max_idx)

        # if similarity is greater than threshold, then save the image
        if max_similarity > SIMILARITY_THRESHOLD:
            img = Image.open(img_fp)
            img.save(f"output/test_sus_crop/{os.path.basename(img_fp).split('.')[0]}_{os.path.basename(suspect_img_path[max_idx]).split('.')[0]}.png")
            print("saved image")
        else:
            print("image not saved")
        print()
