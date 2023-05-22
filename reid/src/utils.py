from PIL import Image

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
