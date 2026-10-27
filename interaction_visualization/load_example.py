from pathlib import Path
from PIL import Image
from processing import txt_related
import re, torch
import numpy as np


def read_existing_example(image_id, examples_dir="./examples"):
    """Load an RGB crop, caption, and ordered Boolean masks with filename anchors."""
    folder = Path(examples_dir) / f"{int(image_id):012d}"
    with Image.open(folder / "cropped_image.png") as source:
        image = source.convert("RGB")
    caption = (folder / "caption.txt").read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"Empty caption in {folder}")
    anchors = []
    for path in folder.glob("*_mask_seg.png"):
        match = re.fullmatch(r"(\d+)_(.+)_mask_seg\.png", path.name)
        if match is None:
            raise ValueError(f"Unexpected segmentation filename: {path.name}")
        keyword = match.group(2).replace("_", " ")
        pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, keyword.split())) + r"(?!\w)"
        if not re.search(pattern, caption, re.IGNORECASE):
            raise ValueError(f"Anchor {keyword!r} is absent from {caption!r}")
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L")) > 0
        if mask.shape != (image.height, image.width) or not mask.any():
            raise ValueError(f"Empty or misaligned mask: {path}")
        anchors.append({"index": int(match.group(1)), "keyword": keyword,
                        "filename": path.name, "mask": torch.from_numpy(mask.copy())})
    if not anchors:
        raise ValueError(f"No segmentation masks found in {folder}")
    anchors.sort(key=lambda anchor: anchor["index"])
    return {"image_id": folder.name, "image": image, "caption": caption, "anchors": anchors}

def prepare_example(example, preprocess, tokenizer, resolution, device):
    # The exported crop and masks already share a pixel grid. Reject incompatible
    # resolutions rather than silently resizing only the image during CLIP preprocessing.
    if example["image"].size != (resolution, resolution):
        raise ValueError(f"Expected an exported {resolution}x{resolution} crop and matching masks.")
    img = preprocess(example["image"]).unsqueeze(0).to(device)
    txt = tokenizer.tokenize([example["caption"]]).to(device)
    if tuple(img.shape) != (1, 3, resolution, resolution):
        raise ValueError(f"Unexpected preprocessed image shape: {tuple(img.shape)}")
    anchors = []
    for anchor in example["anchors"]:
        span, decoded = txt_related.get_token_span(example["caption"], anchor["keyword"], tokenizer)
        if span is None or not len(span):
            raise ValueError(f"No token span for {anchor['keyword']!r}")
        # Keep all subword tokens; GEFI text anchors are indices including SOT at 0.
        anchors.append({**anchor, "token_indices": span.cpu().long(), "decoded": decoded})
    return img, txt, anchors
