"""
Apply a trained Glenfinglas model to a folder of images.

1. Classifies each image (deer / no_deer) with trained CNN
2. Applies MegaDetector to non-empty images to produce counts
3. Writes one Excel row per image, and a JSON file of bounding boxes

In the case that the CNN labels an image as containing deer but MegaDetector fails to define a 
bounding box, the excel output will be marked with review=True and should be checked manually.

The JSON output can be rendered onto copies of the images containing deer with 
draw_detections.py. Images marked for review are separated into another folder.

Usage:
    python classify.py --model glenfinglas_MODEL.pth --input path/to/images --output results.xlsx --detections detections.json
""" 

import os
import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.transforms import functional

from PytorchWildlife.models import detection

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# Config
# ------
EMPTY_LABEL = "no_deer"         # Set to None to run MegaDetector on every image
MD_VERSION = "MDV6-yolov10-e"
MD_THRESH = 0.2                 # Detection confidence threshold for counting
IMG_EXTS = (".jpg", ".jpeg", ".png")
CROP_FRAC = 0.0456              # Set to 0.0 to disable cropping


# Image preprocessing
# -------------------
class CropBottom:
    """Remove the bottom fraction oif image height"""
    def __init__(self, frac=0.0):
        self.frac = frac
        
    def __call__(self, image):
        if self.frac <= 0:
            return image
            
        w, h = image.size
        keep = int(round(h * (1 - self.frac)))
        return functional.crop(image, top=0, left=0, height=keep, width=w)

class SquarePad:
    """Pad to square, preserving aspect ratio"""
    def __init__(self, fill):
        self.fill = fill

    def __call__(self, image):
        w, h = image.size

        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return ImageOps.expand(image, padding, fill=self.fill)
    
def build_transform(input_size, mean, std):
    fill = tuple(int(m * 255) for m in mean)
    return transforms.Compose([
        CropBottom(CROP_FRAC),
        SquarePad(fill),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# Model
# -----
def build_model(arch, num_classes):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnetv2_s":
        m = models.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch in checkpoint: {arch}")
    return m


# Input handling
# --------------
def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)
    
    
# Progress reporting
# ------------------
def progress_bar(done, total, width=50):
    pct = done / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {pct * 100:5.1f}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


# Main
# ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to trained .pth checkpoint")
    ap.add_argument("--input", required=True, help="folder of images (searched recursively)")
    ap.add_argument("--output", default="results.xlsx", help="output .xlsx path")
    ap.add_argument("--detections", default="detections.json", help="output .json for bounding boxes")
    ap.add_argument("--threshold", default=MD_THRESH, help=f"MegaDetector threshold (default: {MD_THRESH})")
    args = ap.parse_args()
    args.threshold = float(args.threshold)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load classifier
    ckpt = torch.load(args.model, map_location=device)
    arch = ckpt["arch"]
    class_names = ckpt["class_names"]
    input_size = ckpt.get("input_size", 224)
    mean = ckpt.get("norm_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("norm_std", [0.229, 0.224, 0.225])
    
    infer_tf = build_transform(input_size, mean, std)
    model = build_model(arch, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {arch} ({input_size}px: {', '.join(class_names)}) on {device}")

    # load MegaDetector
    detector = detection.MegaDetectorV6(device=str(device), pretrained=True, version=MD_VERSION)

    def detect(path):
        """Run MegaDetector and return a list of detection records."""
        img = np.array(Image.open(path).convert("RGB"))
        result = detector.single_image_detection(img, img_path=path, det_conf_thres=args.threshold)

        dets = result["detections"]
        records = []
        for box, cid, conf in zip(dets.xyxy, dets.class_id, dets.confidence):
            if conf < args.threshold:
                continue
                
            if detector.CLASS_NAMES[int(cid)] != "animal":
                continue

            records.append({
                "bbox": [round(float(v), 2) for v in box],   # [x1, y1, x2, y2]
                "confidence": round(float(conf), 4),
            })
        return records
        
    images = find_images(args.input)
    print(f"Found {len(images)} images")
    if not images:
        return    
        
    since = time.time()
    last_pct = -1

    rows = []
    detections = []
    for i, path in enumerate(images, 1):
        img = Image.open(path).convert("RGB")

        x = infer_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), 1)[0]

        score, idx = torch.max(probs, 0)
        label = class_names[idx.item()]
        score = float(score)

        if EMPTY_LABEL is not None and label == EMPTY_LABEL:
            count = 0
            review = False
        else:
            records = detect(path)
            count = len(records)
            review = (count == 0)
            detections.append({
                "image_path": path,
                "width": img.width,
                "height": img.height,
                "detections": records,
            })

        rows.append({
            "image_path": path,
            "image_name": os.path.basename(path),
            "class": label,
            "confidence": round(float(score), 4),
            "count": count,
            "review": review,
        })

        pct = int(100 * i / len(images))
        if pct != last_pct or i == len(images):
            progress_bar(i, len(images))
            last_pct = pct
            
    elapsed = time.time() - since
    print(f"Processed {len(images)} images in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")        

    pd.DataFrame(rows).to_excel(args.output, index=False)
    print(f"Wrote {len(rows)} rows -> {args.output}")
    
    payload = {
        "input_root": os.path.abspath(args.input),
        "md_version": MD_VERSION,
        "threshold": args.threshold,
        "images": detections,
    }
    with open(args.detections, "w") as f:
        json.dump(payload, f, indent=2)
    n_boxes = sum(len(d["detections"]) for d in detections)
    print(f"Wrote {n_boxes} detections over {len(detections)} images -> {args.detections}")
    
    n_review = sum(1 for r in rows if r["review"])
    if n_review:
        print(f"{n_review} image(s) flagged for review: may contain deer, but a bounding box cannot be defined. Check these manually.")

if __name__ == "__main__":
    main()