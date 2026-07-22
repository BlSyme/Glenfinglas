"""
Apply a trained Glenfinglas model to a labelled folder and report accuracy.

The input folder holds one subfolder per class. Each subfolder's name is the
ground-truth label and must match one of the model's class_names. A prediction
is correct when the predicted class name equals the subfolder name.

    test_set/
        no_deer/
        deer/

Reports total accuracy and accuracy per class.

Usage:
    python evaluate.py --model glenfinglas_MODEL.pth --input path/to/test_images
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.transforms import functional

from speciesnet_backbone import build_speciesnet, SpeciesNetPreprocess, read_info


# Config
# ------
IMG_EXTS = (".jpg", ".jpeg", ".png")
CROP_FRAC = 0.0456                      # Set to 0.0 to disable cropping
SPECIESNET_DIR = "speciesnet"           # extracted Kaggle bundle: only used for speciesnet checkpoints


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

def build_transform(arch, input_size, mean, std):
    if arch == "speciesnet":
        return SpeciesNetPreprocess(read_info(SPECIESNET_DIR)["type"])

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
    elif arch == "efficientnetv2_m":
        m = models.efficientnet_v2_m(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "speciesnet":
        m = build_speciesnet(SPECIESNET_DIR, num_classes)
    else:
        raise ValueError(f"Unknown arch in checkpoint: {arch}")
    return m


# Input handling
# --------------
def find_class_images(root):
    classes = {}
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        if not os.path.isdir(cdir):
            continue
        paths = []
        for dirpath, _, filenames in os.walk(cdir):
            for fn in filenames:
                if fn.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(dirpath, fn))
        if paths:
            classes[name] = sorted(paths)
    return classes


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
    ap.add_argument("--input", required=True, help="folder with one subfolder per class")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load classifier
    ckpt = torch.load(args.model, map_location=device)
    arch = ckpt["arch"]
    class_names = ckpt["class_names"]
    input_size = ckpt.get("input_size", 224)
    mean = ckpt.get("norm_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("norm_std", [0.229, 0.224, 0.225])
    if arch == "speciesnet":
        trained_on = ckpt["speciesnet_version"]
        available = read_info(SPECIESNET_DIR)["version"]
        if trained_on != available:
            raise ValueError(f"checkpoint trained on SpeciesNet {trained_on}, but {SPECIESNET_DIR} holds {available}")

    infer_tf = build_transform(arch, input_size, mean, std)
    model = build_model(arch, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"Loaded {arch} ({input_size}px: {', '.join(class_names)}) on {device}")

    classes = find_class_images(args.input)
    if not classes:
        print(f"No class subfolders with images found in {args.input}")
        return

    unknown = [c for c in classes if c not in class_names]
    if unknown:
        print(f"warning: folder(s) not among model classes, will always score 0%: {', '.join(unknown)}")

    total_images = sum(len(p) for p in classes.values())
    print(f"Found {total_images} images across {len(classes)} class folder(s): {', '.join(classes)}")

    since = time.time()
    done = 0
    per_correct = {}
    per_total = {}
    for cls, paths in classes.items():
        correct = 0
        for path in paths:
            img = Image.open(path).convert("RGB")
            x = infer_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                _, idx = torch.max(model(x), 1)
            if class_names[idx.item()] == cls:
                correct += 1
            done += 1
            progress_bar(done, total_images)
        per_correct[cls] = correct
        per_total[cls] = len(paths)

    elapsed = time.time() - since
    print(f"Processed {total_images} images in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    width = max(len(c) for c in per_total)
    print("\nAccuracy per class:")
    for cls in classes:
        acc = per_correct[cls] / per_total[cls] * 100
        print(f"  {cls:<{width}} : {acc:6.2f}%  ({per_correct[cls]}/{per_total[cls]})")

    n_correct = sum(per_correct.values())
    print(f"\nTotal accuracy: {n_correct / total_images * 100:.2f}%  ({n_correct}/{total_images})")

if __name__ == "__main__":
    main()