"""
SpeciesNet as a backbone for train.py.

SpeciesNet's classifier is EfficientNetV2-M trained on ~44M camera-trap images. 
It is the same architecture "efficientnetv2_m" with camera-trap pretraining
instead of ImageNet pretraining.

The backbone is frozen and only the final layer is trained. Unlike the ImageNet 
backbones, which fine-tune end to end.
"""

import os
import json

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


# Config
# ------
IMG_SIZE = 480
MAX_CROP_RATIO = 0.3
MAX_CROP_SIZE = 400


def read_info(model_dir):
    # version, type (full_image), classifier, classifier_labels
    with open(os.path.join(model_dir, "info.json"), encoding="utf-8") as f:
        return json.load(f)


# Image preprocessing
# -------------------
class SpeciesNetPreprocess:
    """
    full_image crops the top and bottom of the frame. It also removes the data bar, 
    so CropBottom is redundant here.
    """
    def __init__(self, model_type, augment=None):
        self.model_type = model_type
        self.augment = augment

    def __call__(self, image):
        t = F.pil_to_tensor(image)                      # [0, 255]
        t = F.convert_image_dtype(t, torch.float32)     # [0.0, 1.0]

        if self.model_type == "full_image":             # SpeciesNet centre crop
            keep = max(
                int(image.height * (1.0 - MAX_CROP_RATIO)),
                image.height - MAX_CROP_SIZE,
            )
            t = F.center_crop(t, [keep, image.width])

        t = F.resize(t, [IMG_SIZE, IMG_SIZE], antialias=False)  # 480x480

        if self.augment is not None:
            t = self.augment(t)                         # RandomHorizontalFlip() if train else None

        t = F.convert_image_dtype(t, torch.uint8)       # remove decimal precision (matches Google code)
        t = F.convert_image_dtype(t, torch.float32)
        return t.permute(1, 2, 0)                       # (3, 480, 480) instead of (480, 480, 3)


# SpeciesNet
# ----------
class SpeciesNet(nn.Module):
    def __init__(self, backbone, num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(num_features, num_classes)        # New final layer

    def forward(self, x):
        return self.head(self.backbone(x))              # (N, num_classes)


# Remove final layer
# ------------------
def cut_head(graph):
    """
    Instead of swapping the final layer, cut the graph short before the final linear 
    step so we can redefine a new one.
    """
    out_node = next(n for n in graph.graph.nodes if n.op == "output")
    bias_add = out_node.args[0]
    matmul = bias_add.args[0]
    features = matmul.args[0]

    out_node.args = (features,)
    
    # cleanup
    graph.graph.eliminate_dead_code()
    graph.recompile()
    graph.delete_all_unused_submodules()
    return graph


# Build
# -----
def build_speciesnet(model_dir, num_classes):
    info = read_info(model_dir)

    # weights_only=False: object, not state_dict.
    backbone = torch.load(
        os.path.join(model_dir, info["classifier"]), map_location="cpu", weights_only=False
    )
    backbone = cut_head(backbone)

    with torch.no_grad():
        probe = backbone(torch.zeros(1, IMG_SIZE, IMG_SIZE, 3))
    if probe.dim() != 2:
        raise RuntimeError(
            f"expected pooled features (N, C) after cut_head, got {tuple(probe.shape)}: the graph layout of SpeciesNet {info['version']} is not what this expects"
        )
    num_features = probe.shape[1]

    for p in backbone.parameters():
        p.requires_grad = False

    return SpeciesNet(backbone, num_features, num_classes)