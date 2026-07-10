"""
Draw MegaDetector bounding boxes onto copies of the source images.

Reads the detections.json written by classify_and_count.py and saves annotated
copies into an output folder, preserving the nested structure of the input folder.
Images marked for review are copied unannotated into a separate review folder.

Source images are never modified.

Usage:
    python draw_detections.py --detections detections.json --output detections/ --review review/
"""

import os
import json
import shutil
import argparse

from PIL import Image, ImageDraw, ImageFont


# Config
# ------
BOX_COLOUR = (0, 255, 0)
BOX_WIDTH = 4                   
LABEL_SIZE = 28


# Create boxes
# ------------
def draw_boxes(image, records, font):
    draw = ImageDraw.Draw(image)
    for r in records:
        x1, y1, x2, y2 = r["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOUR, width=BOX_WIDTH)
        text = f"Deer {r['confidence']:.2f}"
        tx, ty = x1, max(0, y1 - LABEL_SIZE - 2)
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=BOX_COLOUR)
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)
    return image


# Main
# ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="detections.json", help="JSON from classify_and_count.py")
    ap.add_argument("--output", default="detections", help="folder for annotated copies")
    ap.add_argument("--review", default="review", help="folder for images marked for review")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="only draw boxes at or above this confidence")
    args = ap.parse_args()

    with open(args.detections) as f:
        payload = json.load(f)

    input_root = payload["input_root"]
    images = payload["images"]
    print(f"Loaded {len(images)} images from {args.detections} (MegaDetector {payload['md_version']}, threshold {payload['threshold']})")

    font = ImageFont.load_default()
    written = 0
    reviewed = 0
    missing = 0

    def mirrored_path(src, root):
        dst = os.path.join(root, os.path.relpath(src, input_root))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        return dst

    for entry in images:
        src = entry["image_path"]
        if not os.path.exists(src):
            missing += 1
            continue

        records = [r for r in entry["detections"] if r["confidence"] >= args.min_confidence]
        
        if not records:
            shutil.copy2(src, mirrored_path(src, args.review))
            reviewed += 1
            continue

        image = Image.open(src).convert("RGB")
        draw_boxes(image, records, font)
        image.save(mirrored_path(src, args.output))
        written += 1

    print(f"Wrote {written} annotated image(s) -> {args.output}")
    if reviewed:
        print(f"Copied {reviewed} image(s) to be reviewed -> {args.review}")
    if missing:
        print(f"{missing} source image(s) listed in the JSON no longer exist and were skipped")

if __name__ == "__main__":
    main()