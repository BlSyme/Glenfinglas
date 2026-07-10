# Glenfinglas

Camera-trap image deer classifier based on the Glenfinglas dataset. A pretrained CNN classifies each image as containing *deer* or *no_deer*; MegaDetector (PytorchWildlife) is then applied to count the number of animals present in images that are positive for deer. Results can be visualised with bounding boxes drawn on copies of the input images - the originals are never edited.

Warning: Human images were removed from the dataset before the provided models were trained for privacy reasons. The same action is recommended in your dataset before applying the models, or you may experience erroneous results.

## Contents

- `train.py` - used to train a transfer-learning classifier. This has already been used to create pretrained models based on resnet18, resnet50 and efficientnet_v2_s which can be applied out-of-the-box.
- `classify_and_count.py` - used to apply a pretrained model to classify images and count the number of detections: point it at a folder of images and a model, outputs an Excel sheet with one row per image results (class, confidence, count, review) and a JSON file for use by `draw_detections.py`. In particularly occluded or out-of-focus images, an image may be classified as containing *deer* but MegaDetector may still fail to define a bounding box. A *deer* classified image with count = 0 will be flagged with review = True in the Excel output, indicating that the image classification and count should be manually verified.
- `draw_detections.py` - used to render the JSON output from `classify_and_count.py` to visualise detections using bounding boxes. Also outputs a folder of copies of images flagged for review for convenience.

## Requirements

Python 3.12 or newer with:

```
pip install -r requirements.txt
```

A GPU is optional, can be run on a laptop.

## Using a trained model (`classify_and_count.py`)

Download a model (`.pth`), then:

```
python classify_and_count.py --model glenfinglas_MODEL.pth --input path/to/images --output results.xlsx --detections detections.json
```

- `--model` is the path to whichever released `.pth` you want to use.
- `--input` is the path to the image folder on which you intend to apply the model (searched recursively).
- `--output` is the path to the output Excel file of results e.g., `results.xlsx` will create an Excel file called results in the current working directory. The sheet has one row per image: full pathway in input folder, image name, class, confidence, count, review.
- `--detections` is the path to the output JSON file of detections e.g., `detections.json` will create a JSON file called detections in the current working directory. This file can then be used as input for `draw_detections.py`.

## Displaying your results (`draw_detections.py`)

Run `classify_and_count.py`, then:

```
python draw_detections.py --detections detections.json --output path/to/output --review path/to/review
```

- `--detections` is the path to the output JSON file of detections from `classify_and_count.py`. This should match the pathway and file you specified for `--detections` exactly.
- `--output` is the path to the folder you wish to store the annotated copies of the images which feature positive detections.
- `-review` is the path to the folder you wish to store the copies of the images which have been marked for review.

## Training your own model (`train.py`)

Should you wish to train a model on your own dataset:
1. Arrange labelled images in class folders - `data/train/deer/`, `data/train/no_deer/` using naming scheme `location__batch__image.jpg`.
2. Open `train.py` and set `ARCH`, `OUTPUT_PTH`, and optionally adjust the hyperparameters in the CONFIG block. `ARCH` must be one of `resnet18`, `resnet50`, or `efficientnetv2_s` under the current version - other models require deeper changes in the code.
3. Run:
```
python train.py
```


The train/test split is stratified by location and class (~30% test).
The architecture and class names are saved inside the `.pth`.
