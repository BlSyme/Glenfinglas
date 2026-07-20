# Glenfinglas

Camera-trap image deer classifier based on the Glenfinglas dataset. A pretrained CNN classifies each image as containing *deer* or *no_deer*; MegaDetector (PytorchWildlife) is then applied to count the number of animals present in images that are positive for deer. Results can be visualised with bounding boxes drawn on copies of the input images - the originals are never edited.

Warning: Human images were removed from the dataset before the provided models were trained for privacy reasons. The same action is recommended in your dataset before applying the models, or you may experience erroneous results.

## Contents

- `train.py` - used to train a transfer-learning classifier. This has already been used to create pretrained models based on resnet18, resnet50 and efficientnet_v2_s which can be applied out-of-the-box.
- `speciesnet_backbone.py` - used by `train.py` when `ARCH = speciesnet`
- `classify_and_count.py` - used to apply a pretrained model to classify images and count the number of detections: point it at a folder of images and a model, outputs an Excel sheet with one row per image results (class, class confidence, count, review and per-detection confidence) and a JSON file for use by `draw_detections.py`. In particularly occluded or out-of-focus images, an image may be classified as containing *deer* but MegaDetector may still fail to define a bounding box. A *deer* classified image with count = 0 will be flagged with review = True in the Excel output, indicating that the image classification and count should be manually verified.
- `draw_detections.py` - used to render the JSON output from `classify_and_count.py` to visualise detections using bounding boxes. Also outputs a folder of copies of images flagged for review for convenience.

## Requirements

Python 3.12 or newer is recommended with:

```
pip install -r requirements.txt
```

A GPU is optional, can be run on a laptop.

## Suggested directory structure

Following this structure will allow you to copy the terminal prompt provided for each script with very minimal changes. Should you wish to structure your directory differently, simply adjust the terminal prompts accordingly.

```
Glenfinglas/
├── train.py
├── classify_and_count.py
├── draw_detections.py
├── speciesnet_backbone.py
├── models/
│   ├── glenfinglas_resnet18.pth
│   ├── glenfinglas_resnet50.pth
│   ├── glenfinglas_efficientnetv2_s.pth
│   ├── glenfinglas_efficientnetv2_m.pth
│   └── glenfinglas_speciesnet.pth
├── speciesnet/                                          # (optional) speciesnet model to be trained
├── data/
│   ├── train/                                           # (optional) classified images for training
│   │   ├── deer/
│   │   └── no_deer/
│   └── images/                                          # camera trap images to be classified
├── results/                                             # created when you run classify_and_count.py
│   ├── results.xlsx
│   ├── detections.json
│   ├── detections/                                      # positive detections
│   └── review/                                          # images marked for review
├── requirements.txt
└── README.md
```


## Using a trained model (`classify_and_count.py`)

Download a model (`.pth`), then:

```
python classify_and_count.py --model models/glenfinglas_MODEL.pth --input data/images --output results/results.xlsx --detections results/detections.json
```

- `--model` is the path to whichever released `.pth` you want to use.
- `--input` is the path to the image folder on which you intend to apply the model (searched recursively).
- `--output` is the path to the output Excel file of results e.g., `results.xlsx` will create an Excel file called results in the current working directory. The sheet has one row per image: full pathway in input folder, image name, class, confidence, count, review.
- `--detections` is the path to the output JSON file of detections e.g., `detections.json` will create a JSON file called detections in the current working directory. This file can then be used as input for `draw_detections.py`.
- `--threshold` (optional) can be used to specify the MegaDetector animal detection threshold and should be value in the open interval (0, 1). A lower makes the count more sensitive and will increase the chance of overcounting. Default = 0.2. 

## Displaying your results (`draw_detections.py`)

Run `classify_and_count.py`, then:

```
python draw_detections.py --detections results/detections.json --output results/detections/ --review results/review/
```

- `--detections` is the path to the output JSON file of detections from `classify_and_count.py`. This should match the pathway and file you specified for `--detections` exactly.
- `--output` is the path to the folder you wish to store the annotated copies of the images which feature positive detections.
- `-review` is the path to the folder you wish to store the copies of the images which have been marked for review.

## Training your own model (`train.py`)

Should you wish to train a model on your own dataset:
1. Arrange labelled images in class folders - `data/train/deer/`, `data/train/no_deer/` using naming scheme `location__batch__image.jpg`.
2. Open `train.py` and set `ARCH`, `OUTPUT_PTH`, and optionally adjust the hyperparameters in the CONFIG block (see **Configuration**). `ARCH` must be one of `resnet18`, `resnet50`,  `efficientnetv2_s`, `efficientnetv2_m` or `speciesnet` under the current version - other models require deeper changes in the code.
3. Run:
```
python train.py
```

The train/test split is stratified by location and class (~30% test).
The architecture and class names are saved inside the `.pth`.

## Configuration 

### `train.py`

### Models (`ARCH`)
- `resnet18` - lightweight option for smaller/easier datasets. Pretrained on ImageNet.
- `resnet50` - larger ResNet model which may be suitable for larger datasets. Pretrained on ImageNet.
- `efficientnetv2_s` - alternative lightweight model architecture. Pretrained on ImageNet.
- `efficientnetv2_m` - larger EfficientNet V2 model which may be suitable for larger datasets. Pretrained on ImageNet.
- `speciesnet` - the same EfficientNet V2-M architecture, pretrained by Google on over 65M camera trap images (see **SpeciesNet**).

### Image Preprocessing
- `CropBottom` - crops `CROP_FRAC` fraction of input image height from bottom to remove data overlay bar in camera trap images. Set to 0.0456 for Glenfinglas data; set to 0.0 for no crop. This should be kept the same across `train.py` and `classify_and_count.py`.
- `SquarePad` - resizes input images to square using ImageNet mean grey-fill padding. Preserves aspect ratio.
- `Resize` - scales square images to `INPUT_SIZE`x`INPUT_SIZE`. Default values correspond to native architecture resolutions. Should be well-divisible by 2 if edited.

### Data Split
- `TEST_FRAC` - fraction of input images which are held out for the test set.
- `SEED` - seed for shuffle and split of input into test and train sets, so runs are reproducible.

### Class Imbalance
- `USE_CLASS_WEIGHTS = True` - compensates for large class imbalance: Glenfinglas data is dominated by *no_deer* images, so model could otherwise learn to always predict *no_deer* and achieve decent accuracy.

### Hyperparameters
- `LR = 5e-4` - moderate standard learning rate for AdamW optimiser.
- `WEIGHT_DECAY = 1e-2` - limits overfitting: increase for smaller datasets, decrease for larger datasets.
- `STEP_SIZE = 3`, `GAMMA = 0.3` - multiply learning rate by 0.3 every 3 epochs for finer convergence in later epochs.
- `NUM_EPOCHS = 5` - iterate over training data 5 times: relatively few iterations to limit overfitting on "easy" dataset.

### `classify_and_count.py`
- `EMPTY_LABEL` - the name of your 'empty' class folder to be skipped by MegaDetector.
- `MD_VERSION` - selected MegaDetector version. `MDV6-yolov10-e` is set as default since it provides the best accuracy in our testing; `MDV6-yolov10-c` is a lighter alternative which should provide a large speed-up on a CPU, but it is likely to increase the number of results marked for review.
- `MD_THRESH` - default MegaDetector animal detection threshold if no `--threshold` argument is specified when the script is run.

## SpeciesNet

Species classification using the SpeciesNet backbone (EfficientNet V2-M) derives from a
release by Google as part of `cameratrapai`:

> Gadot T, Istrate Ș, Kim H, Morris D, Beery S, Birch T, Ahumada J.
> [To crop or not to crop: Comparing whole-image and cropped classification on a large dataset of camera trap images](https://doi.org/10.1049/cvi2.12318).
> IET Computer Vision. 2024;18(8):1193–1208.

Software: https://github.com/google/cameratrapai