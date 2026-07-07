# Glenfinglas

Camera-trap image deer classifier based on the Glenfinglas dataset. A pretrained CNN classifies each image as containing **deer** or **no_deer**; MegaDetector (PytorchWildlife) is then applied to count the number of animals present in images that are positive for deer. Supports GPU access, or otherwise defaults to CPU.

## Contents

- `train.py` - used to train a transfer-learning classifier. This has already been used to create pretrained models based on resnet18, resnet50 and efficientnet_v2_s which can be applied out-of-the-box.
- `classify.py` - used to apply a pretrained model to classify images: point it at a folder of images and a model, outputs an Excel sheet with one row per image results (class, confidence, count).

## Requirements

Python 3 with:

```
pip install -r requirements.txt
```

A GPU is optional, can be run on a laptop.

## Using a trained model

Download a model (`.pth`), then:

```
python classify.py --model glenfinglas_MODEL.pth --input /path/to/images --output results.xlsx
```

- `--input` is searched recursively, so nested folders are fine.
- `--model` is the path to whichever released `.pth` you want to use.
- `results.xlsx` has one row per image: full pathway in input folder, image name, class, confidence, count.

## Training your own model

Should you wish to train a model on your own dataset:
1. Arrange labelled images in class folders - `data/train/deer/`, `data/train/no_deer/` using naming scheme `location__batch__image.jpg`.
2. Open `train.py` and set `ARCH`, `OUTPUT_PTH`, and optionally adjust the hyperparameters in the CONFIG block. `ARCH` must be one of `resnet18`, `resnet50`, or `efficientnetv2_s` under the current version - other models require deeper changes in the code.
3. Run `python train.py`.

The train/test split is stratified by location and class (~30% test).
The architecture and class names are saved inside the `.pth`.

## Notes
