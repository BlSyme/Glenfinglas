"""
Transfer-learned CNN for Glenfinglas data.

Follows the PyTorch transfer learning tutorial:
https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import os
import time
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional
from PIL import Image, ImageOps


# Config
# ------
DATA_DIR = "data/train"
CLASS_FOLDERS = ["no_deer", "deer"]              # class labels

ARCH = "efficientnetv2_s"                        # resnet18, resnet50 or efficientnetv2_s
OUTPUT_PTH = "glenfinglas_efficientnetv2_s.pth"  # model name

TEST_FRAC = 0.30                                 # fraction of images held out for test set
SEED = 42                                        # random seed for reproducibility 

BATCH_SIZE = 64                                  # number of images per batch: reduce if VRAM is a limitation
NUM_EPOCHS = 6                                   # number of iterations through train set during training
NUM_WORKERS = 4

LR = 5e-4                                        # learning rate: higher = more aggresive ("bigger steps"), may overstep / lower = less aggresive, slower to learn
WEIGHT_DECAY = 1e-2                              # weight decay to limit overfitting
STEP_SIZE = 3                                    # epoch schedule after which lr is multiplied by gamma
GAMMA = 0.3                                      # multiplier for learning rate applied every step_size epochs   

IMG_EXTS = (".jpg", ".jpeg", ".png")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Image preprocessing
# -------------------
CROP_FRAC = 0.0456                               # fraction of bottom of images to remove e.g., data bar overlay: set to 0.0 to disable cropping
INPUT_SIZE = 384                                 # image size after resize: higher = capture finer detail, slower to train / lower = loses some detail, faster to train
NORM_MEAN = [0.485, 0.456, 0.406]                
NORM_STD = [0.229, 0.224, 0.225]                 
PAD_FILL = tuple(int(m * 255) for m in NORM_MEAN)

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
    
data_transforms = {
    "train": transforms.Compose([
        CropBottom(CROP_FRAC),
        SquarePad(PAD_FILL),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
    "test": transforms.Compose([
        CropBottom(CROP_FRAC),
        SquarePad(PAD_FILL),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ]),
}
    

# Location + class stratified split 
# ---------------------------------
def gather_samples():
    samples = []

    for cid, folder in enumerate(CLASS_FOLDERS):
        cdir = os.path.join(DATA_DIR, folder)

        for img in sorted(os.listdir(cdir)):
            if img.lower().endswith(IMG_EXTS):
                location = img.split("__")[0]
                samples.append((os.path.join(cdir, img), cid, location))

    return samples

def stratified_split(samples):
    """Proportional split within each (location, class) cell"""
    groups = defaultdict(list)
    for s in samples:
        groups[(s[2], s[1])].append(s)

    rng = random.Random(SEED)
    train, test = [], []
    for items in groups.values():
        items = list(items)
        rng.shuffle(items)
        n_test = round(len(items) * TEST_FRAC)
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    return train, test

class LabeledDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label, _ = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
    

# Model
# -----
def build_model(arch, num_classes):
    if arch == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnetv2_s":
        model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown ARCH: {arch}")
    return model

def train_model(model, criterion, optimiser, scheduler, dataloaders, dataset_sizes, num_epochs):
    since = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
 
        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimiser.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        train_loss = running_loss / dataset_sizes["train"]
        train_acc = running_corrects.double() / dataset_sizes["train"]
        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
 
        # test 
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["test"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / dataset_sizes["test"]
        val_acc = running_corrects.double() / dataset_sizes["test"]
        print(f"test Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    return model


# Main
# ----
def main():
    samples = gather_samples()
    train_samples, test_samples = stratified_split(samples)

    image_datasets = {
        "train": LabeledDataset(train_samples, data_transforms["train"]),
        "test": LabeledDataset(test_samples, data_transforms["test"]),
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == "train"), num_workers=NUM_WORKERS) for x in ["train", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
    class_names = list(CLASS_FOLDERS)

    print(f"Device: {device}. Arch: {ARCH} ({INPUT_SIZE}px).")
    print(f"Train: {dataset_sizes['train']}. Test: {dataset_sizes['test']}.")

    train_counts = [sum(1 for _, c, _ in train_samples if c == i) for i in range(len(class_names))]
    for name, n in zip(class_names, train_counts):
        print(f"train[{name}] = {n}")

    model = build_model(ARCH, len(class_names)).to(device)

    total = sum(train_counts)
    weights = [total / (len(class_names) * c) if c > 0 else 0.0 for c in train_counts]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    print(f"class weights: {[round(w, 3) for w in weights]}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)

    model = train_model(model, criterion, optimiser, scheduler, dataloaders, dataset_sizes, NUM_EPOCHS)

    torch.save({"arch": ARCH, "class_names": class_names, "input_size": INPUT_SIZE, "norm_mean": NORM_MEAN, "norm_std": NORM_STD, "state_dict": model.state_dict()}, OUTPUT_PTH)
    print(f"Saved checkpoint -> {OUTPUT_PTH}")

if __name__ == "__main__":
    main()