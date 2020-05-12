"""
Kaggle competition "Thousand Facial Landmarks"
"""

"""Script for training. Model is ResNeXt50 (pretrained on ImageNet). Training takes ~120 mins (@ RTX 2060)."""

"""Run script: python .\train.py""" 

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import transforms

from utils import NUM_PTS, CROP_SIZE, SCALE_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default='data')
    
    # train on desktop GPU RTX 2060 with 6Gb of VRAM, time per epoch = 120 min
    parser.add_argument("--batch-size", "-b", default=16, type=int)
    
    # the number of epochs in range [1, 12] was chosen by an experienced way
    parser.add_argument("--epochs", "-e", default=8, type=int)
    
    # this lr is best in my opinion
    parser.add_argument("--learning-rate", "-lr", default=3e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    # 1. Prepare data & models
    
    # individual transformations for validation and test data did not give not reduce MSE-loss
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((SCALE_SIZE, SCALE_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.Grayscale(3), ("image",)), # grayscale image for best score
        TransformByKeys(transforms.ColorJitter(brightness=[0.8, 1.2]), ("image",)), # random choose brightness in range
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])
    
    print('----------------------------------------------------------')
    print('Script for Kaggle competition "Thousand Facial Landmarks"')
    print('----------------------------------------------------------')
    
    print('Reading data...')
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="train")
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                       shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'), train_transforms, split="val")
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                     shuffle=False, drop_last=False)

    print("Creating model...")
    device = torch.device("cuda: 0") # default GPU device, because train this net on CPU is eternity :)
    
    # this network was selected through experimentation from the list: resnet18, resnet34, resnext50, resnext101, alexnet, InceptionV3, InceptionV4 etc
    model = models.resnext50_32x4d(pretrained=True)
    
    # adding new layers with regularization (dropout or batchnorm) did not give effect
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_fn = fnn.mse_loss

    # 2. Train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device=device)
        val_loss = validate(model, val_dataloader, loss_fn, device=device)
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}".format(epoch, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(f"{args.name}_best.pth", "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. Predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'test'), train_transforms, split="test")
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True,
                                      shuffle=False, drop_last=False)

    with open(f"{args.name}_best.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(f"{args.name}_test_predictions.pkl", "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, f"{args.name}_submit.csv")


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))