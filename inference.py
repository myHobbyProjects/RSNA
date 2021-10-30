from pathlib import Path
import os
import sys
# import cv2
import torch
import random
import warnings
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import pydicom as pdcm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.tensorboard.writer import SummaryWriter

import time
from skimage.transform import resize
import json
from classification_main import AverageMeter, MRIdata, mResNet

# main_path = "/Volumes/disk2s2/kaggle/shorts"
main_path = "/home/rudrajit_sengupta/kaggle/dataset"


def inference():
    weight_file_name = f"./model_weights.pt"

    #Data Loader
    transforms = T.Compose([T.ToTensor()])

    data = MRIdata(f"{main_path}/train_labels.csv", transform=transforms)

    train_data_len = int(len(data) * 0.98)
    val_data_len = len(data) - train_data_len

    train_set, val_set = torch.utils.data.random_split(data, [train_data_len, val_data_len])
    # train_loader = DataLoader(
    #                 dataset = train_set,
    #                 shuffle=True,
    #                 batch_size=BATCH_SIZE,
    #                 pin_memory=True,
    #                 num_workers=NUM_WORKERS
    # )

    val_loader = DataLoader(
                    dataset = val_set,
                    shuffle=True,
                    batch_size=1,
                    pin_memory=True,
                    num_workers=1
    )

    net = mResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # resume if weights file exists
    if Path(weight_file_name).is_file():
        ckpt = torch.load(weight_file_name, map_location="cpu")
        start_epoch = ckpt["epoch"]
        net.model.load_state_dict(ckpt["model"])
    else:
        print("Weight file does not exist!")
        return

    preds = []
    target = []
    net.model.eval()
    for idx, data in enumerate(val_loader):
        print("cycle = ", idx)
        # if idx > 10:
        #     break
        with torch.no_grad():
            input, label = data["features"].to(device), data["labels"].to(device)
            output = net(input).to(device)
            # output = torch.sigmoid(output).cpu().numpy().squeeze()
            print(torch.sigmoid(output).cpu().squeeze().item())
            output = round(torch.sigmoid(output).cpu().squeeze().item())
            # if output == 0:
            #     print("idx", idx, "target", label)
            preds.append(output)
            target.append(label)

    accuracy = np.count_nonzero((np.array(preds) == np.array(target)))/len(val_loader)
    print("Accuracy", accuracy)

if __name__ == "__main__":
    inference()