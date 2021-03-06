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

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = 12, 6

# main_path = "//rsna-miccai-brain-tumor-radiogenomic-classification"
# main_path = "/Volumes/disk2s2/kaggle/dataset/rsna/"
# main_path = "/Volumes/disk2s2/kaggle/shorts"
main_path = "/home/rudrajit_sengupta/kaggle/dataset"
# main_path = "/home/rudrajit_sengupta/kaggle/short_dataset"

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 16
NUM_SAMPLES = 64
EPOCHS = 25
LRate = 0.001
SEED = 42
NUM_WORKERS = 8
TRAIN_SPLIT = 0.85

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def build_args():
    parser.add_argument(
        "--stats_file",
        default="",
        type=str,
        help="Operation mode",
    )
    args = parser.parse_args()
    return args

def filter_irrelevant_data(path, indices):
    # train_df = pd.read_csv(f"{main_path}/train_labels.csv")
    train_df = pd.read_csv(path)
    # test_df = pd.read_csv(f"{main_path}/sample_submission.csv")

    for ids in indices:
        train_df.drop(train_df[train_df["BraTS21ID"] == ids].index, inplace=True)

    return train_df #, test_df


def sort_paths(path):
    return sorted(path, key=lambda x: int(x.split('-')[1].split('.')[0]))


def read_dicom_img(dataset_dir, file_id):
    mp_mri_type = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

    file_path = f"{main_path}/{dataset_dir}/{file_id.zfill(5)}"
    final_img = []
    for mri_type in mp_mri_type:
        final_file_path = f"{file_path}/{mri_type}"
        # sorted_files = sort_paths(os.listdir(final_file_path))
        sorted_files = os.listdir(final_file_path)

        for i in range(len(sorted_files)):
            data = pdcm.dcmread(f"{final_file_path}/{sorted_files[i]}")

            img_data = data.pixel_array
            # img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            img_data = resize(img_data, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) + 1)
            final_img.append(img_data)
            if i > NUM_SAMPLES:
                # print(file_id, mri_type)
                # print(".")
                break

    final_img = sorted(final_img, key=lambda x: np.mean(x), reverse=True)[:NUM_SAMPLES]
    return np.asarray(final_img)

# Data Loader
class MRIdata(Dataset):
    def __init__(self, path_to_data, train=True, transform=None):
        self.data = pd.read_csv(path_to_data)
        #self.data = filter_irrelevant_data(path_to_data, [109, 123, 709])
        self.labels = self.data['MGMT_value']
        self.transform = transform
        self.train = train
        self.train_dir = ("train" if train else "test")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp_data = read_dicom_img(self.train_dir, str(self.data['BraTS21ID'][index]))
        inp_data = self.transform(inp_data).permute(1, 0, 2)
        # inp_data = self.transform(inp_data)

        if self.train:
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            final_data = {"features": torch.tensor(inp_data).float(),
                          "labels": labels}
        else:
            final_data = {"features": torch.tensor(inp_data).float(),
                          "IDs": self.data['BraTS21ID'][index]}

        return final_data

def get_elapsed_time(start):
    return time.time() - start

# Define Network Architecture
class MRINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(self.conv_layer(in_chan=64, out_chan=128),
                                  self.conv_layer(in_chan=128, out_chan=128),
                                  self.conv_layer(in_chan=128, out_chan=256))

        self.fc = nn.Sequential(nn.Linear(9216, 512),
                                nn.Dropout(p=0.15),
                                nn.Linear(512, 1))

    def conv_layer(self, in_chan, out_chan):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(out_chan))

        return conv_layer

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class mResNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # self.args = args
        self.model = torchvision.models.resnet50(pretrained=False)
        # self.model = torch.nn.Sequential(*(list(m.children())))

        self.model.conv1 = nn.Conv2d(
            in_channels= NUM_SAMPLES, #args.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.fc2 = nn.Linear(1000, 1, bias=True)

        # Proper ResNet weight initialization
        # nn.init.kaiming_normal_(
        #     self.model[0].weight, mode="fan_out", nonlinearity="relu"
        # )

    def forward(self, x):
        output = self.model(x)
        output = self.fc2(output)
        return output

def accuracy(output, targets):
    return np.round(output).eq(targets).sum().item()/ len(targets)

def do_train(epoch, train_loader, model, optimizer, criterion, writer, stats_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # total_loss = 0.0
    # count = 0

    losses = AverageMeter()
    batchAcc = AverageMeter()

    start = time.time()
    for indx, data in enumerate(train_loader, 0):

        inputs, labels = data["features"].to(device), data["labels"].to(device)
        optimizer.zero_grad()

        outputs = model(inputs).squeeze(1).to(device)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        delta_t = time.time() - start

        losses.update(loss.detach().item(), outputs.shape[0])

        outputs = torch.sigmoid(outputs).cpu().squeeze().detach()
        acc = accuracy(outputs, labels)
        batchAcc.update(acc)

        # total_loss += loss.detach().item()
        # count += 1
        # acc_step_loss = total_loss / count
        # stats = dict(
        #     Epoch=epoch,
        #     Step=indx,
        #     Time=round((time.time() - start), 0),
        #     Loss=round(acc_step_loss, 4))
        #
        # print(json.dumps(stats))
        # print(json.dumps(stats), file=stats_file)
        if writer:
            writer.add_scalar(
                "Loss/train", loss.item(), indx + (epoch * len(train_loader))
            )
            writer.add_scalar(
                "Accuracy/train", acc, indx + (epoch * len(train_loader))
            )

    return losses.avg, batchAcc.avg, delta_t


# def do_train(train_loader, stats_file, writer):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     # stats_file = args.stats_file
#     # net = MRINet().to(device)
#     net = mResNet().to(device)
#
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(net.parameters(), lr=LRate)
#     net.model.train()
#
#     for epoch in range(EPOCHS):
#         total_loss = 0.0
#         count = 0
#
#         start = time.time()
#         for indx, data in enumerate(train_loader, 0):
#
#             inputs, labels = data["features"].to(device), data["labels"].to(device)
#             optimizer.zero_grad()
#
#             outputs = net(inputs).squeeze(1)
#
#             loss = criterion(outputs, labels)
#             loss.backward()
#             total_loss += loss.detach().item()
#             optimizer.step()
#
#             count += 1
#             acc_step_loss = total_loss / count
#             stats = dict(
#                 Epoch=epoch,
#                 Step = indx,
#                 Time=round((time.time() - start), 0),
#                 Loss=round(acc_step_loss, 4))
#
#             print(json.dumps(stats))
#             print(json.dumps(stats), file=stats_file)
#             if writer:
#                 writer.add_scalar(
#                     "training_loss", loss.item(), indx + (epoch * len(train_loader))
#                 )
#             # print(".")
#
#         print(f"Epoch:{epoch}/{EPOCHS} - Loss:{acc_step_loss}")
#
#
#     print("Training Complete")

def do_validation(epoch, val_loader, model, criterion, writer, stats_file):
    model.eval()
    losses = AverageMeter()
    batchAcc = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    for indx, data in enumerate(val_loader, 0):
        with torch.no_grad():
            inputs, labels = data["features"].to(device), data["labels"].to(device)
            output = model(inputs).to(device)
            loss = criterion(output[:,0], labels)
            losses.update(loss.detach().item(), output.shape[0])

            output = torch.sigmoid(output).cpu().squeeze().detach()
            acc = accuracy(output,labels)
            batchAcc.update(acc)
            delta_t = time.time() - start

            if writer:
                writer.add_scalar(
                    "Loss/validation", loss.item(), indx + (epoch * len(val_loader))
                )
                writer.add_scalar(
                    "Accuracy/validation", acc, indx + (epoch * len(val_loader))
                )

    return losses.avg, batchAcc.avg, delta_t

# def do_inference(test_loader):
#     preds = []
#
#     for query in test_loader:
#         with torch.no_grad():
#             output = net(query["features"].to(device))
#             output = torch.sigmoid(output).cpu().numpy().squeeze()
#             preds.append(output)
#
#     return preds

def main_rsna():
    prevLoss = float("inf")

    #Filter Irrelevant Data
    # train_df, test_df = filter_irrelevant_data([109, 123, 709])

    stats_file = open("stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    log_dir = "./logs"
    writer = SummaryWriter(log_dir)

    weight_file_name = f"./model_weights.pt"

    #Data Loader
    transforms = T.Compose([T.ToTensor()])

    data = MRIdata(f"{main_path}/train_labels.csv", transform=transforms)

    train_data_len = int(len(data) * TRAIN_SPLIT)
    val_data_len = len(data) - train_data_len

    train_set, val_set = torch.utils.data.random_split(data, [train_data_len, val_data_len])
    train_loader = DataLoader(
                    dataset = train_set,
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    pin_memory=True,
                    num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
                    dataset = val_set,
                    shuffle=False,
                    batch_size=BATCH_SIZE,
                    pin_memory=True,
                    num_workers=NUM_WORKERS
    )
    # test_data = MRIdata(f"{main_path}/sample_submission.csv", train=False,
    #                     transform=transforms)
    # test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    # net = MRINet().to(device)
    net = mResNet()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LRate)

    # resume if weights file exists
    if Path(weight_file_name).is_file():
        ckpt = torch.load(weight_file_name, map_location="cpu")
        start_epoch = ckpt["epoch"]
        net.model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        count = 0

        thisTrainbatchLoss, thisTrainbatchAcc, Train_delta_t = do_train(epoch, train_loader, net, optimizer, criterion, writer, stats_file)

        thisValbatchLoss, thisValbatchAcc, Val_delta_t = do_validation(epoch, val_loader, net, criterion, writer, stats_file)

        stats = dict(
            Epoch=epoch,
            BatchLoss=[round(thisTrainbatchLoss, 4), round(thisValbatchLoss, 4)],
            BatchAcc=[round(thisTrainbatchAcc, 4), round(thisValbatchAcc, 4)],
            Time = [round(Train_delta_t), round(Val_delta_t)]
        )
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

        # print(f"Epoch:{0}/{1}\tLoss:{2} {3}\tAccuracy:{4} {5}\n"
        #       .format(epoch, EPOCHS, thisTrainbatchLoss, thisValbatchLoss, thisTrainbatchAcc, thisValbatchAcc))

        if thisValbatchLoss < prevLoss:
            prevLoss = thisValbatchLoss
            state = dict(
                epoch=epoch + 1,
                model=net.model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, weight_file_name)

    if writer:
        print("Closing tensorboard...")
        writer.close()


    #inderference
    # do_inference(test_loader)


if __name__ == "__main__":

    main_rsna()
