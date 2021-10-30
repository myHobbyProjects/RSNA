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

# main_path = "/Volumes/disk2s2/kaggle/shorts"
# main_path = "/Volumes/disk2s2/kaggle/dataset/rsna/"
main_path = "/home/rudrajit_sengupta/kaggle/dataset"

# Hyperparameters
IMG_SIZE = 224
NUM_IN_CHNL = 1
BATCH_SIZE = 2
NUM_SAMPLES = 64
EPOCHS = 50
LRate = 0.003 #0.001
SEED = 42
NUM_WORKERS = 8
TRAIN_SPLIT = 1.0 #0.85
NUM_MINI_BATCHES = 32
BAD_DATA_LIST = [109, 123, 709]

class mResNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # self.args = args
        self.model = torchvision.models.resnet152(pretrained=False)
        # self.model = torch.nn.Sequential(*(list(m.children())))

        self.model.conv1 = nn.Conv2d(
            in_channels= NUM_IN_CHNL, #args.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.fc2 = nn.Linear(1000, 512, bias=True)
        self.fc3 = nn.Linear(512, 1, bias=True)

        # Proper ResNet weight initialization
        # nn.init.kaiming_normal_(
        #     self.model[0].weight, mode="fan_out", nonlinearity="relu"
        # )

    def forward(self, x):
        output = self.model(x)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

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

def accuracy(output, targets):
    return np.round(output).eq(targets).sum().item()/ len(targets)

def filter_irrelevant_data(path, indices):
    missing_keys = []
    # train_df = pd.read_csv(f"{main_path}/train_labels.csv")
    train_df = pd.read_csv(path)
    # test_df = pd.read_csv(f"{main_path}/sample_submission.csv")

    for ids in indices:
        missing_keys.append(train_df[train_df["BraTS21ID"] == ids].index[0])
        train_df.drop(train_df[train_df["BraTS21ID"] == ids].index, inplace=True)

    return train_df, missing_keys

def read_dicom_img(dataset_dir, file_id):
    imgs=[]
    dir_path = f"{main_path}/{dataset_dir}/{file_id.zfill(5)}"
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # if root.split("/")[-1] in ['FLAIR', 'T1w', 'T2w']: #['FLAIR', 'T1w', 'T1wCE', 'T2w']:
        #     continue
        for name in files:
            fname = os.path.join(root, name)
            img = pdcm.dcmread(fname).pixel_array
            if img.mean() == 0:
                continue
            img = resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
            img = (img - np.min(img)) / (np.max(img) + 1)
            imgs.append(img)

    return np.asarray(imgs)
    # return np.expand_dims(x, axis=0) #np.asarray(imgs)

class MRIdata(Dataset):
    def __init__(self, path_to_data, train=True, transform=None):
        # self.data = pd.read_csv(path_to_data)
        self.data,  self.missing_keys = filter_irrelevant_data(path_to_data, BAD_DATA_LIST)
        self.labels = self.data['MGMT_value']
        self.transform = transform
        self.train = train
        # self.train_dir = ("train" if train else "test")
        self.train_dir = ("target" if train else "test")
        self.data_path = path_to_data

    def __len__(self):
        return len(self.data)
        # return sum([len(files) for (root, dirs, files) in os.walk(self.data_path)])

    def __getitem__(self, index):
        # print("index", index)
        if index in self.missing_keys:
            index += 1
        filename = str(self.data['BraTS21ID'][index])
        # try:
        #     filename = str(self.data['BraTS21ID'][index])
        # except KeyError:
        #     print("key {0} not found!".format(index))

        inp_data = read_dicom_img(self.train_dir, filename)
        inp_data = self.transform(inp_data).permute(1, 0, 2).to(torch.float)
        # inp_data[0] = self.transform(inp_data[0]).permute(1, 0, 2)

        if self.train:
            labels = torch.tensor(self.labels[index], dtype=torch.float)
            # final_data = {"features": torch.tensor(inp_data).float(),
            #               "labels": labels}
            final_data = {"features": inp_data,
                          "labels": labels}
        else:
            final_data = {"features": torch.tensor(inp_data).float(),
                          "IDs": self.data['BraTS21ID'][index]}

        return final_data

def collate_fn(data):
    """
    https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
    :param data:
    :return:
    """
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    maxLength = float('-inf')
    for thisData in data:
        if thisData['features'].shape[0] > maxLength:
            maxLength = thisData['features'].shape[0]

    for thisData in data:
        thisData['features'] = torch.unsqueeze(thisData['features'], 1)
        n_feat, _, x, y = thisData['features'].shape
        thisData['len'] = n_feat
        zeros = torch.zeros((maxLength - n_feat), 1, x, y)
        thisData['features'] = torch.cat((thisData['features'], zeros), 0)

    return data

def do_something(IS_TRAIN, epoch, loader, model, optimizer, criterion, writer=None, stats_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if IS_TRAIN == "OK_TRAIN":
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    batchAcc = AverageMeter()

    start = time.time()
    for batchIdx, batch in enumerate(loader, 0):
        for object in batch:
            thisLen = object['len']
            mini_batches = list(torch.tensor_split(object['features'][:thisLen], \
                                                   (1 if thisLen < NUM_MINI_BATCHES else NUM_MINI_BATCHES)))
            random.shuffle(mini_batches)
            tr_labels_minibat = object['labels'].repeat(object['len'])

            for mini in mini_batches:
                input = {'images': mini, 'labels':object['labels'].repeat(mini.shape[0])}
                images = input['images'].to(device)
                labels = input['labels'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(IS_TRAIN == 'OK_TRAIN'):
                    outputs = model(images).squeeze(1).to(device)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                losses.update(loss.detach().item(), outputs.shape[0])

                outputs = torch.sigmoid(outputs).cpu().squeeze().detach()
                acc = accuracy(outputs, input['labels'])
                batchAcc.update(acc)

                if writer:
                    writer.add_scalar(
                        "Loss/train", loss.item(), indx + (epoch * len(loader))
                    )
                    writer.add_scalar(
                        "Accuracy/train", acc, indx + (epoch * len(loader))
                    )
        delta_t = time.time() - start

    return losses.avg, batchAcc.avg, delta_t

def test_train():
    # BATCH_SIZE = 2
    # NUM_MINI_BATCHES = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mResNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LRate)

    transforms = T.Compose([T.ToTensor()])
    data = MRIdata(f"{main_path}/train_labels.csv", transform=transforms)

    train_data_len = int(len(data) * TRAIN_SPLIT)
    val_data_len = len(data) - train_data_len
    train_set, val_set = torch.utils.data.random_split(data, [train_data_len, val_data_len])

    if NUM_WORKERS == 0:
        train_loader = DataLoader(
            dataset = train_set,
            shuffle=False,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            # persistent_workers=True,
            # prefetch_factor=1,
        )
    else:
        train_loader = DataLoader(
                        dataset = train_set,
                        shuffle=False,
                        batch_size=BATCH_SIZE,
                        pin_memory=True,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_fn,
                        persistent_workers=True,
                        # prefetch_factor=1,
        )

    val_loader = DataLoader(
        dataset = val_set,
        shuffle=False,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    start_epoch = 0
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        count = 0

        loader = train_loader
        thisTrainbatchLoss, thisTrainbatchAcc, Train_delta_t = \
            do_something("OK_TRAIN", epoch, loader, net, optimizer, criterion, None, None)

        # loader = val_loader
        # thisValbatchLoss, thisValbatchAcc, Val_delta_t = \
        #     do_something("OK_VAL", epoch, loader, net, optimizer, criterion, None, None)

        # thisTrainbatchLoss, thisTrainbatchAcc, Train_delta_t = \
        #     do_train(epoch, train_loader, net, optimizer, criterion, None, None)

        # thisValbatchLoss, thisValbatchAcc, Val_delta_t = do_validation(epoch, val_loader, net, criterion, writer, stats_file)
        thisValbatchLoss, thisValbatchAcc, Val_delta_t = (0, 0, 0)
        stats = dict(
            Epoch=epoch,
            BatchLoss=[round(thisTrainbatchLoss, 4), round(thisValbatchLoss, 4)],
            BatchAcc=[round(thisTrainbatchAcc, 4), round(thisValbatchAcc, 4)],
            Time = [round(Train_delta_t), round(Val_delta_t)]
        )
        print(json.dumps(stats))
        # print(json.dumps(stats), file=stats_file)




if __name__ == "__main__":
    test_train()