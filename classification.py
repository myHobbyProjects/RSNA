from pathlib import Path
import os
import sys
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
from torch.utils.data import random_split, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
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
BATCH_SIZE = 4
NUM_SAMPLES = 16
EPOCHS = 100
LRate = 0.001
WEIGHT_DECAY=0.000
SEED = 42
NUM_WORKERS = 4
TRAIN_SPLIT = 0.85
BAD_DATA_LIST = [109, 123, 709]
KFOLD_NUM_SPLIT = 5

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
    missing_keys = []
    # train_df = pd.read_csv(f"{main_path}/train_labels.csv")
    train_df = pd.read_csv(path)
    # test_df = pd.read_csv(f"{main_path}/sample_submission.csv")

    for ids in indices:
        missing_keys.append(train_df[train_df["BraTS21ID"] == ids].index[0])
        train_df.drop(train_df[train_df["BraTS21ID"] == ids].index, inplace=True)

    return train_df, missing_keys


def sort_paths(path):
    return sorted(path, key=lambda x: int(x.split('-')[1].split('.')[0]))


class cacheClass():
  def __init__(self):
    self.cache = {}

  def update(self, objid, dataset_dir, mritype, start):
    key = "%s_%s_%s"%(dataset_dir, mritype, objid)
    self.cache[key] = start

  def get(self, objid, dataset_dir, mritype):
    key = "%s_%s_%s"%(dataset_dir, mritype, objid)
    if key not in self.cache.keys():
      return None

    return self.cache[key]

# pick selected images
def read_dicom_img(dataset_dir, file_id, cache):
    # mp_mri_type = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    mp_mri_type = ['T1w']

    file_path = f"{main_path}/{dataset_dir}/{file_id.zfill(5)}"
    final_img = []
    isum = []

    for mri_type in mp_mri_type:
        start = cache.get(file_id.zfill(5), dataset_dir, mri_type)

        final_file_path = f"{file_path}/{mri_type}"
        sorted_files = sort_paths(os.listdir(final_file_path))

        if start is not None:
         for i in range(start, (start + NUM_SAMPLES)):
            data = pdcm.dcmread(f"{final_file_path}/{sorted_files[i]}")

            img_data = data.pixel_array
            isum.append(np.sum(img_data))
            img_data = resize(img_data, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

            img_data = (img_data - np.min(img_data)) / (np.max(img_data) + 1)
            final_img.append(img_data)
         return np.asarray(final_img)

        for i in range(len(sorted_files)):
            data = pdcm.dcmread(f"{final_file_path}/{sorted_files[i]}")

            img_data = data.pixel_array
            isum.append(np.sum(img_data))
            img_data = resize(img_data, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

            img_data = (img_data - np.min(img_data)) / (np.max(img_data) + 1)
            final_img.append(img_data)

        maxsumpos = np.argmax(np.array(isum))
        start = int(maxsumpos - (NUM_SAMPLES/2))
        if start < 0:
          start = 0

        if (start + NUM_SAMPLES) > len(final_img):
          delta = (start + NUM_SAMPLES) - len(final_img)
          start -= delta

        cache.update(file_id.zfill(5), dataset_dir, mri_type, start)

        final_img = final_img[start:(start + NUM_SAMPLES)]
    return np.asarray(final_img)


# Data Loader
class MRIdata(Dataset):
    def __init__(self, path_to_data, train=True, transform=None):
        # self.data = pd.read_csv(path_to_data)
        self.data, self.missing_keys  = filter_irrelevant_data(path_to_data, BAD_DATA_LIST)
        self.labels = self.data['MGMT_value']
        self.transform = transform
        self.train = train
        self.train_dir = ("train" if train else "test")
        # self.train_dir = ("target1" if train else "test")
        self.MetaCache = cacheClass()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index in self.missing_keys:
          index +=1
        inp_data = read_dicom_img(self.train_dir, str(self.data['BraTS21ID'][index]),self.MetaCache)
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
class mResNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # self.args = args
        self.model = torchvision.models.resnet152(pretrained=False)

        self.model.conv1 = nn.Conv2d(
            in_channels= NUM_SAMPLES, #args.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
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

def accuracy(output, targets):
    return np.round(output).eq(targets).sum().item()/ len(targets)

def do_train_val(is_train, epoch, loader, model, optimizer, criterion, stats_file, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model
    if is_train:
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    batchAcc = AverageMeter()

    start = time.time()
    for indx, data in enumerate(loader, 0):
        inputs, labels = data["features"].to(device), data["labels"].to(device)
        if is_train:
            optimizer.zero_grad()
        outputs = model(inputs).squeeze(1).to(device)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        delta_t = time.time() - start

        losses.update(loss.detach().item(), outputs.shape[0])
        outputs = torch.sigmoid(outputs).cpu().squeeze().detach()
        acc = accuracy(outputs, labels.cpu().detach())
        batchAcc.update(acc)

        if writer:
            if is_train:
                loss_tag = "Loss/train"
                acc_tag = "Accuracy/train"
            else:
                loss_tag = "Loss/val"
                acc_tag = "Accuracy/val"

            writer.add_scalar(
                loss_tag, loss.item(), indx + (epoch * len(loader))
            )
            writer.add_scalar(
                acc_tag, acc, indx + (epoch * len(loader))
            )

    return losses.avg, batchAcc.avg, delta_t

def update_tensorboard(is_train, writer, epoch, loss, acc):
    if is_train:
      loss_tag = "Loss/train"
      acc_tag = "Accuracy/train"
    else:
      loss_tag = "Loss/val"
      acc_tag = "Accuracy/val"

    writer.add_scalar(loss_tag, loss, epoch)
    writer.add_scalar(acc_tag, acc, epoch)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mResNet().to(device)

    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(net.parameters(), lr=LRate, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(net.parameters(), lr=LRate, momentum=0.9)

    # resume if weights file exists
    if Path(weight_file_name).is_file():
        ckpt = torch.load(weight_file_name, map_location="cpu")
        start_epoch = ckpt["epoch"]
        net.model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    #Data Loader
    transforms = T.Compose([T.ToTensor()])

    data = MRIdata(f"{main_path}/train_labels.csv", transform=transforms)


#    train_data_len = int(len(data) * TRAIN_SPLIT)
#    val_data_len = len(data) - train_data_len

#    train_set, val_set = torch.utils.data.random_split(data, [train_data_len, val_data_len])

    kfolds = KFold(KFOLD_NUM_SPLIT, shuffle=True, random_state=SEED)
    foldId = 0
    for train_idx, val_idx in kfolds.split(np.arange(len(data))):
      foldId += 1
      train_sampler = SubsetRandomSampler(train_idx)
      val_sampler = SubsetRandomSampler(val_idx)

      train_loader = DataLoader(
                      dataset = data,
                      shuffle=False,
                      batch_size=BATCH_SIZE,
                      pin_memory=True,
                      num_workers=NUM_WORKERS,
                      persistent_workers=True,
                      sampler = train_sampler
      )

      val_loader = DataLoader(
                      dataset = data,
                      shuffle=False,
                      batch_size=BATCH_SIZE,
                      pin_memory=True,
                      num_workers=NUM_WORKERS,
                      persistent_workers=True,
                      sampler = val_sampler
      )

      for epoch in range(start_epoch, EPOCHS):

          thisTrainbatchLoss, thisTrainbatchAcc, Train_delta_t = \
              do_train_val(True, epoch, train_loader, net, optimizer, criterion, stats_file)
          update_tensorboard(True, writer, epoch, thisTrainbatchLoss, thisTrainbatchAcc)

          thisValbatchLoss, thisValbatchAcc, Val_delta_t = \
              do_train_val(False, epoch, val_loader, net, None, criterion, stats_file)
          update_tensorboard(False, writer, epoch, thisValbatchLoss, thisValbatchAcc)

          stats = dict(
              Fold=foldId,
              Epoch=epoch,
              BatchLoss=[round(thisTrainbatchLoss, 4), round(thisValbatchLoss, 4)],
              BatchAcc=[round(thisTrainbatchAcc, 4), round(thisValbatchAcc, 4)],
              Time = [round(Train_delta_t), round(Val_delta_t)]
          )
          print(json.dumps(stats))
          # print(json.dumps(stats), file=stats_file)

          if thisTrainbatchLoss + thisValbatchLoss < prevLoss:
              prevLoss = thisTrainbatchLoss + thisValbatchLoss
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
