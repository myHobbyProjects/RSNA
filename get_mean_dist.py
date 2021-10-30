import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# basePath = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
#basePath = "/Volumes/disk2s2/kaggle/shorts/train"
basePath = "/home/rudrajit_sengupta/kaggle/dataset/target"
# basePath = "./testdir"

if __name__ == "__main__":
    TotalFilecount = 0
    badFiles = 0

    for root, dirs, files in os.walk(basePath, topdown=False):
        for name in files:
            print("fname {0}".format(name))
            TotalFilecount += 1
            fname = os.path.join(root, name)
            img = pydicom.dcmread(fname)
            if np.mean(img.pixel_array) == 0:
                badFiles += 1

    print("TotalFilecount {0} BadFileCount {1}".format(TotalFilecount, badFiles))