"""
Acquire JuniperSun Dataset and split it into train, test, and validation parts.
Save dataset as ndarrays.

* save raw data to `data/junipersun/raw/allEMGdata-JuniperSun-0323new.mat`
* save meta info to `data/junipersun/info.txt`
* save split data to `data/junipersun/split/` + filename

@author: Gati Aher
"""

from scipy.io import loadmat
import gdown

import numpy as np
import os
import json

from sklearn.model_selection import train_test_split

RAW_SAVE = "data/junipersun/raw/allEMGdata-JuniperSun-0323new.mat"
INFO_SAVE = "data/junipersun/info.txt"
SPLIT_SAVE = "data/junipersun/split/"

RAND_SEED = 42
np.random.seed(RAND_SEED)

if __name__ == "__main__":

    # aquire dataset by downloading mat file from google drive
    urlname = 'https://drive.google.com/uc?id=1S4R8OsJJK3F_YbXecFfSdviNY5Ro0geY'
    gdown.download(urlname, RAW_SAVE, False)

    # load dataset
    matfile = loadmat(RAW_SAVE, struct_as_record=True)
    EMG = matfile['EMG']

    # get and save meta information
    meta_info = {
        "sampling_rate": float(EMG['srate'][0, 0][0, 0]),
        "num_subjects": 1,
        "num_gestures": 3
    }
    with open(INFO_SAVE, "w") as f:
        json.dump(meta_info, f)

    # format data, put in order (n x t x c)
    X = np.swapaxes(EMG['data'][0, 0].astype(np.float32), 0, -1)

    # get labels for gestures
    epochlabels = np.moveaxis(EMG['epochlabels'][0, 0], -1, 0)
    labels_gestures = np.zeros(epochlabels.shape, dtype=int)
    gesture_map = {
        'none': 0,
        "['rock']": 1,
        "['paper']": 2,
        "['scissors']": 3
    }
    for i, v in enumerate(epochlabels):
        labels_gestures[i] = gesture_map[str(v[0])]

    # split dataset, stratify so each split has same number of gestures (do not have to stratify by person for this dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels_gestures, stratify=labels_gestures, test_size=30, random_state=RAND_SEED)

    # mean center the data, by channel
    X_train_mean = np.mean(X_train, axis=(0, 1)).reshape((1, 1, -1))
    X_train = np.subtract(X_train, X_train_mean)
    X_test = np.subtract(X_test, X_train_mean)

    # verify data has right dimensions
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("train: # of each gesture", np.bincount(y_train[:, 0]))
    print()
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
    print("test: # of each gesture", np.bincount(y_test[:, 0]))
    print()

    # save dataset
    np.save(os.path.join(SPLIT_SAVE, "X_train.npy"), X_train)
    np.save(os.path.join(SPLIT_SAVE, "X_test.npy"), X_test)

    np.save(os.path.join(SPLIT_SAVE, "y_g_train.npy"), y_train)
    np.save(os.path.join(SPLIT_SAVE, "y_g_test.npy"), y_test)
