"""
Perform feature evaluation by class separability and classification

Output: dataframe of feature evaluation

@author: Gati Aher
"""

import os
import json
import numpy as np
import pandas as pd

from extract_features import moving_window

from sklearn.metrics import davies_bouldin_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

from sklearn.model_selection import cross_val_score

RAND_SEED = 42


def perform_feature_extraction(X, sampling_rate, window_len, window_incr, path_to_data, section="train"):
    # segment into overlapping windows, n = n x w analysis segments
    # perform feature extraction (n x t x c) -->  (f x (n x w) x c), (f,)
    features, feature_labels, n_windows = moving_window(
        X, sampling_rate, window_len, window_incr)
    print("***n_windows: ", n_windows)

    # perform feature scaling (per channel) so that each feature has zero mean and unit variance
    # TODO: on sets with multiple subjects, perform feature scaling per subject
    feature_mean = np.mean(features, axis=1).reshape(
        (features.shape[0], 1, -1))
    feature_std = np.std(features, axis=1).reshape((features.shape[0], 1, -1))
    features = np.divide(np.subtract(features, feature_mean), feature_std)

    # save feature file
    features_save = os.path.join(
        path_to_data, "FA_data/features_{}_{}_{}.npy".format(section, round(window_len, 2), round(window_incr, 2)))
    np.save(features_save, features)
    print("saved features to ", features_save,
          "shape", features.shape)

    # save label file
    feature_labels_save = os.path.join(
        path_to_data, "FA_data/feature_labels_{}_{}_{}.txt".format(section, round(window_len, 2), round(window_incr, 2)))
    with open(feature_labels_save, 'w', newline='') as f:
        for l in feature_labels:
            f.write(l)
            f.write('\n')
    print("saved feature labels to ",
          feature_labels_save, "len", len(feature_labels))


if __name__ == "__main__":
    # TODO: parse arguments
    path_to_data = "data/junipersun/"

    # load meta data
    info_load = os.path.join(path_to_data, "info.txt")
    with open(info_load) as json_file:
        meta_info = json.load(json_file)

    # load train dataset
    X = np.load(os.path.join(path_to_data, "split/X_train.npy"))

    # load test dataset
    X_test = np.load(os.path.join(path_to_data, "split/X_test.npy"))

    # perform feature extraction, find best window size
    # window_len = 0.2  # seconds
    # window_incr = 0.1  # seconds

    for wl in range(3000, 3001, 400):
        wli = wl * 0.001
        for wi in range(100, 200, 200):
            wii = wi * 0.001
            print("perform feature extraction: train")
            perform_feature_extraction(
                X, meta_info['sampling_rate'], wli, wii, path_to_data)
            print("perform feature extraction: test")
            perform_feature_extraction(
                X_test, meta_info['sampling_rate'], wli, wii, path_to_data, "test")
