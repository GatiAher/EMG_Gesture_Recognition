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

if __name__ == "__main__":
    # TODO: parse arguments
    path_to_data = "data/junipersun/"

    # load meta data
    with open(os.path.join(path_to_data, "info.txt")) as json_file:
        meta_info = json.load(json_file)

    # load train dataset, gestures labels, subject labels
    X = np.load(os.path.join(path_to_data, "split/X_train.npy"))
    y_g = np.load(os.path.join(path_to_data, "split/y_g_train.npy"))
    if (meta_info['num_subjects'] > 1):
        y_s = np.load(os.path.join(path_to_data, "split/y_s_train.npy"))

    # segment into overlapping windows, n = n x w analysis segments
    # perform feature extraction (n x t x c) -->  (f x (n x w) x c), (f,)
    # 200 ms with 100 ms overlap gives whole numbers
    features, feature_labels, n_windows = moving_window(
        X, meta_info["sampling_rate"], 0.2, 0.1)
    y_g_w = np.repeat(y_g, n_windows)

    # save f x DBI, FLDI, SVM, LDA, and RF to dataframe
    df = pd.DataFrame(columns=["DBI", "FLDI", "LDA_avg", "LDA_std", "SVM_avg", "SVM_std"],
                      index=feature_labels)

    # for each f, calculate davies bouldin index(DBI) of (n x w) x c matrix
    for i, f in enumerate(feature_labels):
        df.at[f, "DBI"] = davies_bouldin_score(features[i, :, :], y_g_w)

    # for each f, calculate fisher's linear discriminant index (FLDI) of (n x w) x c matrix
    N_classes = np.bincount(y_g_w)[1:]
    for i, f in enumerate(feature_labels):
        clf = LinearDiscriminantAnalysis(store_covariance=True)
        clf.fit(np.squeeze(features[i, :, :]), y_g_w)
        # get clf properties
        within_class_cov = clf.covariance_
        overall_mean = clf.xbar_
        class_means = clf.means_
        between_class_cov = np.zeros(
            (meta_info["num_gestures"], features.shape[-1]))
        for c in range(meta_info["num_gestures"]):
            between_class_cov += N_classes[c] * (class_means[c] -
                                                 overall_mean).dot((class_means[c] - overall_mean).T)
        # get fisher's ratio
        fishers_ratio = within_class_cov / between_class_cov
        df.at[f, "FLDI"] = np.sum(fishers_ratio)

    print(df)

    # for each f, classify (n x w) x c matrix with LDA using 10 fold cross-validation
    for i, f in enumerate(feature_labels):
        clf = LinearDiscriminantAnalysis()
        scores = cross_val_score(clf, features[i, :, :], y_g_w, cv=10)
        df.at[f, "LDA_avg"] = scores.mean()
        df.at[f, "LDA_std"] = scores.std()

    print(df)

    # # for each f, classify (n x w) x c matrix with SVM using 10 fold cross-validation
    # for i, f in enumerate(feature_labels):
    #     print("on feature", f)
    #     clf = svm.SVC(kernel='linear', C=1, random_state=RAND_SEED)
    #     scores = cross_val_score(
    #         clf, features[i, :, :], y_g_w, cv=3, n_jobs=-1)
    #     df.at[f, "SVM_avg"] = scores.mean()
    #     df.at[f, "SVM_std"] = scores.std()

    print(df)
    df.to_csv(os.path.join(path_to_data, "feature_analysis.csv"))
