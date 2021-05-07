"""
Perform feature evaluation by class separability and classification

Output: dataframe of feature evaluation

@author: Gati Aher
"""

import os
import json
import numpy as np
import pandas as pd

from extract_features import extract_features

from sklearn.metrics import davies_bouldin_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def moving_window(X, sampling_rate_Hz, window_span_sec, window_incr_sec, verbose=False):
    """
    Return a feature matrix of shape (f x (n x w) x c)

    Args:
        X: ndarray (n x t x c)

    Return
        features: (f x (n x w) x c)
        feature_labels: (f,)
        n_windows: (int) number of windows
    """
    # number of time points in window
    window_span = int(sampling_rate_Hz * window_span_sec)
    # number of time points to shift window by
    window_incr = int(sampling_rate_Hz * window_incr_sec)
    # number of analysis segments
    n_windows = int((X.shape[1] - window_span) // window_incr) + 1
    print("number of analysis segments:", n_windows)

    feature_labels = extract_features(None, only_return_labels=True)
    features = np.zeros(
        (len(feature_labels), X.shape[0] * n_windows, X.shape[2])
    )

    # sliding windows
    i = 0
    for e in range(window_span, X.shape[1], window_incr):
        s = e - window_incr
        fm, _ = extract_features(X[:, s:e, :])
        features[:, i:i+fm.shape[1], :] = fm
        if (verbose):
            print("\n----\n \t mean \t std ")
            for f, flabel in enumerate(feature_labels):
                for c in range(fm.shape[2]):
                    print(flabel + str(c), np.mean(features[f, i:i+X.shape[0], c]),
                          np.std(features[f, i:i+X.shape[0], c]))
        i += X.shape[0]

    # replace nans
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return features, feature_labels, n_windows


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
    df = pd.DataFrame(columns=["DBI", "FLDI", "SVM", "LDA", "RF"],
                      index=feature_labels)

    # for each f, calculate davies bouldin index (DBI) of (n x w) x c matrix
    for i, f in enumerate(feature_labels):
        df.at[f, "DBI"] = davies_bouldin_score(features[i, :, :], y_g_w)

    # for each f, calculate fisher's linear discriminant index (FLDI) of (n x w) x c matrix
    N_classes = np.bincount(y_g_w)
    print("N_classes", N_classes)
    for i, f in enumerate(feature_labels):
        clf = LinearDiscriminantAnalysis(store_covariance=True)
        clf.fit(np.squeeze(features[i, :, :]), y_g_w)
        # within_class_cov = clf.covariance_
        # print("within_class_cov.shape", within_class_cov.shape)
        # between_class_cov = np.zeros(
        #     (meta_info["num_gestures"], meta_info["num_gestures"])
        # )




        # TODO: load test dataset, labels_gestures, labels_subjects

        # TODO: for each f, classify with SVM, LDA, and RF

        # TODO: save f x DBI, FLDI, SVM, LDA, and RF to dataframe

    pass
