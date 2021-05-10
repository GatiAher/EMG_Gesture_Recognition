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

from matplotlib import pyplot as plt

RAND_SEED = 42


def perform_feature_evaluation(features, feature_labels, y_g_w, num_gestures, save_to):
    # save f x DBI, FLDI, SVM, LDA, and RF to dataframe
    df = pd.DataFrame(columns=["DBI", "FLDI", "LDA_avg", "LDA_std", "LDA_max", "SVM_avg", "SVM_std", "SVM_max"],
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
        between_class_cov = np.zeros(within_class_cov.shape)
        for c in range(meta_info["num_gestures"]):
            between_class_cov += ((class_means[c] - overall_mean).T).dot(class_means[c] - overall_mean)
        # get fisher's ratio
        fishers_ratio = within_class_cov / between_class_cov
        df.at[f, "FLDI"] = np.sum(fishers_ratio)

    # for each f, classify (n x w) x c matrix with LDA using cross-validation
    for i, f in enumerate(feature_labels):
        clf = LinearDiscriminantAnalysis()
        scores = cross_val_score(clf, features[i, :, :], y_g_w, cv=5)
        df.at[f, "LDA_avg"] = scores.mean()
        df.at[f, "LDA_std"] = scores.std()
        df.at[f, "LDA_max"] = max(scores)

    # for each f, classify (n x w) x c matrix with SVM using cross-validation
    for i, f in enumerate(feature_labels):
        print("on feature", f)
        clf = svm.SVC(kernel='linear', decision_function_shape='ovo',
                      C=0.5, random_state=RAND_SEED)
        scores = cross_val_score(
            clf, features[i, :, :], y_g_w, cv=5, n_jobs=-1)
        df.at[f, "SVM_avg"] = scores.mean()
        df.at[f, "SVM_std"] = scores.std()
        df.at[f, "SVM_max"] = max(scores)

    print(df)
    df.to_csv(dataframe_save)


if __name__ == "__main__":
    # TODO: parse arguments
    path_to_data = "data/junipersun/"
    # more closely mimicing real-time behavior
    classify_segments_individually = False

    # load meta data
    with open(os.path.join(path_to_data, "info.txt")) as json_file:
        meta_info = json.load(json_file)

    # load gestures labels, subject labels
    y_g = np.load(os.path.join(path_to_data, "split/y_g_train.npy"))

    for wl in range(200, 3001, 400):
        wli = wl * 0.001
        for wi in range(100, 200, 100):
            wii = wi * 0.001

            # load features
            features = np.load(os.path.join(
                path_to_data, "FA_data/features_train_{}_{}.npy".format(round(wli, 2), round(wii, 2))))

            # load feature labels
            with open(os.path.join(path_to_data, "FA_data/feature_labels_train_{}_{}.txt".format(round(wli, 2), round(wii, 2))), 'r') as f:
                feature_labels = [line.rstrip('\n') for line in f]
            print(feature_labels)

            if (classify_segments_individually):
                # format gesture labels
                n_windows = features.shape[1] / y_g.shape[0]
                y_g_w = np.repeat(y_g, n_windows)
                # save dataframe to
                dataframe_save = os.path.join(
                    path_to_data, "FA_results/feature_analysis_individual_segments_{}_{}.csv".format(round(wli, 2), round(wii, 2)))
            else:
                # don't format gesture labels
                y_g_w = np.squeeze(y_g).tolist()
                # reshape features
                features = features.reshape(
                    (features.shape[0], y_g.shape[0], -1))
                print("reshaped features.shape:", features.shape)
                # save dataframe to
                dataframe_save = os.path.join(
                    path_to_data, "FA_results/feature_analysis_{}_{}.csv".format(round(wli, 2), round(wii, 2)))

            perform_feature_evaluation(
                features, feature_labels, y_g_w, meta_info["num_gestures"], dataframe_save)
