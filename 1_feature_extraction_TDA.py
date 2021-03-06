"""
Perform topological analysis where nodes are features

Output: dataframe of node information, html graphs of nodes

@author: Gati Aher

Learn About Applied Topological Data Analysis:
* Duve, Ryan. “Intro to Topological Data Analysis and Application to NLP Training Data for Financial Services.” Medium, Towards Data Science, 7 Apr. 2021, https://towardsdatascience.com/intro-to-topological-data-analysis-and-application-to-nlp-training-data-for-financial-services-719495a111a4
* https://kepler-mapper.scikit-tda.org/en/latest/notebooks/KeplerMapper-Newsgroup20-Pipeline.html

TDA applied to EMG data:
* Phinyomark Angkoon, Khushaba Rami N., Ibáñez-Marcelo Esther, Patania Alice, Scheme Erik and Petri Giovanni 2017 Navigating features: a topologically informed chart of electromyographic features spaceJ. R. Soc. Interface.142017073420170734 http://doi.org/10.1098/rsif.2017.0734
* Côté-Allard, Ulysse et al. “Interpreting Deep Learning Features for Myoelectric Control: A Comparison With Handcrafted Features.” Frontiers in bioengineering and biotechnology vol. 8 158. 3 Mar. 2020, http://doi:10.3389/fbioe.2020.00158
    * code: https://github.com/UlysseCoteAllard/sEMG_handCraftedVsLearnedFeatures
"""

import os
import json
import csv
import numpy as np

from extract_features import extract_features
import numpy as np


if __name__ == "__main__":
    # TODO: parse arguments
    path_to_data = "data/junipersun/"

    # load meta data
    with open(os.path.join(path_to_data, "info.txt")) as json_file:
        meta_info = json.load(json_file)

    # load train dataset, gestures labels, subject labels
    X = np.load(os.path.join(path_to_data, "split/X_train.npy"))

    # perform feature extraction (n x t x c) --> (f x n x c), (f,)
    features, feature_labels = extract_features(X, drop_constants=True)
    print("features.shape", features.shape)

    # perform feature scaling (per channel) so that each feature has zero mean and unit variance
    # TODO: on sets with multiple subjects, perform feature scaling per subject
    feature_mean = np.mean(features, axis=1).reshape(
        (features.shape[0], 1, -1))
    feature_std = np.std(features, axis=1).reshape((features.shape[0], 1, -1))
    features = np.subtract(features, feature_mean) / feature_std

    # save feature files
    feature_train_save = os.path.join(
        path_to_data, "TDA_data/features_train.npy")
    np.save(feature_train_save, features)
    print("saved features to ", feature_train_save,
          "shape", features.shape)

    feature_train_labels_save = os.path.join(
        path_to_data, "TDA_data/feature_labels_train.txt")
    with open(feature_train_labels_save, 'w', newline='') as f:
        for l in feature_labels:
            f.write(l)
            f.write('\n')
    print("saved feature labels to ",
          feature_train_labels_save, "len", len(feature_labels))
