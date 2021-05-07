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
import numpy as np

from extract_features import extract_features

import kmapper as km
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler


def perform_TDA(data, labels, NR, PO, n_clusters, filt="knn_distance_2", save_string="TDA", title="TDA"):
    """
    Perform Topological Data Analysis

    Args:
        data: 2-dimensional array, where first dimension is kept as member label
        names: labels for first dimension of data
        NR: number of hypercubes
        PO: percent overlap between hypercubes
        nclusters: number of clusters in hypercube
        filt: filtering scheme, default is "knn_distance_2"
        save_string: path where to save html and json
        title: title of map graph
    """

    # Step 1. initiate a Mapper
    mapper = km.KeplerMapper(verbose=0)

    # Step 2. Projection
    projected_data = mapper.fit_transform(data, projection=filt)

    # Step 3. Covering, clustering & mapping
    graph = mapper.map(projected_data, data,
                       cover=km.Cover(n_cubes=NR, perc_overlap=PO),
                       clusterer=AgglomerativeClustering(n_clusters=n_clusters,
                                                         linkage="ward",
                                                         affinity="euclidean",
                                                         memory=None,
                                                         connectivity=None,
                                                         compute_full_tree="auto",
                                                         )
                       )
    with open(save_string + ".json", "w") as f:
        json.dump(graph, f)

    mapper.visualize(graph,
                     X_names=labels,
                     path_html=save_string + ".html",
                     title=title,
                     custom_tooltips=np.array(labels),
                     #  color_values=list(range(len(labels)))
                     )


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

    # perform feature extraction (n x t x c) --> (f x n x c), (f,)
    features, feature_labels = extract_features(X, drop_constants=True)

    # perform feature scaling (per channel) so that each feature has zero mean and unit variance
    # TODO: on sets with multiple subjects, perform feature scaling per subject
    feature_mean = np.mean(features, axis=1).reshape(
        (features.shape[0], 1, -1))
    feature_std = np.std(features, axis=1).reshape((features.shape[0], 1, -1))
    features_scaled = np.subtract(features, feature_mean) / feature_std

    # reshape array to be two-dimensional
    features_reshape = features_scaled.reshape((features.shape[0], -1))
    print("features_reshape.shape", features_reshape.shape)

    # TODO: on bigger sets, run PCA (f x n x c) --> (f x p)

    # run topographic analysis
    NR = 2
    PO = 0.75
    n_clusters = 3
    save_string = os.path.join(
        path_to_data, "TDA_{}_{}_{}".format(NR, PO, n_clusters))
    perform_TDA(features_reshape, feature_labels, NR, PO, n_clusters,
                save_string=save_string, title=save_string)
