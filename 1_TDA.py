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


def get_feature_group(feature_name):
    # signal amplitude and power
    SAP = ["ABF", "DAMV", "DASDV", "DLD", "DTM", "DVARV", "DV", "IEMG", "LD", "M2", "MMAV1",
           "MMAV2", "MAV", "MAX", "MHW", "MNP", "MTW", "RMS", "SM", "SSI", "TM", "TTP", "VAR", "V", "WL"]
    if feature_name in SAP:
        return 0

    # frequency information
    FI = ["FR", "MDF", "MNF", "SSC", "ZC"]
    if feature_name in FI:
        return 1

    # non-linear complexity
    NLC = ["SampEn(1)", "SampEn(2)", "SampEn(3)",
           "APEN", "WAMP", "BC", "KATZ", "MFL"]
    if feature_name in NLC:
        return 2

    # time-series modeling
    TSM = ["AR(2)", "AR(3)", "AR(4)", "AR(5)", "DAR(2)", "DAR(3)", "DAR(4)",
           "DAR(5)", "CC(2)", "CC(3)", "CC(4)", "CC(5)", "DCC", "DFA", "PSR", "SNR"]
    if feature_name in TSM:
        return 3

    # unique
    UNI = ["CE", "DPR", "HIST", "KURT", "MAVS", "OHM",
           "PKF", "PSDFD", "SKEW", "SMR", "TSPSD", "VCF", "VFD"]
    if feature_name in UNI:
        return 4

    # other
    return 5


def perform_TDA(data, labels, NR, PO, n_clusters, filt="knn_distance_2", save_string="TDA", title="TDA", color_values=None, color_function_name=None):
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
    mapper = km.KeplerMapper(verbose=2)

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

    if color_values is None or color_function_name is None:
        mapper.visualize(graph,
                         X_names=labels,
                         path_html=save_string + ".html",
                         title=title,
                         custom_tooltips=np.array(labels),
                         )
    else:
        mapper.visualize(graph,
                         X_names=labels,
                         path_html=save_string + ".html",
                         title=title,
                         custom_tooltips=np.array(labels),
                         color_function_name=color_function_name,
                         color_values=color_values,
                         node_color_function=['mean', 'std', 'median', 'max'],
                         )


if __name__ == "__main__":
    # TODO: parse arguments
    path_to_data = "data/junipersun/"

    # load meta data
    with open(os.path.join(path_to_data, "info.txt")) as json_file:
        meta_info = json.load(json_file)

    # load features
    features = np.load(os.path.join(
        path_to_data, "TDA_data/features_train.npy"))

    # reshape array to be two-dimensional
    features_reshape = features.reshape((features.shape[0], -1))
    print("features_reshape.shape", features_reshape.shape)

    with open(os.path.join(path_to_data, "TDA_data/feature_labels_train.txt"), 'r') as f:
        feature_labels = [line.rstrip('\n') for line in f]

    # gesture labels for each trial
    y_g = np.load(os.path.join(path_to_data, "split/y_g_train.npy"))

    print(feature_labels)

    # define color lens

    # color by feature class
    lens_feature_class = [get_feature_group(f) for f in feature_labels]
    lens_name_feature_class = "feature class"

    # rock by channel
    lens_avg_rock_ch0 = np.mean(features[:, np.where(y_g == 1)[1], 0], axis=1)
    lens_name_avg_rock_ch0 = "average feature value of rock samples ch0"
    lens_avg_rock_ch1 = np.mean(features[:, np.where(y_g == 1)[1], 1], axis=1)
    lens_name_avg_rock_ch1 = "average feature value of rock samples ch1"
    lens_avg_rock_ch2 = np.mean(features[:, np.where(y_g == 1)[1], 2], axis=1)
    lens_name_avg_rock_ch2 = "average feature value of rock samples ch2"

    # paper by channel
    lens_avg_paper_ch0 = np.mean(features[:, np.where(y_g == 2)[1], 0], axis=1)
    lens_name_avg_paper_ch0 = "average feature value of paper samples ch0"
    lens_avg_paper_ch1 = np.mean(features[:, np.where(y_g == 2)[1], 1], axis=1)
    lens_name_avg_paper_ch1 = "average feature value of paper samples ch1"
    lens_avg_paper_ch2 = np.mean(features[:, np.where(y_g == 2)[1], 2], axis=1)
    lens_name_avg_paper_ch2 = "average feature value of paper samples ch2"

    # scissor by channel
    lens_avg_scissor_ch0 = np.mean(
        features[:, np.where(y_g == 3)[1], 0], axis=1)
    lens_name_avg_scissor_ch0 = "average feature value of scissor samples ch0"
    lens_avg_scissor_ch1 = np.mean(
        features[:, np.where(y_g == 3)[1], 1], axis=1)
    lens_name_avg_scissor_ch1 = "average feature value of scissor samples ch1"
    lens_avg_scissor_ch2 = np.mean(
        features[:, np.where(y_g == 3)[1], 2], axis=1)
    lens_name_avg_scissor_ch2 = "average feature value of scissor samples ch2"

    color_values = np.c_[lens_feature_class,
                         lens_avg_rock_ch0,
                         lens_avg_paper_ch0,
                         lens_avg_scissor_ch0,
                         lens_avg_rock_ch1,
                         lens_avg_paper_ch1,
                         lens_avg_scissor_ch1,
                         lens_avg_rock_ch2,
                         lens_avg_paper_ch2,
                         lens_avg_scissor_ch2
                         ]
    color_function_name = [lens_name_feature_class,
                           lens_name_avg_rock_ch0,
                           lens_name_avg_paper_ch0,
                           lens_name_avg_scissor_ch0,
                           lens_name_avg_rock_ch1,
                           lens_name_avg_paper_ch1,
                           lens_name_avg_scissor_ch1,
                           lens_name_avg_rock_ch2,
                           lens_name_avg_paper_ch2,
                           lens_name_avg_scissor_ch2,
                           ]

    # run topographic analysis
    for nr in range(6, 9, 2):
        NR = nr
        for po in range(50, 101, 20):
            PO = po / 100.0
            for n in range(2, 5):
                n_clusters = n
                save_string = os.path.join(
                    path_to_data, "TDA_results/TDA_{}_{}_{}".format(NR, PO, n_clusters))
                perform_TDA(features_reshape, feature_labels, NR, PO, n_clusters,
                            save_string=save_string, title=save_string, color_values=color_values, color_function_name=color_function_name)
