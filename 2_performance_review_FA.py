import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RAND_SEED = 42

if __name__ == "__main__":
    path_to_data = "data/junipersun/"

    # load meta data
    with open(os.path.join(path_to_data, "info.txt")) as json_file:
        meta_info = json.load(json_file)

    # load feature labels
    with open(os.path.join(path_to_data, "FA_data/feature_labels_train_3.0_0.1.txt"), 'r') as f:
        feature_labels = [line.rstrip('\n') for line in f]
    print(feature_labels)

    # load gestures labels
    y_g = np.load(os.path.join(path_to_data, "split/y_g_train.npy"))

    # make gesture colormap
    cmap = dict(zip(list(range(meta_info["num_gestures"] + 1)), "yrbg"))

    # define master dataframes
    master_DBI = pd.DataFrame(index=feature_labels)
    master_FLDI = pd.DataFrame(index=feature_labels)
    master_LDA_avg = pd.DataFrame(index=feature_labels)
    master_SVM_avg = pd.DataFrame(index=feature_labels)

    for wl in range(200, 3001, 400):
        wli = wl * 0.001
        for wi in range(100, 200, 200):
            wii = wi * 0.001

            col = "{}_{}".format(round(wli, 2), round(wii, 2))

            # load feature analysis results
            FA_results = pd.read_csv(os.path.join(
                path_to_data, "FA_results/feature_analysis_{}.csv".format(col)), header=0, index_col=0)

            # add result to master dataframes
            master_DBI.loc[:, col] = FA_results["DBI"]
            master_FLDI.loc[:, col] = FA_results["FLDI"]
            master_LDA_avg.loc[:, col] = FA_results["LDA_avg"]
            master_SVM_avg.loc[:, col] = FA_results["SVM_avg"]

            # load features
            features = np.load(os.path.join(
                path_to_data, "FA_data/features_train_{}_{}.npy".format(round(wli, 2), round(wii, 2))))

            # pretty gesture labeling
            n_windows = features.shape[1] / y_g.shape[0]
            y_g_w = np.repeat(y_g, n_windows)
            row_colors = [cmap[y] for y in np.sort(y_g_w)]

            for c in range(features.shape[-1]):
                features_c = np.squeeze(features[:, :, c]).T
                print("features_c.shape", features_c.shape)
                df_features_c = pd.DataFrame(
                    features_c, columns=feature_labels)
                df_features_c.loc[:, 'gesture'] = y_g_w
                df_features_c = df_features_c.sort_values(by='gesture')

                # plot feature correlation matrix
                plt.figure()
                plt.subplots(figsize=(20, 15))
                sns.heatmap(df_features_c.corr(), annot=False, cmap='coolwarm')
                plt.title(col + " , channel " + str(c))
                plt.savefig(os.path.join(
                    path_to_data, "FA_results/correlation_{}_c{}.png".format(col, c)))
                plt.close()

                # plot dendrogram of features
                sns.clustermap(df_features_c, row_cluster=False,
                               cmap='coolwarm', row_colors=row_colors, figsize=(20, 15))
                plt.savefig(os.path.join(
                    path_to_data, "FA_results/dendrogram_{}_c{}.png".format(col, c)))
                plt.close()

    plt.figure()
    plt.subplots(figsize=(20, 15))
    sns.heatmap(master_DBI, annot=True, cmap='coolwarm')
    plt.savefig(os.path.join(path_to_data, "FA_results/DBI_overall.png"))

    plt.figure()
    plt.subplots(figsize=(20, 15))
    sns.heatmap(master_FLDI, annot=True, cmap='coolwarm')
    plt.savefig(os.path.join(path_to_data, "FA_results/FLDI_overall.png"))

    plt.figure()
    plt.subplots(figsize=(20, 15))
    sns.heatmap(master_LDA_avg, annot=True, cmap='coolwarm')
    plt.savefig(os.path.join(path_to_data, "FA_results/LDA_avg_overall.png"))

    plt.figure()
    plt.subplots(figsize=(20, 15))
    sns.heatmap(master_SVM_avg, annot=True, cmap='coolwarm')
    plt.savefig(os.path.join(path_to_data, "FA_results/SVM_avg_overall.png"))
