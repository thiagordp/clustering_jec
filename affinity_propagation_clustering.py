"""
Affinity Propagation Clustering

@date Mar 15, 2021
"""
import logging
import os
import shutil

import pandas as pd
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise

import matplotlib.pyplot as plt
from itertools import cycle


def affinity_clustering(docs, bow_tm, vocab):
    logging.info("=" * 100)
    logging.info("Affinity Clustering")
    col_name = "SimMatrix"

    df_bow = pd.DataFrame(bow_tm, columns=vocab)

    for i in range(1):
        try:
            logging.info("-" * 100)
            logging.info("Start clustering " + str(i))

            af = AffinityPropagation(
                verbose=True,
                copy=True,
                affinity="euclidean",
                random_state=42,
                max_iter=5000)

            af.fit_predict(df_bow)
            cluster_centers_indices = af.cluster_centers_indices_
            labels = af.labels_
            n_clusters_ = len(cluster_centers_indices)

            print('Estimated number of clusters: %d' % n_clusters_)
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df_bow, labels, metric='sqeuclidean'))

            # Save files in directories according to assigned clusters

            # Path to directory for copying the clusters and text files.
            output_path = "data/affinity/"
            # Path to folders where the original text files are.
            input_path = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/jec_base_665/TXTs/"

            for doc_name, doc_label in zip(docs, labels):
                doc_in_path = input_path + str(doc_name) + ".txt"
                doc_out_path = output_path + "C" + str(doc_label) + "/"

                if not os.path.exists(doc_out_path):
                    os.makedirs(doc_out_path)

                doc_out_path += str(doc_name) + ".txt"
                shutil.copy(doc_in_path, doc_out_path)

            # Plot docs and clusters using PCA.
            plt.close('all')
            plt.figure(1)
            plt.clf()

            # Apply PCA.
            pca = PCA(n_components=2)
            bow_pca = pca.fit_transform(bow_tm)

            plt.close('all')
            plt.figure(figsize=(10, 6))
            plt.clf()

            colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
            for k, col in zip(range(n_clusters_), colors):
                class_members = labels == k
                cluster_center = bow_pca[cluster_centers_indices[k]]
                plt.plot(bow_pca[class_members, 0], bow_pca[class_members, 1], col + '.')
                plt.plot(cluster_center[0], cluster_center[1], 'o', markersize=4)
                for x in bow_pca[class_members]:
                    plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.tight_layout()
            plt.savefig("data/fig_affinity_clustering.png", dpi=300)

        except Exception as e:
            print(e)
