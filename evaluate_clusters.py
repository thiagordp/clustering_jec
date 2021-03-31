"""
Code for Clusters evaluation metrics

@authors Sabo et. al
"""
import glob
import json
import logging
import math
import random
import re
from math import isnan

import coloredlogs

coloredlogs.install()

import numpy as np
import pandas as pd
import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def load_process_files():
    # Get list of files
    logging.info("Load process files")

    file_list = glob.glob("/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/jec_base_665/TXTs/*")

    logging.info("Reading Files")
    texts = {}

    # Load text from files
    for file_path in file_list:
        with open(file_path, encoding="utf-8", mode="r") as fp:
            text = fp.read()

        file_num = file_path.split("/")[-1].replace(".txt", "")
        texts[int(file_num)] = text

    # Show keys
    keys = list(texts.keys())
    logging.info("Files: " + str(len(keys)))
    # logging.info(sorted(keys))

    # Process files
    logging.info("Processing files")
    for key_doc in tqdm.tqdm(texts.keys()):
        texts[key_doc] = process_text(texts[key_doc])

    logging.info("Building BOW Model")
    vectorizer = CountVectorizer(ngram_range=(1, 2))  # , min_df=0.1, max_df=0.9)
    bow_docs = vectorizer.fit_transform(texts.values())

    logging.info("Vocabulary:\t" + str(len(list(vectorizer.vocabulary_.keys()))))

    ndarray = bow_docs.toarray()
    listOflist = ndarray.tolist()

    dict_bow = {}
    for i, key in zip(range(len(texts.keys())), texts.keys()):
        dict_bow[key] = list(listOflist[i])

    logging.info("Finished loading files")

    return dict_bow, keys, vectorizer, listOflist, list(vectorizer.vocabulary_.keys())


def process_text(text):
    # Lowercase
    text = text.lower()

    # Remove URLS
    text = re.sub(r'^https?:\/\/.*[\r\n]*', "", text)

    # Extract Alfanumberic Tokens
    tokens = re.findall(r'\w+', text)

    # Remove Stopwords
    list_stopwords = stopwords.words("portuguese")
    tokens = [word for word in tokens if word not in list_stopwords]

    # Stemming
    snow_stemmer = PorterStemmer()
    tokens = [snow_stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


def calculate_entropy(clustering_alg, clusters_docs_dict):
    """
    Calculate Entropy from JSON files with expert notes

    Ref:
    :param clustering_alg: Clustering Algorithm
    :param clusters_docs_dict: Dict from JSON with all results
    :return: Entropy Values
    """

    logging.info("#" * 50)
    logging.info("Alg " + clustering_alg)

    clusters_dict_list = clusters_docs_dict[clustering_alg]
    list_docs = []

    n_docs = 0

    h_omega = 0
    h_w_list = []
    n_w_list = []

    for cluster_dict in clusters_dict_list:
        #logging.info("-" * 50)
        #logging.info("Cluster: " + cluster_dict["id"] + ": " + str(cluster_dict["num_docs"]))
        topics_list = cluster_dict["topics"]

        h_w = 0.0

        # Get total of docs in cluster
        n_w = np.sum([topic_dict["num_docs"] for topic_dict in topics_list])

        n_docs += n_w

        num_topics = len(topics_list)

        if num_topics == 1 and cluster_dict["num_docs"] > len(topics_list[0]):
            topic_diff = cluster_dict["num_docs"] - len(topics_list[0])
            n_w += topic_diff

            p_wc = (1.0 * topic_diff) / n_w
            h_w += -1.0 * p_wc * math.log2(p_wc)

        # if num_topics = 0... entropy is zero... and we do not need to sum....

        for topic_dict in topics_list:
            if n_w > 0:
                p_wc = (1.0 * topic_dict["num_docs"]) / n_w

                if p_wc > 0:
                    h_w += -1.0 * p_wc * math.log2(p_wc)

        h_w_list.append(h_w)
        n_w_list.append(n_w)

        #logging.info(round(h_w, 4))

    h_big_omega = 0
    #logging.info("Nw " + str(n_w_list))
    #logging.info("hw " + str(h_w_list))
    n_docs = np.sum(n_w_list)

    for i in range(len(h_w_list)):
        h_big_omega += h_w_list[i] * (n_w_list[i] / n_docs)

    logging.info("H(Ω)= " + str(round(h_big_omega, 4)))

    return 0


def calculate_impurity(clustering_alg, clusters_docs_dict):
    """
     Calculate Purity from JSON files with expert notes

     Ref:
     :param clustering_alg: Clustering Algorithm
     :param clusters_docs_dict: Dict from JSON with all results
     :return: Entropy Values
     """

    logging.info("#" * 50)
    logging.info("Alg " + clustering_alg)

    clusters_dict_list = clusters_docs_dict[clustering_alg]
    list_docs = []

    n_docs = 0

    purity_list = []
    n_w_list = []

    for cluster_dict in clusters_dict_list:
        #logging.info("-" * 50)
        #logging.info("Cluster: " + cluster_dict["id"] + ": " + str(cluster_dict["num_docs"]))
        topics_list = cluster_dict["topics"]

        purity_cw = 0.0

        # Get total of docs in cluster
        n_cluster = np.sum([topic_dict["num_docs"] for topic_dict in topics_list])

        n_docs += n_cluster

        num_topics = len(topics_list)

        # If there is no topics indicated... All docs from cluster belongs the to same topic.
        if num_topics == 0:
            purity_cw = 1

        # One there's only one topic, it is the "different" topic.
        # So we need to count the "normal" topic.
        elif num_topics == 1 and cluster_dict["num_docs"] > len(topics_list[0]):
            n_different = len(topics_list[0])
            n_normal = cluster_dict["num_docs"] - len(topics_list[0])

            n_docs += n_normal
            n_cluster += n_normal

            purity_cw = max(n_different, n_normal) / cluster_dict["num_docs"]
        else:
            num_docs_per_topic = [topic_dict["num_docs"] for topic_dict in topics_list]

            max_num = max(num_docs_per_topic)
            purity_cw = max_num / np.sum(num_docs_per_topic)

        # Get the max value from topics.
        purity_list.append(purity_cw)
        n_w_list.append(n_cluster)

        # logging.info(round(h_w, 4))

    purity = 0
    for i_cluster in range(len(purity_list)):
        n_i = n_w_list[i_cluster]

        purity += (1.0 * purity_list[i_cluster] * n_i) / n_docs

    logging.info("Purity = " + str(round(purity, 4)))

    return 0


def save_bow_to_csv(data, vocab):
    logging.info("BOW to CSV")

    data_list = []

    logging.info("Converting to data array")
    for key in data.keys():
        data_row = [key]
        data_row.extend(data[key])
        data_list.append(data_row)

    logging.info("Creating Dataframe")
    cols = ["doc"]
    cols.extend(vocab)
    df = pd.DataFrame(data_list, columns=cols)

    logging.info("Saving into CSV")
    df.to_csv("data/bow_model.csv", index=False)
    logging.info(df.describe())


def calculate_cluster_tendency():
    logging.info("-" * 100)
    logging.info("Cluster Tendency")

    logging.info("Reading CSV")
    bow_default_df = pd.read_csv("data/bow_model.csv", index_col=0)
    x = []

    for i in range(10):
        bow_df = bow_default_df.copy()
        hopkins_value = round(hopkins(bow_df), 4)
        logging.info("Hopkins test: " + str(hopkins_value))

    return 0


def hopkins(data_array):
    logging.info("Hopkins Test")

    d_cols = data_array.shape[1]  # Columns
    n_rows = len(data_array)  # rows
    m_samples = int(0.1 * n_rows)  # heuristic from article [1]
    logging.info("Data has " + str(n_rows) + " rows and " + str(d_cols) + " columns")
    logging.info("Calculating Hopkins for " + str(m_samples) + " samples")

    nbrs = NearestNeighbors(n_neighbors=1).fit(data_array.values)

    rand_X = random.sample(range(0, n_rows, 1), m_samples)

    mins_array, maxs_array = calculate_mins_max(data_array.values)

    random_arrays = generate_random(m_samples, d_cols, mins_array, maxs_array)

    ujd = []
    wjd = []

    for j in range(0, m_samples):
        # random_unif = np.random.uniform(np.amin(data_array, axis=0), np.amax(data_array, axis=0), d_cols)

        random_unif = random_arrays[j]
        u_dist, _ = nbrs.kneighbors(random_unif.reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(data_array.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H


def calculate_mins_max(data):
    logging.info("Calculating mins and maxs for each dimension in data")
    len_rows = len(data)
    len_col = len(data[0])

    min_values = [2 ** 32] * len_col
    max_values = [-2 ** 32] * len_col

    for i_row in tqdm.tqdm(range(len_rows)):
        row = data[i_row]

        for j_col in range(len_col):
            value_col = row[j_col]

            # Check the minimum value
            if value_col < min_values[j_col]:
                min_values[j_col] = value_col

            # Check the maximum value
            if value_col > max_values[j_col]:
                max_values[j_col] = value_col

    logging.info("Minimum values (20) array:")
    logging.info(min_values[0:20])
    logging.info("Maximum values (20) array:")
    logging.info(max_values[0:20])

    return min_values, max_values


def generate_random(n_samples, dim, mins_array, maxs_array):
    m_cols = len(mins_array)

    list_random_arrays = []

    for i_sample in range(n_samples):
        random_array = []

        for j_col in range(m_cols):

            min_value = mins_array[j_col]
            max_value = maxs_array[j_col]

            if min_value >= max_value:
                random_num = max_value
            else:
                random_num = np.random.uniform(
                    min_value,
                    max_value,
                    1
                )
            random_array.append(random_num)

        list_random_arrays.append(np.array(random_array))

    return list_random_arrays


def process_notes_from_expert():
    logging.info("-" * 50)
    logging.info("Process notes from Legal Expert")

    # Open and read expert analysis from raw text file.
    with open("data/clustering_evaluation/analise_clustering.txt", encoding="utf-8", mode="r") as fp:
        logging.info("Reading notes from expert")
        raw_text = fp.read()

    logging.info("Lines: " + str(len(raw_text)))

    clustering_algs = [
        "K-means",
        "Hierárquica",
        "Affinity"
    ]

    alg_dicts = dict()

    for alg in clustering_algs:
        raw_text = raw_text.replace(alg, "#" + alg)

    results_by_alg = raw_text.split("#")

    # Iterate over results from algorithms (First item is the header of the file)
    for ind_alg in range(len(results_by_alg[1:])):

        alg_result = results_by_alg[1:][ind_alg]

        if len(alg_result.strip()) < 5:
            continue

        list_results = list()
        # Split the data in lines
        lines = alg_result.split("\n")
        logging.info("-" * 10)
        logging.info(lines[0])

        # Process the data by lines
        cluster_log_started = False
        cluster_data = dict()
        for line in lines:

            tokens = line.split()
            line = " ".join(tokens)  # Merge the tokens using only spaces, removing chars like '\n' and '\t'

            if len(tokens) <= 1:
                continue

            regexp = re.compile(r'C[0-9]+')

            # Check if start from Cluster results
            if regexp.search(line):

                if len(cluster_data.keys()) > 0:
                    list_results.append(cluster_data)

                cluster_data = dict()

                logging.warning(line)

                regexp = re.findall(r'([0-9]+)', line)

                cluster_id = "C" + regexp[0]
                docs_cluster = int(regexp[1])

                logging.warning("Num docs in Cluster: ")
                logging.warning(cluster_id)
                logging.warning(docs_cluster)

                cluster_data["num_docs"] = int(docs_cluster)
                cluster_data["id"] = cluster_id
                cluster_data["topics"] = list()

            # Else if it's the data for the cluster.
            else:
                if line[0:10].find("*") >= 0:
                    continue

                line = line.replace("–", "-")
                line = line.replace("check-in", "check_in")

                splits = line.split(" - ")

                if len(splits) > 1:
                    topic_data = dict()

                    topic_description = splits[0].strip()
                    string_docs = splits[1].replace(" e ", ",").split(",")

                    docs = [doc.strip() for doc in string_docs if doc.strip() != ""]

                    logging.info(topic_description)

                    topic_data["topico"] = topic_description
                    if docs[0] == "Nenhum":
                        topic_data["docs"] = []
                    else:
                        topic_data["docs"] = docs

                    num_docs_topic = len(topic_data["docs"])
                    topic_data["num_docs"] = num_docs_topic
                    cluster_data["topics"].append(topic_data)

                    if ind_alg >= 1:  # Hierarchical
                        topic_data = dict()
                        topic_data["topico"] = "Padrão"
                        topic_data["docs"] = []
                        topic_data["num_docs"] = cluster_data["num_docs"] - num_docs_topic
                        cluster_data["topics"].append(topic_data)

                    logging.info(docs)
                else:
                    logging.error("Else: " + str(line))

        list_results.append(cluster_data)

        alg_dicts[clustering_algs[ind_alg]] = list_results

    with open("data/clustering_evaluation/results.json", "w+") as fp:
        json.dump(alg_dicts, indent=4, fp=fp, ensure_ascii=False)

    return alg_dicts
