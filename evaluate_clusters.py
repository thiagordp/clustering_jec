"""
Code for Clusters evaluation metrics

@authors Sabo et. al
"""
import glob
import logging
import random
import re
from math import isnan

import numpy as np
import pandas as pd
import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


def calculate_entropy():
    pass


def calculate_impurity():
    pass


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
    bow_df = pd.read_csv("data/bow_model.csv", index_col=0)

    hopkins_value = round(hopkins(bow_df.to_dense()), 4)
    logging.info("Hopkins test: " + str(hopkins_value))

    return hopkins_value


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
