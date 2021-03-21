"""

"""
import glob
import os
import sys
import time
import numpy as np
import pandas as pd
import tqdm
import logging
import nltk
from sklearn.metrics.pairwise import cosine_similarity

from affinity_propagation_clustering import affinity_clustering
from evaluate_clusters import load_process_files, save_bow_to_csv, calculate_cluster_tendency

DATASET_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/jec_base_665/TXTs/"
# DATASET_PATH = "/media/trdp/Arquivos/Studies/Msc/Thesis/Experiments/Datasets/processos_transp_aereo/txts_atualizados_sd_manual/novos/"
REPLACE_CHARS = "°-.:/,;()[]{}º\xa0§“”'‘’ª&$#\"@!?+-~£\x81<>"


def interpret_groups(file_name, output_path):
    data = ""
    with open(file_name) as fp:
        data = fp.read()

    data = data.replace("[] {", "").replace("[", "").replace("]", "").replace("},", "").replace("}", "").replace(",", "")

    splits = sorted(data.split("\n"))

    dict_clusters = {}
    for sp in splits:
        line = sp.strip()

        tokens = line.split()

        cluster = tokens[1]
        doc = int(tokens[0])

        if cluster in dict_clusters.keys():
            dict_clusters[cluster].append(doc)
        else:
            dict_clusters[cluster] = [doc]

    final_data = []
    for key in dict_clusters.keys():
        array_docs = dict_clusters[key]
        final_data.append([key, len(array_docs), str(array_docs).replace("[", "").replace("]", "")])

        # Create folder:
        path_to_cluster = output_path + key + "/"
        if not os.path.exists(path_to_cluster):
            os.makedirs(path_to_cluster)

        for doc in array_docs:
            source_path = DATASET_PATH + str(doc) + ".txt"
            dest_path = output_path + key + "/" + str(doc) + ".txt"
            os.system("cp " + source_path + " " + dest_path)

    df = pd.DataFrame(final_data, columns=["cluster", "Nº Docs", "Lista Docs"])
    df.to_excel(output_path + "list_clusters.xlsx")


def calculate_sequences():
    list_files = glob.glob(DATASET_PATH + "*.txt")
    numbers = sorted([int(num.split("/")[-1].replace(".txt", "")) for num in list_files])

    index = 1
    for i in range(len(numbers)):
        if index != numbers[i]:
            print(i, numbers[i])

            index = numbers[i] + 1
        else:
            index += 1


def token_count():
    docs = glob.glob(DATASET_PATH + "*")

    total_tokens = 0
    len_docs = len(docs)
    vocab = {}
    tokens_doc = []

    for doc_path in tqdm.tqdm(docs):

        with open(doc_path, encoding="utf-8") as fp:
            text = fp.read()

        text = text.lower()

        for replace_char in REPLACE_CHARS:
            text = text.replace(replace_char, " ")

        tokens = text.split()

        total_tokens += len(tokens)
        tokens_doc.append(int(len(tokens)))

        for token in tokens:
            if token not in vocab.keys():
                vocab[token] = 1
            else:
                vocab[token] += 1

    logging.info("Total Tokens:" + str(total_tokens))
    print("Total Docs:  ", len_docs)
    print("Tokens/Doc:  ", round(total_tokens / len_docs, 0))
    print("Tokens/doc 2:", int(round(np.mean(tokens_doc), 0)))
    print("STD 2:       ", int(round(np.std(tokens_doc), 0)))
    print("Vocab size:  ", len(vocab.keys()))
    # print("Vocab:", vocab)


def evaluate_cluster_results():
    # data, vocab = load_process_files()
    # save_bow_to_csv(data, vocab)

    for i in range(100):
        h = calculate_cluster_tendency()


def main():

    logging.basicConfig(format='%(asctime)s | %(levelname)s |\t%(message)s ',
                        datefmt='%d-%b-%y %H:%M:%S',
                        level=logging.INFO)

    nltk.download('stopwords')
    # interpret_groups("data/clustering_hierarchical.txt", "data/hierarchical/")
    # interpret_groups("data/clustering_kmeans.txt", "data/k_means/")
    # token_count()
    # calculate_sequences()
    # evaluate_cluster_results()

    _, docs, _, bow_docs, vocab = load_process_files()
    affinity_clustering(docs, bow_docs, vocab)


if __name__ == "__main__":
    main()
