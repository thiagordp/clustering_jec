import pickle


with open('clustering_hierarchical.pkl', 'rb') as f:
    data = pickle.load(f)    
    open("clustering_hierarchical.txt", "w+").write(str(data))

with open('clustering_kmeans.pkl', 'rb') as f:
    data = pickle.load(f)
    open("clustering_kmeans.txt", "w+").write(str(data))

with open('bag_of_words.pkl', 'rb') as f:
    data = pickle.load(f)
    open("bag_of_words.txt", "w+").write(str(data))