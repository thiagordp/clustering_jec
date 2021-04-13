# Clustering of Brazilian legal judgments about failures in air transport service: an evaluation of different approaches

Repository with the code used in the paper published in Artificial Intelligence and Law.

## Install requiriments

To install dependencies from this project, run:

    pip install -r requirements.txt

## Project Structure

- **Cluster Analysis (`cluster_analysis.py`):** Main file to run Affinity clustering, the quantitative analysis, evaluate clusters.
- **Affinity Clustering (`affinity_propagation_clustering.py`):** Algorithm implementation
- **Cluster evaluation (`evaluate_clusters.py`):** Script to calculate Entropy, Clustering Tendency and Purity.
- **Orange Pipeline (`data/ProcessAnalysis.ows`)**: File with the pipeline to run K-means and Hierarchical Clustering
