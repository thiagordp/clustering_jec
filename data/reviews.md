# COMMENTS TO THE AUTHOR:

## Reviewer #1: 

The paper describes the results from a clustering approach on Brazilian legal judgments. The selected domain is consumer law (air transport service). The used clustering methods are standard methods that are reused using pre-implemented software packages. The evaluation was performed by comparing the clustering results with the input and judgment of a legal expert.


### Major remarks:
- The selection of clustering methods seems very arbitrary. Why those three? Why not additional ones? E.g. DBScan? (Including the intro at section 5.4.)
- Lingo creates 65 groups --> you mean clusters? What is the difference between cluster and group?
- "We considered that Lingo performed reasonably well because it identifies a part of the descriptions given by the legal expert and also descriptions not identified by the legal expert." --> I don't understand the scientific message of this statement.
- Section 2.2. starts with a very basic introduction into ML. Could be more dense? The paper is not supposed to give a basic introduction into ML.
- The dataset is very badly described. The authors don't share more information than a brief statement in Section 4 (enumeration 2): How does the data look like? How much tokens? How many tokens per document? Time distribution of the documents? Metadata? Did you consider making the dataset (and the software code) publicly available to reconstruct the experiment? etc.
- Section 5.2. : introducing BOW, including a fictional example on that very low-level is irrelevant for the audience
- Section 6. : very hard to understand the outcome of the clustering approaches. The authors compare the algorithms result with the expert clustering. Was it one legal expert that did the clustering? How about inter annotator agreement?
- Section 6: the evaluation seems to be a little bit arbitrary. To which standard method do you refer when it comes up to the evaluation? Clustering approaches are never easy to evaluate. Have you considered to add an additional more reliable metric, to harden the results (e.g. https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html)

### Minor remarks:

- "Lingo creates 65 groups described in total" --> rephrasing, not correct.
- "main partitional clustering model is K-means" --> Reference? Who says that K-means is the main method? By the way, this is irrelevant.
- Section 5.1. Garcia2015 -> incomplete reference
- Page 15: "but we considered it relevant to supervised learning as input" --> I don't understand this remark. How does supervised learning fit into the discussion?
- last sentence: "For the classification task, we will work in machine learning and deep learning models, especially Convolutional and Recurrent Neural Networks." --> The whole paper is about unsupervised learning, what would be a the classification task? very unclear and confusing to me.

### Conclusion:
The paper is an approach to generate clusters from Brazilian legal judgments (n = 665) using standard clustering methods. The paper is well-written but lacks a little bit of a clear focus. From a data science point of view the paper lacks of a clear description of the data set and some mathematical contribution of how the approach contributes to state-of-the-art. From a legal perspective the paper lacks of a grounded theory that significantly adds to legal interpretation or how the approach supports a use case the legal industry has. I would propose to rework the paper to focus more on the aspects that are relevant for legal experts and keep the data science part shorter (just make clear that you reuse existing implementations and standard methods).

------------------

## Reviewer #2: 
This article provides a case study for various clustering techniques on Brazilian legal judgments. Although the case study reviews hierarchical and K-mean, the principal analysis focuses on Lingo's ability to create meaningful clusters for a legal domain expert. For this specific use case, Lingo provided mixed results. While 14% of groups provided relevant and complete descriptions, 65% of the groups were irrelevant.

The article's main contribution provides a case study for analyzing legal judgments with unsupervised machine learning techniques. This work provides value because it offers insights into Lingo's usefulness in the legal domain. Although automated measures, like Topic Coherence, exist for topic understandability (Newman et al., 2010), these measures don't necessarily correlate to human judgments (Stevens et al., 2012). 

The authors did a great job of providing a detailed legal analysis of the clusters. The legal expert described the groupings for the hierarchical and K-mean techniques. The legal expert noted that the classification task was difficult for the K-Means technique because the documents were too varied. The authors then compared the manual description to the automated description provided by Lingo. 

Although the expert analysis is this article's main strength, a single legal expert's review is insufficient for a journal article because simular studies leverage multiple reviewers and or automated measures to make the performance analysis less subjective. For instance, Conrad et al. (2005) leveraged multiple annotators and automated measures such as entropy, purity, and f1-score. Kumar & Raguveer's (2012) study also leveraged entropy and purity measures. An entropy measure could have supported the expert's claim about the difficulty of producing the k-means group labels. 

If the authors intend to include automated measures, they should re-examine the comparison cluster algorithms. The article examines hard clustering (K-means) and soft clustering (Lingo). An apples-to-apples comparison is difficult because the quantitive metrics differ for clustering types (Conrad et al. 2005). Furthermore, K-means doesn't generate topics, so the Legal expert needed to annotate or label the cluster. In contrast, LDA, LSA, or NMF provide a better comparison because they perform soft clustering, generate topic descriptions, and allow for topic coherence measures (Stevens et al., 2012).

While the article lacks a complete comparison between clustering techniques, it failed to contextualize the results to other Lingo studies. Lingo's relevance score (18%) seemed low compared to Lingo's initial study results of 70-80% usefulness (Osinski, 2004). The results leave the reader to wonder if the experiment required additional legal-specific data prep because many clusters included dates as topics. 

-----------

Conrad, J. G., Al-Kofahi, K., Zhao, Y., & Karypis, G. (2005). Effective document clustering for large heterogeneous law firm collections. Proceedings of the 10th International Conference on Artificial Intelligence and Law - ICAIL '05, 177. https://doi.org/10.1145/1165485.1165513

Kumar, R., & Raghuveer, K. (2012). Legal Documents Clustering using Latent Dirichlet Allocation. International Journal of Applied Information Systems, 2(6), 27-33.

Newman, D., Noh, Y., Talley, E., Karimi, S., & Baldwin, T. (2010). Evaluating topic models for digital libraries. Proceedings of the 10th Annual Joint Conference on Digital Libraries - JCDL '10, 215. https://doi.org/10.1145/1816123.1816156

Osiński, S., Stefanowski, J., & Weiss, D. (2004). Lingo: Search Results Clustering Algorithm Based on Singular Value Decomposition. In M. A. Kłopotek, S. T. Wierzchoń, & K. Trojanowski (Eds.), Intelligent Information Processing and Web Mining (pp. 359-368). Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-540-39985-8_37

Stevens, K., Kegelmeyer, P., Andrzejewski, D., & Buttler, D. (2012). Exploring Topic Coherence over Many Models and Many Topics. Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, 952-961. https://www.aclweb.org/anthology/D12-1087

-----
