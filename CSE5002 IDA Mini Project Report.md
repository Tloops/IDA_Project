# CSE5002 IDA Mini Project Report

> **Name**: 聂秋实
>
> **SID**: *my personal SID is concealed*



## 1 Introduction

The project provides a dataset of attributed social network at MIT, and the task is to train a classifier to predict missing labels of "class year". The task provides 5298 people with their social network connections and their basic attributes. 4000 people for training, 1298 for testing.

Both sources of information are important:

- social network topology
- node attributes

So in this project, I utilized both of them, represent them as a feature vector, and train models using such feature vectors. Detailed experiments and some discussions are also provided. The main language for this project is `python`, due to its strong machine learning packages. The code will be available at [https://github.com/Tloops/IDA_Project](https://github.com/Tloops/IDA_Project) after the deadline.



## 2 Method

### 2.1 Data processing

#### 2.1.1 Graph topology feature extraction

For graph topology, we need to construct feature embeddings nodes and edges. There are many works that are suitable for this job. Among them, **Node2vec** is chosen.

**Node2vec** hopes that the embedding can preserve the information in the graph, such as finding its neighbors in the graph through the embedding vector of a certain point. There was already a popular method in NLP -- **SkipGram**, which uses a sliding window to sample sentences, so that the probability of the central word and the words in the sliding window appearing at the same time is high, while the probability of the central word and the words outside the sliding window appearing at the same time is small. This ensures that the model can learn the surrounding words as information. However, the input of **SkipGram** is a sequence of word, so we need to interprete the graph data into sequences of nodes in order to generate embeddings for nodes. As an improvement for **Random Walk**, it is proposed to use a search parameter $\alpha$ to control the tendency of random walk, which is determined by the two values of $p$ and $q$:
$$
\alpha_{pq}(t,x)=\begin{cases}
\frac1p &\text{if }d_{tx}=0\\
1 &\text{if }d_{tx}=1\\
\frac1q &\text{if }d_{tx}=2
\end{cases}
$$
For implementation, python provides an offical package `node2vec`. It implemented the basic functions for **Node2vec**. The embedding has values range between $[-1, 1]$.



#### 2.1.2 Attribute feature preprocessing

As the attribute is provided as 'raw' data of the `node_id, degree, gender, major, second_major, dormitory, high_school`, so we need to normalize the data into $[-1, 1]$, too. For implementation, I use the `MinMaxScaler` the scale the attributes into distribution of range $[-1, 1]$.



### 2.2 Classification Models

After the features are generated and processed, the features can be sent into common classifiers. In this project, I use SVM, decision tree, random forest, bagging, KNN and Naive Bayes.



## 3 Experiments

### 3.1 Embedding Generating

The node embedding is generated using `emb_generate.py`. The graph is represented using `networkx`. To decide the dimension of the embedding should be, we test the performance of different dimension of embedding using the performance of SVC. The performance of them is shown in the chart below.

|              |   32   |   64   |    128     |  256   |
| :----------: | :----: | :----: | :--------: | :----: |
| SVC accuracy | 0.8320 | 0.8505 | **0.8513** | 0.8482 |

As the embedding with dimension 128 has the best accuracy, we will use **128** to test other classifiers.



### 3.2 Performance Results

I have tested 6 common classifiers for this project:

- SVC: default setting
- Decision Tree: default setting
- Random Forest: `n_estimators=200, max_features=X_train.shape[1]//2, max_depth=10`
- Bagging: default setting
- KNN: `n_neighbors=100`
- Naive Bayes: default setting

The accuracy (correctly classified / total samples) of different method in the test dataset is shown in the chart below.

|                           |    SVC     | Decision Tree | Random Forest |  Bagging   |    KNN     | Naive Bayes |
| :-----------------------: | :--------: | :-----------: | :-----------: | :--------: | :--------: | :---------: |
| w/ min-max normalization  | **0.8513** |  **0.5054**   |  **0.7458**   | **0.6633** | **0.7219** |   0.8005    |
| w/o min-max normalization |   0.1841   |  **0.5054**   |  **0.7458**   |   0.6625   |   0.1857   |   0.5470    |
|   Only graph embedding    |   0.8482   |    0.4368     |    0.7257     |   0.6233   |   0.7096   | **0.8143**  |
| Only attributes (w/ norm) |   0.3158   |    0.2804     |    0.3559     |   0.3182   |   0.2735   |   0.1710    |

The first two lines show the accuracy using both graph embedding and the attributes. We can see that it is important for SVC, KNN and Naive Bayes to do normalization, and it doesn't matter for Decision Tree and Random Forest. The third and fourth lines show the accuracy when we only use graph embedding and only use the attributes. We can see that for most of the classifiers, concatenation of the two features gives the best accuracy. SVC achieves the best accuracy, and the second is Naive Bayes.



## 4 Conclusion and Discussion

This project finds a way to predict class year using the social network and related attributes. The social network is transfered into embeddings and is concatenated with the attributes as the input of 6 common classifiers. By comparison, **SVC** outperforms the rest 5 classifiers and gets the best accuracy on test dataset.

However, there are still some limitations for this project.

1. There are many more advanced graph embedding methods (e.g. GNN), but it's not explored in this project.
2. In the data, there are many empty value marked as 0. Further improvement can be made to deal with these empty values.