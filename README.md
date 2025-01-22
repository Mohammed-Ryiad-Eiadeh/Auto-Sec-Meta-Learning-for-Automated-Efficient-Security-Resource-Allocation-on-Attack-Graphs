# Auto-Sec: Meta-Learning for Automated Efficient Security Resource Allocation on Attack Graphs

# Abstract
Cybersecurity threats are increasingly affecting interdependent systems. This research explores security decision-making in these complex environments and proposes a resource allocation framework managed by a single rational defender to enhance security. We employ attack graphs to model vulnerabilities and present four defense strategies that incorporate network analysis algorithms, including degree centrality, betweenness centrality, harmonic centrality, and TrustRank. The resource allocation methods utilize graph theories such as Adjacent Nodes and Markov Blanket. Each ranking algorithm is paired with these allocation strategies, demonstrating low sensitivity to simultaneous attacks, which aids in developing the performance matrix. We apply random walk node embedding with negative sampling and a gradient descent algorithm for graph-structural feature extraction. Our tabular dataset includes features and ground truth labels based on the optimal resource allocation method for each graph, which we use to train ten classifiers for evaluating unseen data and making predictions. Our framework is validated against two datasets of 100 real-world graphs and 1,000 randomly generated graphs, measuring security improvements against six established baselines, including ARGOSMART and the global best. The results show that our framework performs better compared to these benchmarks, with faster execution than naive methods, and enhances security decision-making across various scenarios, accounting for all attack paths for each entry-asset variation. We have made the full implementation of our resource allocation framework available to the research community for further exploration and development.

# Framework
This framework aims to optimize the security of interdependent systems using genetic algorithm (GA), graph embedding techniques, and classification models. It begins by passing an attack graph with multiple entry nodes and critical assets. GA identifies the most critical attack paths between each entry and asset by using path encoding, crossover, mutation, and a customized fitness function that evaluates asset losses and defense investments. These paths guide resource allocation using methods like TrustRank, Katz centrality, degree centrality, PageRank, and Markov blanket to efficiently allocate defense resources. Also, this framework uses graph embedding to enhance feature learning for resource allocation decisions. Random walks capture structural features of the attack graph, followed by negative sampling and stochastic gradient descent (SGD) to create low-dimensional embeddings for each node. These embeddings, combined with resource allocation methods, form a dataset used to train and evaluate meta-learning classification algorithms (e.g., logistic regression, SVM, and LeNet-5). The goal is to predict the most effective resource allocation strategy efficiently. Performance is evaluated through metrics such as accuracy, hit-at-$k$, and inference time, facilitating optimal security decision-making

![Screenshot (15)](https://github.com/user-attachments/assets/dd3b0d3c-3f75-4b59-966a-43b26dc82d93)

# Customized Fitness Function
$F_2(P) = \max_{P \in P_m} \big(\exp\big(-\sum_{(v_i,v_j)\in P} {x_{i,j}}\big) + Wf\sum_{v_m\in P} L_m\big).$
   
   $P$ is the given attack path.

   $P_m$ is a set of attack paths.

   $v_i,v_j$ are the nodes in $P$.

   $L_m$ is the loss corresponding to node $v_m$

   $Wf$ is the weight factor lies in [0,1]
   
This function accounts for the total asset loss that the system will lose if the attack is occured successfully.

# Our Contribution
- We introduce a meta-learning method for predicting efficient security resource allocation techniques to secure interconnected systems with interdependent assets. Our approach quantifies the improvements of different security resource allocation decisions.

- We implement a node-embedding approach that uses random walks, negative sampling, and gradient descent to capture the attack graph's structural features as meta-features. These meta-features are combined with our performance matrix (representing different allocation methods) to construct a dataset, which is used to train meta-learning classifiers for predicting the most efficient allocation methods.

- We benchmark the performance of seven different graph-theoretic resource allocation methods to counter various attack models aimed at compromising interdependent systems. We also apply six node-ranking algorithms to optimize resource allocation decisions based on the importance of critical assets.

- We assess the effectiveness of our defense strategies across two attack graph testbeds: a real-world testbed with 100 graphs and a synthetic testbed with 1000 graphs. We compare our results against six meta-learning baselines.

- We release our framework's source code to the research community.

# Datasets We Used In Our Work
For the evaluation of our approach, we use two sets of attack graphs representing different interdependent systems and network topologies:

1. **Test Bed 1**: This consists of 100 attack graphs, grouped into three categories:
    - **Category 1**: Four real-world interdependent systems:
        - DER.1 [13]
        - SCADA [12]
        - E-commerce [14]
        - VOIP [14]
    - **Category 2**: Two graph typologies, HG1 and HG2, from prior research [15]
    - **Category 3**: 94 datasets from the interactive scientific graph data repository [16]:
        - aves-sparrow-social-2009 (ASC2009)
        - aves-sparrow-lyon-flock-season3 (ASFS3)
        - aves-weaver-social-03 (AWS03)
        - aves-barn-swallow-non-physical (ABSNP)
    This repository features network data from leading US universities.

2. **Test Bed 2**: This consists of 1000 randomly generated attack graphs.

In both test beds, a node `v_i ∈ V` represents either an attack step or a critical asset, and all graphs are directed. **Table 1** below provides a detailed description of the ten datasets used in our experiments, including the number of nodes, edges, critical assets, vulnerability types, attack paths, risks, and graph types. The risk is defined as: $\prod_{(v_i,v_j)\in P} \exp(-{x_{i,j}})$

We acknowledge that using diverse graphs improves the meta-learning model's performance, as varied training data helps the model generalize better during test data selection.

# Dataset Summary Table

| **System** | **# Nodes** | **# Edges** | **# Critical Assets** | **$v_s$ / $v_m$** | **Vulnerability** | **Attack Path** | **Risk** | **Graph Type** |
|------------|-------------|-------------|-----------------------|-------------------|-------------------|-----------------|----------|----------------|
| SCADA [Hota et al., 2016] | 13 | 20 | 6 | 1 / 12 | Malicious code | [1,6,11,12] | 0.76 | Directed |
| DER.1 [Jauhar et al., 2015] | 22 | 32 | 6 | 9 / 12 | Man-in-the-middle | [9,10,11,12] | 0.09 | Directed |
| E-Commerce [Modelo et al., 2008] | 20 | 32 | 4 | 1 / 7 | SQL injection | [1,2,5,6,7] | 0.19 | Directed |
| VOIP [Modelo et al., 2008] | 22 | 35 | 4 | 1 / 6 | Cross-site scripting | [1,2,4,6] | 0.55 | Directed |
| HG1 [Zeng et al., 2019] | 7 | 10 | 2 | 1 / 7 | Undefined | [1,5,7] | 0.38 | Directed |
| HG2 [Zeng et al., 2019] | 15 | 22 | 5 | 2 / 7 | Undefined | [2,1,3,7] | 0.17 | Directed |
| ABSNP [Nr et al., 2015] | 17 | 122 | 6 | 1 / 4 | Undefined | [1,17,4] | 0.69 | Directed |
| ASFS3 [Nr et al., 2015] | 27 | 163 | 9 | 4 / 24 | Undefined | [4,23,11,25,16,24] | 0.59 | Directed |
| ASS2009 [Nr et al., 2015] | 31 | 211 | 9 | 9 / 2 | Undefined | [9,25,14,2] | 0.94 | Directed |
| AWS03 [Nr et al., 2015] | 42 | 152 | 15 | 21 / 25 | Undefined | [21,24,27,25] | 0.89 | Directed |

Note: all of these datasets are stored in the project directory and is called dynamically so no need to set up their paths.

# Parameter Configuration of Our Experiments

We start by outlining the key hyperparameters used in various components of our framework. For resource allocation, the parameters for the GA, which simulates concurrent attack paths, are as follows: maximum iterations ($M=50$), population size ($N=250$), crossover probability ($m_p=0.4$), mutation rate ($m_r=0.2$), selection ratio ($s_r=0.6$) and weight factor ($Wf=0.001$). The defender's security budget ($S=20$), and the maximum iterations for the PR algorithm ($Rank_{iter}=100$), with epsilon ($\epsilon=0.0001$), the same as for the TR algorithm. In the graph embedding stage, the number of hops ($H=100$), dimension size ($D=256$), number of epochs ($Epoch=100$), learning rate ($\eta=0.01$), and the number of nodes for negative sampling is $NS=|V_m|$. It is important to note that our framework are designed to be applicable to any security budget and various graph types with variable initial security investments. For the classification phase, the embedding results (set of features) is splitted into 60\% for training and 40\% for testing. All experiments have been conducted using Java language (JDK 17) on a gaming machine with an Intel® Core™ i7-8750H CPU @ 2.20GHz (12 CPUs) and 32GB RAM.

# Results
**Table 1. Comparison of Various Approaches and Baselines on the 100-Graph Dataset For ISAC, $k=9$ centroids are used. The 'LibLinear' and 'Multinomial NB' approaches achieve the highest accuracy, while 'Best-Based-Rank' and 'FM' ranked highest in average performance. Note that the Naïve allocation method took 6.142 seconds to process 40 graphs.**

| **Algorithm**      | **acc** | **hit-at-5** | **hit-at-10** | **Avg Performance** | **Test time (ms)** | **Times faster** |
|--------------------|---------|--------------|---------------|---------------------|--------------------|------------------|
| LeNet-5            | 0.25    | 0.65         | 0.90          | 40.87               | 63                 | 97.49206         |
| MLP                | 0.20    | 0.85         | 0.95          | 41.40               | **15**             | 409.46667        |
| FM                 | 0.23    | 0.80         | **1.00**      | **42.56**           | 41                 | 1490.46488       |
| SVM                | 0.25    | 0.68         | 0.90          | 41.11               | **15**             | 409.46667        |
| LR                 | 0.18    | **0.90**     | 0.95          | 40.94               | 16                 | 383.87500        |
| KNN                | 0.15    | 0.75         | 0.85          | 38.66               | 16                 | 383.87500        |
| Cart               | 0.20    | 0.65         | 0.88          | 40.38               | **15**             | 409.46667        |
| Liblinear          | **0.30**| 0.78         | 0.98          | 42.30               | 16                 | 383.87500        |
| XGBoost            | 0.25    | 0.80         | 0.90          | 40.33               | **15**             | 409.46667        |
| Multinomial NB     | **0.30**| 0.53         | 0.90          | 39.60               | 31                 | 198.12903        |
| ARGOSMART          | 0.20    | 0.70         | 0.90          | 39.97               | 18                 | 341.222          |
| Best-Based-Rank    | 0.23    | 0.80         | **1.00**      | **42.56**           | 16                 | 383.875          |
| Global Best        | 0.20    | 0.45         | 0.88          | 37.70               | 63                 | 97.492           |
| ISAC-Kmeans        | 0.15    | 0.45         | 0.90          | 38.57               | 285                | 21.551           |
| ISAC-Kmeans++      | 0.20    | 0.55         | 0.95          | 40.02               | 165                | 37.224           |
| ISAC-HDBSCAN       | 0.20    | 0.45         | 0.88          | 37.70               | 154                | 39.883           |

This Table shows the evaluation of ten classifiers and six baselines on a dataset of 100 graphs, using metrics like accuracy, hit-at-5, hit-at-10, and inference time. LibLinear and Multinomial NB achieve the highest accuracy (0.30), while FM and Best-Based-Rank lead in hit-at-10 (1.00). LR outperforms in hit-at-5 (0.90), and MLP, SVM, XGBoost, and CART achieve the fastest inference times (15 ms). Best-Based-Rank is the top-performing baseline, but several classifiers surpass it. The framework demonstrates superior accuracy, efficiency, and robustness, with significant speed-up over Naïve allocation methods. The speed-up is calculated as: Speed-up = (Naïve_time) / (Our_time).

**Table2. Comparison of Various Approaches and Baselines on the 1000-Graph Dataset. **Note**: For ISAC, \( k = 9 \) centroids are used. The 'Multinomial NB' approach achieves the highest accuracy, while 'Best-Based-Rank' ranks highest in average performance, followed by 'SVM' and 'KNN'. The Naïve allocation method took 194.9636 seconds to process 400 graphs.**

| **Algorithm**      | **acc** | **hit-at-5** | **hit-at-10** | **Avg Performance** | **Test time (ms)** | **Times faster** |
|---------------------|---------|--------------|---------------|----------------------|--------------------|------------------|
| LeNet-5            | 0.28    | 0.59         | 0.95          | 44.04               | 172                | 1133.51          |
| MLP                | 0.22    | 0.81         | 0.99          | 45.66               | 47                 | 4148.16          |
| FM                 | 0.28    | 0.59         | 0.95          | 44.05               | 94                 | 2074.08          |
| SVM                | 0.17    | **0.95**     | **1.00**      | 46.89               | 235                | 829.63           |
| LR                 | 0.28    | 0.63         | 0.99          | 45.83               | 905                | 215.43           |
| KNN                | 0.16    | **0.95**     | **1.00**      | 46.79               | 390                | 499.91           |
| Cart               | 0.25    | 0.79         | 0.98          | 45.67               | **16**             | 12185.23         |
| Liblinear          | 0.26    | 0.66         | 0.99          | 46.15               | 32                 | 6092.61          |
| XGBoost            | 0.19    | 0.70         | 0.99          | 45.71               | 32                 | 6092.61          |
| Multinomial NB     | **0.30**| 0.60         | 0.96          | 44.78               | 94                 | 2074.08          |
| ARGOSMART          | 0.19    | 0.79         | 0.99          | 45.78               | 6798               | 28.679           |
| Best-Based-Rank    | 0.17    | **0.95**     | **1.00**      | **47.03**           | 94                 | 2074.08          |
| Global Best        | 0.28    | 0.59         | 0.94          | 43.94               | 165                | 1181.598         |
| ISAC-Kmeans        | 0.15    | **0.95**     | **1.00**      | 46.54               | 479                | 407.022          |
| ISAC-Kmeans++      | 0.26    | 0.79         | 0.98          | 45.65               | 414                | 470.927          |
| ISAC-HDBSCAN       | 0.15    | 0.88         | 0.98          | 45.56               | 874                | 223.070          |

This table shows the evaluation of ten classifiers and six baselines on the synthetic graph dataset shows that Multinomial NB achieved the highest accuracy (0.30), while SVM, KNN, ISAC-Kmeans, and Best-Based-Rank excelled in hit-at-5 (0.95) and hit-at-10 (1.00). CART achieved the fastest inference time (16 ms), while ISAC-Kmeans, despite low accuracy (0.15), performed well in hit-at-5 and hit-at-10. Overall, the proposed framework outperformed the baselines.

**Table 3. Comparison of Execution Time Between Naive Approach and Our Method to Carry Out The Testset Time in Seconds**

| **Dataset**   | **Naive Approach** | **Our Method** | **Gain** |
|---------------|--------------------|----------------|----------|
| Real Graph    | 6.142             | 0.0243         | 0.996    |
| Synthetic     | 194.9636          | 0.2017         | 0.999    |

This table shows that our method significantly outperforms the naive approach in execution time. For the Real Graph dataset, the naive method took 6.142 seconds, while our method reduced it to 0.0243 seconds (99.6% faster). For the Synthetic dataset, the time was reduced from 194.9636 seconds to 0.2017 seconds (99.9% faster), highlighting the efficiency of our approach.

# How to Run the Code (Please Read Carefully)

1) Download the latest version of IntelliJ IDEA.
2) Install JDK 17 or higher.
3) Set up the environment variable for the JDK 17+ bin folder.
4) Open IntelliJ IDEA.
5) Open the project in IDEA.
6) Ensure you have an active internet connection.
7) Wait for IDEA to download all dependencies specified in the pom.xml file.
8) Follow these steps to execute the code:
   - **Step 1**: Start with 'ResourceAllocationMain' and run it. Select the data folder you want to process.
   - **Step 2**: Next, run 'ExtractTopAllocationAsLabels'.
   - **Step 3**: Then, execute 'MainClassEmbedding'.
   - **Step 4**: Take the first column (allocation methods) from the 'Top Allocation Methods' file and add it to the 'Embedding' file as the class columns.
   - **Step 5**: Run 'SplitData' to generate the training and test datasets for classification.
   - **Step 6**: Execute 'ClassificationMain' to run the classification code.
   - **Step 7**: After preparing the train-test portions for ISAC, change the class labels to their numerical IDs as defined in the 'mapLabelToResourceAllocation' hashmap. Then, run the file.
   - **Step 8**: Now everything is ready to execute the baselines.

**Note: Make sure to configure the desired parameters for each file before executing.**

# Conclusion
This study proposes an efficient resource allocation strategy for securing interdependent systems controlled by a rational defender. By modeling these systems with cyber attack graphs, the research introduces 27 resource allocation methods and node ranking algorithms to guide decision-making within a limited budget, enhancing system security. The framework, integrating graph embedding techniques, outperforms leading methods like global best and ARGOSMART. Using random walk-based node embeddings, it effectively predicts optimal resource allocation strategies, achieving faster execution times and better security improvements on two datasets. Overall, the framework offers a robust solution for decision-making in complex systems, strengthening cybersecurity defenses.

# References

[12] A. R. Hota, A. Clements, S. Sundaram, and S. Bagchi. 2016. Optimal and game-theoretic deployment of security investments in interdependent assets. In International Conference on Decision and Game Theory for Security. 101–113.

[13] S. Jauhar, B. Chen, W. G. Temple, X. Dong, Z. Kalbarczyk, W. H. Sanders, and D. M. Nicol. 2015. Model-based cybersecurity assessment with nescor smart grid failure scenarios. In Dependable Computing (PRDC), 2015 IEEE 21st Pacific Rim International Symposium on. IEEE, 319–324.

[14] G. Modelo-Howard, S. Bagchi, and G. Lebanon. 2008. Determining placement of intrusion detectors for a distributed application through bayesian network modeling. In International Workshop on Recent Advances in Intrusion Detection. Springer, 271–290.

[15] J. Zeng, S. Wu, Y. Chen, R. Zeng, and C. Wu, ‘‘Survey of Attack Graph Analysis Methods from the Perspective of Data and Knowledge Processing,’’ Security and Communication Networks, vol. 2019, 2019.

[16] R. A. Rossi and N. K. Ahmed, ‘‘The network data repository with interactive graph analytics and visualization,’’ in Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015. [Online]. Available: http://networkrepository.com.

# Contact With The Authors

Send email to the following Authors for any question about this work, and it is our pleasure to ansawer your question.

Mohammad Aleiadeh, maleiade@purdue.edu

Mustafa Abdallah, abdalla0@purdue.edu
