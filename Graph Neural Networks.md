## Introduction

Traditional ML approaches have been developed assuming data to be encoded into feature vectors; however, many important real-world applications generate data that are naturally represented by more complex structures, such as graphs. Graphs are particularly suited to represent the relations (arcs) between the components (nodes) constituting an entity. For instance, in social network data, single data "points" (i.e., users) are closely inter-related.

A graph $G = (V, E)$ can be represented using the so called **adjacency matrix**. A $n \times n$ matrix $A$ such that 
$$
A[i,j] = \begin{cases}
1 & \text{if }edge(i,j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

**Example:**

![[graph-dnn-exa.png]]

The Adjacency matrix of the graph above is the following: $$\begin{bmatrix} 0 & 1 & 0 & 0 & 1 \\ 1 & 0 & 1 & 1 & 1 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 & 1 \\ 1 & 1 & 0 & 1 & 0 \end{bmatrix}$$

In undirected graphs this matrix is **symmetric**, while in directed graphs it is **asymmetric**. In case of a weighted graph, each cell of the matrix has either the value of the edge weight $w$ or $-$. Each node and edge is represented by a feature vector:

![[graph-dnn-repr.png]]

The position $(m,n)$ of the adjacency matrix contains the number of walks of length one from node $m$ to node $n$. Position $(m,n)$ of the **squared** adjacency matrix $A^2$ contains the number of walks of length two from node $m$ to node $n$, and so on for $A^{k}$.

The main problem settings that can arise when dealing with structured data are the following:

- **Predictions over nodes in a network**: In this setting, the dataset is composed of a single (possibly disconnected) large graph. Each example is a node in the graph, and the learning tasks are defined as predictions over the nodes. Given an unseen node $u$, the task is to predict the correct target $y_u$. An example in this setting is the prediction of properties of a social network user based on his or her connections. This is the idea: we have one big graph, and each node in the graph represents a data point (an entity). The goal is to predict some property or label of that node.
    
- **Predictions over graphs**: In this case, each example is composed of a whole graph, and the learning tasks are predictions of properties of the whole graphs. An example is the prediction of toxicity in humans of chemical compounds represented by their molecular graph. The idea: each entire graph is one data point. So your dataset is made of many small graphs, and the goal is to classify or regress some property of each graph.
    
- **Link-prediction tasks**: the model predicts whether or not there should be an edge between nodes. This is the idea: given a graph, we want to predict whether an edge should exist between two nodes. You're not labeling nodes or graphs.
    

## Learning on graphs is difficult

Let $\mathbf{}$ be the matrix in which the feature vectors of each node are stored. We can observe that node indexing in graphs is arbitrary. This means that, differently from images, permuting the node indices results in a permutation of the columns of $\mathbf{X}$ and a permutation of both the rows and columns of $A$. However, the underlying graph is unchanged. This property is called Permutation Invariance. More formally, given a permutation matrix $\mathbf{P}$, we get a different representation of the same graph:

$$\begin{align} \mathbf{X}' &= \mathbf{XP} \\
 \mathbf{A}' &= \mathbf{P}^T \mathbf{AP} \end{align}$$

This property can give the intuition about why learning on graphs is difficult. In fact, determining if two graphs are equal (_graphs isomorphism_) is a problem for which are not known polynomial-time algorithms. Furthermore, sub-graph isomorphism, which is the problem of determining if a graph is a sub-graph of another graph, is NP-Complete. These problems affect machine learning because a model should be able to predict the same output for isomorphic graphs (which can be represented in different ways). Furthermore, the model we design should capture the similarity between two graphs (sub-graph isomorphism).

In general, the main problems we face when learning on graphs are the following:

1. Same graph can be represented in different ways;
2. How to recognize that a given graph $G_2$ is a sub-graph of $G_1$
3. How to represent graphs of different sizes (i.e., different number of nodes) into fixed-size vectors without losing expressiveness?
4. How to avoid explosion in the number of parameters with the size of the graphs?

Problem 3 is commonly faced by using recursive models that exploit a causal state space [Sperduti & Starita., TNN 1997], while Problem 4 is commonly faced by exploiting shared parameters.

Regarding Problems 1 and 2, a sound and meaningful representation for graphs can be achieved by using a neural network with **convolution operator**, defined on graphs.

## Graph Neural Networks - General Idea

A Graph Neural Network (GNN) receives in input a graph (adj. matrix and node representations for simplicity) and passes it through a series of $k$ layers. Each layer computes a hidden representation for each node, with the last layer computing the final nodes' embeddings $\mathbf{H}_k$. Similarly to CNNs, each node representation includes information about the node and its context within the graph. Another similarity is that in CNN we have matrix in input and also here in Graph, moreover in output we have a pre-fixed representation in both cases.

- For **node-level tasks**, the output is computed from $\mathbf{H}_k$;
    
- For **graph-level tasks**, these are tasks where we want one output for the entire graph then the nodes' embeddings are combined (e.g., by averaging), and the resulting vector is mapped via a linear transformation or neural network to a fixed-size vector from which the classification/regression task is performed. This approach let the graph be permutation invariant: It means that the order of the nodes doesn't matter—shuffling the nodes won't change the final output. This is crucial for graphs because there's no natural ordering of nodes.
    
- For **link-prediction tasks**, the embeddings of the two endpoint nodes must be mapped to a single number representing the probability that the edge is present (e.g. dot product of the nodes' embeddings and pass the result through a sigmoid function to create a probability).
    

![[graph-dnn-tasks.png]]

## Graph Convolution

The general idea of graph convolution starts from a parallel between graphs and images. GNNs implement convolution in a similar way how CNNs do, that is, learning the features by inspecting neighboring nodes. GNNs generalize the definition of convolution for non-regular structured data.

### NN4G by Micheli

**NN4G** is an architecture based on a graph convolution that is defined as:

$$
\begin{align}
 & \boldsymbol{h}^{1}_{v} = \sigma(\bar{W}^{1}\boldsymbol{x}_{v}) \\
 & \boldsymbol{h}^{i}_{v} = \sigma\left( \bar{W}\boldsymbol{x}_{v} + W^{i}\sum_{u \in neigh(v)}\boldsymbol{h}^{i-1}_{u}  \right) & i > 1
\end{align}
$$

where:

- $\sigma$ is a nonlinear activation function applied element-wise.
- $neigh(v)$ represent the neighborhood of node $v$.
- $\mathbf{W}^i$ is a weights' matrix;
- $\mathbf{x}_u$ is the feature vector of node $u$.

Actually, this is a simplified notation, since the original one uses skip connections (see GNN book chapter).

![[graph-dnn-nn4g.png]]

Note that:

- The first layer ($i = 1$), which has no previous layers, computes the nodes' representations only on the basis of each vertex feature vector.
    
- Each convolution performed on the $i$-th hidden units, with $i > 1$, takes as input the neighbors' representations of all previous layers. Basically, it merges the representations of each node with those of its neighbors:
    
    - $X_1(g), X_2(g), X_3(g)$ are scalar values computed by aggregating the representations $\mathbf{x}_i(g)$ for each unit $i$. In particular they are defined as: $$X_i(g) = \frac{1}{k}\sum_{v \in Vert(g)}x_i(v)$$ if $k = 1$, this corresponds to a sum. Basically, we compute a representation per-graph per-layer. Then, these 3 representations are parametrized by the weights $w_1, w_2, w_3$ and used to compute the output for the whole graph (graph-level task).

The convolutional operation presented above can be defined in a compact way using matrix multiplications:

$$
\begin{align}
 & H^{1} = \sigma(\bar{W}^{1}X) \\
 & H^{i} = \sigma(\bar{W}^{i}X + W^{i}AH) & i>1
\end{align}
$$

where $A$ is the adjacency matrix. Exploiting matrix multiplications makes the computation really fast. Furthermore, note that, as for images, the receptive field of the layers increases as we stack more layers.

Note also that, since with the convolution operator we are merging neighboring nodes' representations, isomorphic graphs in which the order of the nodes is changed will have the same nodes' representations.

### Graph Fourier Transform

The operation described above is graph convolution, but how is it derived? Defining the formal convolution operator on graph is difficult.

Let $x$ be a single undirected graph, $x: V \rightarrow \mathbb{R}$ be a signal on the nodes $V$ of the graph $G$, i.e., a function that associates a real value with each node of $V$. We can represent every signal as a vector $\mathbf{x} \in \mathbb{R}^n$, which from now on we will refer to as signal. In order to set up a convolutional network on $G$, we need the notion of convolution between a signal $\mathbf{x}$ and a filter signal $\mathbf{f}$.

A graph G = (V, E) consists of a set of nodes V and edges E connecting nodes. Graph signals are real-valued functions defined on the nodes of G, i.e. f: V → R. The value f(i) assigned to node i represents some measurement or quantity of interest. For example, in a social network, f(i) could denote the income of user i. The connectivity specified by the graph encodes relationships between nodes.

The key idea is to use a Fourier transform. In the frequency domain, thanks to the **Convolution Theorem**, the (undefined) convolution of two signals becomes the (well-defined) component-wise product of their transforms. So, if we knew how to compute the Fourier transform of a function defined on a graph, we could define the convolution operator.

The passages that we are going to do: adjacency matrix → Laplacian → eigendecomposition → convolution in spectral domain.

The Convolution Theorem states that convolution in one domain (time, space) corresponds to pointwise multiplication in frequency domain.

The **graph Fourier transform** is defined starting from the (normalized) Laplacian matrix of the graph, which is defined as:

$$L = I_n - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$

where:

- $I_n$ is the identity matrix;
- $A$ is the adjacency matrix;
- $D$ is the degree matrix, that is, the **diagonal** matrix containing the number of edges attached to each vertex. It's a diagonal matrix, where each diagonal entry $D_{ii}$ is the degree of node $i$. So $D$ reflects how many neighbors each node has (or the sum of edge weights if weighted). Notice that D's diagonal is something like this: $\frac{1}{\sqrt{d_i}}$, so the sqrt of the degree of node $i$. That is: $$
D = \begin{bmatrix}
\frac{1}{\sqrt{ d_{1} }} & 0 & \dots & 0 \\
0 & \ddots & & \vdots  \\
\vdots &  & \ddots \\
0 & \dots & \dots & \frac{1}{\sqrt{ d_{n} }}
\end{bmatrix}
$$

Then, we can compute the eigendecomposition of $L$, which is always possible because it is real symmetric (since it's made by composition of symmetric matrices), and positive semidefinite:

$$L = U \Lambda U^T$$

where $\Lambda = diag([\lambda_0, ..., \lambda_{n-1}])$ and $U$ is the Fourier basis of the graph. We need the eigendecomposition in order to find the basis $U$ of eigenvectors in order to apply the transform.

Finally, Given a spatial signal $\mathbf{x}$:

- $\hat{\mathbf{x}} = U^T\mathbf{x}$ is its graph Fourier Transform
- $\mathbf{x} = U \hat{\mathbf{x}}$ is the inverse Fourier transform

Convolution operator in the node domain has a correspondence in the Graph frequency domain:

$$x_1 *_G x_2 = \underbrace{ U(\underbrace{ \underbrace{ (U^Tx_1) }_{ \text{to freq. domain} } \odot (U^Tx_2)  }_{ \text{product in frequency domain} }) }_{ \text{go back to the original domain} }$$

Therefore, convolution between a parametric filter and a signal can be defined as:

$$y = \mathbf{f}_\theta *_G \mathbf{x} = U\left( (U^T\mathbf{f}_\theta) \odot (U^T \mathbf{x})\right)$$

It can be proved that this operator corresponds to the one used for NN4G presented previously (see slides for more details). Remember that $\mathbf{f}_\theta$ is the filter of the convolution.

Therefore, the convolution between a **parametric filter** $\mathbf{f}_\theta$ and a signal $\mathbf{x}$ is given by:

$$\mathbf{y} = \mathbf{f}_\theta *_G \mathbf{x} = U \left( (U^T \mathbf{f}_\theta) \odot (U^T \mathbf{x}) \right)$$

We now define:

- $\hat{\mathbf{f}}_\theta = U^T \mathbf{f}_\theta$: the filter in the spectral domain,
- $\hat{\mathbf{x}} = U^T \mathbf{x}$: the signal in the spectral domain.

Thus, the convolution becomes:

$$\mathbf{y} = U \left( \hat{\mathbf{f}}_\theta \odot \hat{\mathbf{x}} \right)$$

Now observe that the Hadamard product can be written as a matrix-vector product by introducing a diagonal matrix. Let:

$$\hat{F}_\theta = \text{diag}(\hat{\mathbf{f}}_\theta)$$

Then:

$$\hat{\mathbf{f}}_\theta \odot \hat{\mathbf{x}} = \hat{F}_\theta \hat{\mathbf{x}}$$

and the graph convolution becomes:

$$\mathbf{y} = U \hat{F}_\theta \hat{\mathbf{x}} = U \hat{F}_\theta U^T \mathbf{x}$$

This is the **linear form in the node domain** of spectral graph convolution. It is useful because:

- it can be implemented efficiently,
- it reveals the linear structure of the operator,
- it forms the theoretical basis for **Graph Convolutional Networks (GCNs)**.

How to design the diagonal matrix $\hat{F}_\theta$?

The simplest way is to use a non-parametric filter: $\hat{F}_\theta = diag(\theta)$ but this is not a good solution because the number of params is equal to the number of nodes in the graph, that means that we are not flexible enough.

We should then use a polynomial filter, which is a parametric filter of the form:

$$\hat{F}_\theta = \sum_{k=0}^{K} \theta_k \Lambda^k$$

**exactly K-localized**  
**K parameters**

We can learn directly in the graph domain:

$$g_\theta(L) = \sum_{k=0}^{K} \theta_k L^k$$

$$\mathbf{y} = f_\theta *_G \mathbf{x} = U \hat{F}_\theta U^T \mathbf{x} = \sum_{k=0}^{K} \theta_k U \Lambda^k U^T \mathbf{x} = \sum_{k=0}^{K} \theta_k L^k \mathbf{x}$$

where k is the dimension of receptive field, and y, output is directly the feature map.

Why? We decide how many k we have, it does not depend on number of nodes! It is a smart choice because if I plug this in my filter of convolution... see below

The reasons why this is cool:

- Small number of parameters
- At the end I don't need the eigendecomposition

## Summary

- **Graph Fourier Transform**
- Fourier basis are eigenvectors of the normalized Graph Laplacian: $$L = U \Lambda U^T$$
- We can then define the graph convolution in the frequency domain: $$f *_G x = U \hat{F} U^T x$$
- For some choice of filters, e.g., polynomials of the spectral matrix: $$\hat{F}_\theta = \sum_{k=0}^{K} \theta_k \Lambda^k \quad \Rightarrow \quad f *_G x = \sum_{k=0}^{K} \theta_k L^k x$$
- Therefore, the convolution can be computed directly in the node space.

**1-localized GCN**: These networks map multisets of representations (both the node and its neighbors from the previous layer) to create a new representation. The mathematical formula shows:

- $h^{l+1}_v = f({h^l_v, h^l_u, \forall u \in ne(v)})$
- Where $H^0 = X$ (initial features)
- This means each node's representation at layer l+1 is computed from its own representation and its neighbors' representations at layer l.

$f$ This is described as a linear mapping followed by a non-linear activation function. An example formula is provided:

- $H^{l+1} = \sigma(\hat{D}^{-1/2}\hat{A} \hat{D}^{-1/2}H^l \theta^l)$
- This is a specific formulation where $\hat{A}$ is likely the adjacency matrix with self-loops, $\hat{D}$ is the degree matrix, $H^l$ is the feature matrix, and $\theta^l$ represents learnable parameters.

It states that with sufficient expressiveness and injective readout, a multilayer 1-localized GCN can be as expressive as the 1-dimensional WL isomorphism test (which is a standard benchmark for graph representation power).

Let's now see how to combine before classify. **Problem Statement**: With Graph Convolution, we have representations for each node in the graph, but need a way to create a single representation for the entire graph.

Aggregation (Pooling) Function:

- Maps a set of node representations to a graph-level representation
- Must be differentiable to work with gradient-based learning

Solutions:

- Naive approaches: Simple sum or average of node representations
- Advanced approaches: More complex alternatives like Universal readout (DeepSets)

![[graph-dnn-aggregation.png]]

This aggregation step is crucial for tasks like graph classification, where the goal is to assign a label to the entire graph rather than to individual nodes.

## Aggregation Layer for graph classification

With Graph Convolution we have a representation for each graph node. How can we map node representations to a graph-level representation? There are some simple solutions, like the sum or the average of nodes' representations as we saw previously, or we can rely on more complex alternatives: Universal readout.

## Graph Recurrent Neural Networks

Scarselli et al. proposed a network architecture where, instead of stacking multiple layers, a single recurrent layer is adopted:

$$\mathbf{h}_v^{t+1} = \sum_{u \in N(v)}f(\mathbf{h}_u^t, \mathbf{x}_v, \mathbf{x}_u)$$

where $f$ is a function (e.g. neural network) with shared parameters across all the nodes and all the time steps. The recurrent system is defined as a contraction mapping, and thus it is guaranteed to converge to a fixed point $\mathbf{h}^*$.

### Gated Graph Neural Networks

The idea is to remove the constraint for the recurrent system to be a contraction mapping, and implement this idea by adopting recurrent neural networks to define the recurrence. Specifically, the gated recurrent unit (GRU) is adopted.