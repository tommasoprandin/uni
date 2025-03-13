# MapReduce and Partitioning

In a wide spectrum of domains there was an increasing need to _analyze large amounts of data_. Available tools and commonly used platforms could not practically handle very large datasets.

It was important to find powerful computing systems, which feature multiple processors, multiple storage devices, and high-speed communication networks. However, these powerful systems had several problems like the maintenance cost, huge programming skills required for the task allocation for leveraging on parallel computing and, moreover, fault-tolerance becomes serious issue: a large number of components implies a low Mean-Time Between Failures (MTBF).

When we talk about fault-tolerance we mean the ability of a system to continue functioning properly in the event of the failure of some of its components. From the moment that there are a large number of components, it implies a very low MTBF that is a measure used in engineering to quantify the reliability of a system. It represents the average time elapsed between two consecutive failures of a component within the system. In other words, it indicates how long a component is expected to operate before it fails. Clearly, the reliability of a system is inversely proportional to the number of its components.

## MapReduce in General

MapReduce was firstly introduced as a programming framework for _big data processing_ on _distributed_ platforms, where large datasets are processed in parallel.

Being a programming framework means that MapReduce provides both a model for writing applications and the infrastructure to execute them, based on clusters. The framework handles many of the complexities involved with running parallel algorithms, including fault tolerance, data distribution, and task scheduling which are hidden to the programmer.

Over time, as the field of big data evolved, the concept of MapReduce was integrated into larger, more versatile systems. Now MapReduce applications runs on clusters of commodity processors (as a laptop) and cloud infrastructures. Several software frameworks have been proposed to support MapReduce programming. For example Apache Spark or Apache Hadoop.

### Cluster's architecture

A cluster refers to a computing platform consisting of multiple interconnected compute nodes via a fast network. Typically, these nodes are traditional computers (often referred to as "commodity hardware"), each with its own RAM, disk, and operating system.

This collection of nodes becomes a single computing platform - a _multiprocessor_ - that can be used to run a given application, thanks to the nodes communicating with each other via the network.

When a large data set is involved, the processing job can be divided into smaller chunks, which are then processed simultaneously by different nodes in the cluster. This parallelism significantly speeds up processing time compared to doing the same job on a single machine.

A typical cluster architecture is composed by racks of 16-64 compute nodes (CPU or more CPUs) connected by switches to work together.

![[mr-cluster-arch.png]]

Moreover, a fundamental part of the cluster is the **Distributed File System**: the role of the DFS in a cluster environment, especially in large-scale data processing tasks like those performed by MapReduce, is vital. It allows the system to efficiently distribute the workload across multiple nodes by ensuring that the necessary data is available locally where the tasks are executed, reducing network traffic and speeding up processing times.

The DFS subdivides the dataset into chunk (e.g. of 64 MB) and replicates each chunk in different nodes in order to ensure fault-tolerance. An example of DFS is Hadoop DFS.

### What is a cloud infrastructure?

As said before, MapReduce applications runs not only on clusters of commodity processors but also on cloud infrastructure. Cloud infrastructure refers to the virtualized and scalable resources that are provided over the internet by cloud service providers.

These resources include computing power, storage, networking, and even entire platforms and software that run on remote data centers. The basic idea is to provide users the capability to access and manage resources without the need to own and maintain physical hardware.

We will use MapReduce as a conceptional model for high-level algorithm (more abstract) design and analysis and Spark as a programming framework for implementing MapReduce algorithms. The most important thing for MapReduce is the _data centric_ view. Moreover it is inspired by functional programming (use of maps and reduce functions).

## MapReduce as a Conceptual Model

A MapReduce computation can be viewed as a sequence of rounds. A_ round_ transforms a set of key-value pairs into another set of key-value pairs (data centric view), through the following two phases:

- **Map phase**: for each input key-value pair separately, a user-specified map function is applied to the pair and produces ≥ 0 other key-value pairs (provides other pairs), sometimes called intermediate pairs
    
- **Shuffle**: intermediate pairs are grouped by key, by creating, for each key $k$, a list $L_k$ of values of intermediate pairs with key $k$
    
- **Reduce phase**: for each key k separately, a user-specified reduce function is applied to $(k, L_k)$ which produces ≥ 0 key-value pairs, which is the output of the round. The function $m: (k, L_{k}) \to \{(k'_{1}, v'_{1}), \dots , (k'_{n}, v'_{n})\}$ is called _reducer_.

![[mr-model.png]]

We can give also a representation of multiple round: the input of a round comprises input data and/or data from the outputs of the previous round. It's convenient to define shared variables, available to all executors, which maintain global information.

![[mr-multiple-rounds.png]] ^8433e5

### Why key-value pairs?
It's natural to question why we use key-value pairs instead of representing data as object. The answer is since MapReduce has a data-centric view the keys are needed as address to reach objects and as labels to define the groups in the reduce phases. In a traditional procedural program the objects would be referred to using pointers to their memory address, this is not obviously possible to do in such a distributed, multi-computer cluster, where memory addresses makes sense only locally, thus we need an alternative way to address objects. Keys are then necessary for grouping them before the reduction phase.

The domains for key and values can change from one round to another and, using frameworks as Spark, it's possible to manage data without using keys (they are implicitly computed).

### Algorithm Specification
A MapReduce algorithm must be specified so that:

- input and output are clearly defined
- sequence of rounds executed has to be unambiguously implied
- for each round are clear:
    - input, intermediate and output sets of key-value pairs
    - function applied in the two phases
    - bounds for key performance indicators

In order to specify a MR algorithm with a fixed number of rounds, we will use pseudocode as in the following example that describes the _word count_ application illustrated in the [[#^8433e5 | previous picture]]:

> **Input**: $k$ documents $D_{1}, \dots, D_{k}$, for any $i \in 1,\dots,k$ the document $D_{i} = (\text{name}, W_{i})$, where $W_{i}$ is the list of words in the document.
> **Output**: a set of pairs $C = \{ (w, c(w)) : w \in D_{i} \cup \dots \cup D_{k}\}$, where $w$ is any word found in the documents, and $c(w)$ is its number of occurrences.
> **Round 1**:
> - _Map_: $\forall D_{i}$ apply separately: $$
D_{i} \to \{ (w_{i}, c_{i}(w)) : w_{i} \in D_{i}, c_{i}(w) = \# \text{ occurrencies of } w_{i} \text{ in } D_{i} \}$$
> - _Reduce_: we can now group the output as $R_{w} = (w, L_{w})$, where $L_{w} = (c_{1}(w), \dots, c_{k}(w))$. Then the reducer is: $$
R_{w} \to \left( w, \sum_{i=1}^{k}c_{i}(w)\right)$$

### Algorithm Execution
There it follows an execution example:
![[mr-execution-exa.png]]

And the underlying system architecture:

When we run our algorithm on distributed platform:

- **Executor creation**: the user program is _forked_ into a _master process_ and multiple _executor processes_. The master is responsible for assigning tasks to executors, and monitor their status.
- **Input and output** files reside on a _Distributed File System_, while intermediate data are stored on the executors' _local memories_/disks.
- **Map phase**: to each executor is assigned a subset of input pairs and applies the map function to them sequentially, one at a time.
- **Shuffle**: the system collects the intermediate pairs from the executors' local spaces and groups them by key.
- **Reduce phase**: each executor is assigned a subset of $(k, L_k)$ pairs, and applies the reduce function to them sequentially, one at a time.

![[mr-cluster-exec-arch.png]]

### Key Performance Indicators
In order to analyze the MR algorithm is needed to estimate the key performance indicators:

- **Number of rounds** $R$; rough estimate of the running time under the assumption that SHUFFLES dominate the complexity and have comparable costs across different rounds.
- **Local space** $M_L$ that is the maximum amount of main memory required by a single invocation of a map or reduce function. The maximum is taken over all rounds and all invocations at each round. In other words is the maximum amount of main memory needed at each executor, i.e., at each compute node.
- **Aggregate space** $M_A$ that is the maximum amount of space which is occupied by stored data at the beginning/end of a map or reduce phase. The maximum is taken over all rounds. It's the overall amount of disk space needed in the system, i.e., the capacity of the DFS.

![[mr-cluster-kpi.png]]

As example, we can analyze the example above for Word Counting: 
- Number of rounds: obviously the number of rounds is 1 $\implies O(1)$
- Local space: for providing $M_L$ we define $N_i$ that is the number of words of a single document. We take the maximum across all documents that is $N_{MAX}$.
	- Map Phase: we have $O(N_{MAX})$ local space, since the map is invoked singularly on each document.
	- Reduce Phase: we have $O(k)$ local space, since the size of $L_{k}$ is proportional to the number of words $k$.
	$\implies$ the maximum amount of local space used is $O(max\{ N_{MAX}, k \})$, which can be reduced to $O(N_{MAX} + k)$
- In order to compute $M_A$ we define $N$ as the total number of words and, space occupied for input, intermediate and output phase is always $O(N)$ so $M_A = O(N)$.

### Design Goals for MapReduce Algorithms

In order to make feasible the computation feasible over very large datasets, it is important to keep in mind some "rules".

First of all we observe that:
> For _every_ problem solvable by a sequential algorithm in space $S$, there is an equivalent MapReduce algorithm that:
> - Uses only _one round_
> - Has local memory usage $M_{L}$ proportional to $S$
> - Has aggregate memory usage $M_{A}$ proportional to $S$

This comes from the simple observation that, by using a map function that gives all the data in input the same label, and then applying to the only reducer the sequential algorithm, we obtain the same results.

Unfortunately this is not practical for very large inputs, since:
- Very large amounts of memory are needed
- Parallelism is not leveraged 

We can summarize the guiding goals for a MapReduce algorithm designer as follows:

**Design Goals**: ^c92080
1. Constant (and as few as possible) number of rounds ($O(1)$), since the bulk of time complexity comes from the _shuffle_ phase.
2.  Sublinear local space ($M_{L} \in O(|input|^\epsilon, \epsilon < 1)$), since input is supposed to be extremely large.
3. Linear aggregate space ($M_{L} \in O(|input|)$), at most _slightly_ superlinear.
4. Map and Reduce functions have _low complexity_.

Goals 1 and 4 aim to minimize time complexity, while 2 and 4 aim to minimize space complexity.

As always there is a tradeoff between time and space, thus a suitable balance must be sought by the designer.

### Suitability of MapReduce for Big Data Computing

The following table sums up the key advantages and disadvantages for the MapReduce framework in the Big Data field.

| Advantages | Disadvantages |
| - | - |
| **Data-centric view**: The algorithm design focuses on data transformations, making it more intuitive to design  | **Runtime analysis**: Runtime costs are only coarsely captured by the number of rounds. More sophisticated analysis are available but hard to use in practice. |
| **Usability**: Frees the designer from complex architectural considerations (tasks, parallelism, distribution, failure management) | **Curse of the last reducer**: Quite often there will be a much slower reducer then others, thus the algorithm should take care of balancing load as much as possible.|
| **Portability**: Once designed the algorithm can run on any architecture (on premise or cloud) that supports the paradigm | **Not suitable for HPC**: The architecture is not suitable when maximum performance is required, since those scenarios require careful management of the underlying platform. |
| **Cost**: MapReduce can be run on moderately expensive platforms and there are cloud options available | | 

### Partitioning

In order to achieve the local space-bound goals [[#^c92080 | stated]], a common approach is to design the algorithm such as to _partition_ the incoming data and work on these subdivisions. This often requires more rounds (and very rarely more aggregate space), but it is a necessary tradeoff to limit local memory usage $M_{L}$, and make computation feasible.

#### Deterministic Partitioning

In the case our data input is provided with some kind of unique labelling, we can partition the objects based on that label _deterministically_ (i.e. with the application of a function).

Let's see an example:

##### Class count

^ec69bc

**Problem**: given a set $S$ of $N$ objects with class labels, count how many objects belong to that class.

**Input**: set $S$ of $N$ objects represented by pairs $(i, (\gamma_{i}, o_{i}))$, for $i = 0,\dots,N$, where $\gamma_{i}$ is the class of the $i$-th object.

**Output**: the set of pairs $(\gamma, c(\gamma))$, where $\gamma$ is a class labeling some object(s) of $S$ and $c(\gamma)$ is the number of objects of $S$ labeled with $\gamma$.\

###### Naive approach without partitioning

**Description**:
- *ROUND 1*:
	- _MapPhase_: $\forall$ input pair $(i, (\gamma_{i}, o_{i})) \mapsto (\gamma_{i}, 1)$
	- _ReducePhase_: $\forall$ the labels $\gamma$ produced by the intermediate pairs, 
	Let $L_{\gamma} :=$ list of 1's from intermediate pairs.
	Then: $(\gamma, L_{\gamma}) \mapsto (\gamma, |L_{\gamma}| = c(\gamma))$

**Analysis**:
1. Only 1 rounds always _OK_
2. In the worst case, where all the elements belong to the same class $\gamma$, $L_{\gamma}$ is proportional to the input size $\implies M_{L} \in O(|input|)$  _NOT OK_
3. In the worst case, where all the elements belong to different classes, the aggregate storage is proportional to the input size $\implies M_{A} \in O(|input|)$ _OK_

We see that this approach doesn't work well because it uses too much local space.

###### 2-Round approach with deterministic partitioning

Let $l$ be the number of desired partitions.
**Description**:
- *ROUND 1*:
	- _MapPhase_: $\forall$ input pair $(i, (\gamma_{i}, o_{i})) \mapsto (i\text{ mod }l, \gamma_{i})$
	- _ReducePhase_: $\forall\ j = 1,\dots,l$ separately
	Let $L_{j} :=$ list of class labels $\gamma$ from intermediate pairs.
	Then $(j, L_{j}) \mapsto \{ (\gamma, c(\gamma, j):\gamma \in L_{j} \}$
![[mr-wc-r1.png]]
- *ROUND 2*:
	- _MapPhase_: empty
	- _ReducePhase_: $\forall\ \gamma$ separetely
	Let $L_{\gamma} :=$ list of $c(\gamma, j)$ from round 1.
	Then $(\gamma, L_{\gamma}) \mapsto (\gamma, \sum_{c(\gamma, j)\in L_{\gamma}}{c(\gamma, j)})$
![[mr-wc-r2.png]]

**Analysis**: ^676a53
- 2 constant rounds _OK_
- Let's break down local memory usage for every step:
	- Map on round 1 operates on every input pair separately $\implies \in O(1)$
	- Reduce on round 1 operates on lists that are _at most_ $\frac{N}{l}$ long $\implies \in O\left( \frac{N}{l} \right)$
	- Map on round 2 does nothing
	- Reduce on round 2 operates on lists that have at most $l$ partial counts for every category $\implies O(l)$
	$\implies M_{L} \in O\left( \frac{N}{l} + l \right)$, then by imposing $l = \sqrt{ N }$ we obtain $M_{L} \in O(\sqrt{ N })$ _OK_
- The aggregate space requirement remain bounded by the input size $\implies M_{A} \in O(N)$ _OK_

This shows that we have found a better time-space tradeoff for execution.
In general the choice of $l = \sqrt{ N }$ partition is usually the optimal one, if more stringent bounds on the local space are imposed, then a more aggressive partitioning strategy will be needed.

In practice the number of possible partitions is fixed by the amount of _available workers_ (as long as local space is not an issue).

#### Random Partitioning

In the case where input objects are not provided with a unique label that lets us apply a partitioning function, we have to split them via a random assignment. In particular we will use a uniformly distributed random variable to try and keep partitions equally sized.

Let's see the example [[#^ec69bc | above ]], modified with un-indexed data:

##### Class count (unindexed)
**Problem**: given a set $S$ of $N$ objects with class labels, count how many objects belong to that class.

**Input**: set $S$ of $N$ objects represented by pairs $(\gamma_{i}, o_{i})$, for $i = 0,\dots,N$, where $\gamma_{i}$ is the class of the $i$-th object.

**Output**: the set of pairs $(\gamma, c(\gamma))$, where $\gamma$ is a class labeling some object(s) of $S$ and $c(\gamma)$ is the number of objects of $S$ labeled with $\gamma$.\

Let's now see a solution and its analysis with random partitioning:
###### 2-Round approach with random partitioning

Let $l$ be the number of desired partitions.
**Description**:
- *ROUND 1*:
	- _MapPhase_: Let $X \sim \mathcal{U}(0;l)$.
	Then $\forall$ input pair, let $\mathrm{x}$ be a realization of $X$; $(\gamma_{i}, o_{i}) \mapsto (\mathrm{x}, \gamma_{i})$
	- _ReducePhase_: $\forall\ \mathrm{x} = 0,\dots,l$ separately
	Let $L_{\mathrm{x}} :=$ list of class labels $\gamma$ from intermediate pairs.
	Then $(\mathrm{x}, L_{\mathrm{x}}) \mapsto \{ (\gamma, c(\gamma, \mathrm{x}):\gamma \in L_{\mathrm{x}} \}$
![[mr-wc-rand-r1.png]]
- *ROUND 2*:
	- _MapPhase_: empty
	- _ReducePhase_: $\forall\ \gamma$ separetely
	Let $L_{\gamma} :=$ list of $c(\gamma, j)$ from round 1.
	Then $(\gamma, L_{\gamma}) \mapsto (\gamma, \sum_{c(\gamma, j)\in L_{\gamma}}{c(\gamma, j)})$
![[mr-wc-rand-r2.png]]

**Analysis**:
Let
$$
\begin{align}
&m_{\mathrm{x}} = \text{number of intermediate pairs with random key } \mathrm{x} \\
&m = max\{ m_{\mathrm{x}}:0\leq \mathrm{x} < l \}
\end{align}
$$

Analysis is identical to the [[#^676a53|deterministic]] one, except that we need to adapt local memory usage, since partitions are not equally and deterministically made of $\frac{N}{l}$ elements, but they are at worst $m$ large $\implies M_{L} \in O(m+l)$.
Now if we can prove that $m \sim \frac{N}{l}$, then random partitioning will have the same complexity as the deterministic one. In other words we want to show that, under certain conditions, the random approach creates evenly sized partitions.

We will show that for large enough datasets the asymptotic local space is the same of deterministic partitioning.

> **Theorem**:
> Fix $l = \sqrt{ N }$, then, by partitioning the dataset with keys drawn from a uniform distribution $\sim \mathcal{U}(0; \sqrt{ N })$, with probability _at least_ $1-\frac{1}{N^5}$:$$
m \in O(\sqrt{ N })$$

**Proof**

Let's recall two important bounds that will be important for the proof:
1. _Union Bound_
	Given a countable set of _events_ $E_{1}, \dots, E_{n}$, we have: $$
Pr\left(\bigcup_{i=1}^{n}{E_{i}}\right) \leq \sum_{i=1}^{n}Pr(E_{i})$$ ^414b8b
1. _Chernoff Bound_
	Let $X_{1}, \dots, X_{n}$ be $n$	iid Bernoulli random variables with $\phi = p\ \forall\ 1 \leq i \leq n$. Thus $X = \sum_{i=1}^{n}X_{i}$ is a binomial distribution $\mathcal{B}(n, p)$. Let now $\mu = \mathbb{E}[X] = n \cdot p$.
	For every $\delta_{1} \geq 6$ and $\delta_{2} \in (0;1)$ we have that: ^580066
	1. $Pr(X\geq \delta_{1}\mu) \leq 2^{-\delta_{1}\mu}$
	2. $Pr(X\leq (1-\delta_{2})\mu) \leq 2^{-\delta_{2}^2\mu/2}$

Assume now $N \geq 16$. Pick now an arbitrary key $x$ drawn from the uniform distribution, $\implies x \in [0;l = \sqrt{ N })$.

Now recall $m_{x} = |\text{intermediate pairs with random key }x|$.

Define a Bernoulli random variable $Y_{i}$ such that
$$
Y_{i} = \begin{cases}
1 & \text{if } (\gamma_{i}, o_{i}) \mapsto (x, \gamma_{i}) \\
0 & \text{otherwise}
\end{cases}
$$
$\implies Y_{i}$ is a Bernoulli with $\phi = \frac{1}{l} = \frac{1}{\sqrt{ N }}$.

Thus we can define $m_{x}$ as:
$$
m_{x} = \sum_{i=1}^{N}{Y_{i}} \implies m_{x} \text{ is } \mathcal{B}\left( N, \frac{1}{\sqrt{ N }} \right) \implies \mathbb{E}[m_{x}] = \sqrt{ N }
$$

By the Chernov bound [[#^580066|1]], we obtain that:
$$
\begin{align}
 & P(m_{x} \geq 6\mathbb{E}[x]) \leq 2^{-6\mathbb{E}[x]} \\
 & P(m_{x} \geq 6\sqrt{ N }) \leq 2^{-6\sqrt{ N }} \\
\end{align}
$$

Now for $N \geq 16$ by hypothesis we have that $\sqrt{ N } \geq \log_{2}(N)$, thus:
$$
2^{-6\sqrt{ N }} \leq 2^{-6\log_{2}(N)} = \left( \left( \frac{1}{2} \right)^{\log_{2}{N}} \right)^6 = \left( \frac{1}{N} \right)^6
$$
Hence we have bounded $P(m_{x} \geq 6\sqrt{ N }) \leq \frac{1}{N^6}$.

Now we have to extend the bound to the maximally-sized partition.
Let $m = max{\{ m_{x}:x \in [0; \sqrt{ N }) \}}$.
Define the event $E_{x} = \text{partition with key } x \text{ has } m_{x} \geq 6\sqrt{ N }$.

Then:
$$
P(m \geq 6\sqrt{ N }) = P\left(\bigcup_{i = 0}^{\sqrt{ N } - 1}{E_{i}}\right) \text{(at least one partition is} \geq 6\sqrt{ N })
$$

By the [[#^414b8b|union bound]] 
$$
\begin{align}
P\left(\bigcup_{i = 0}^{\sqrt{ N } - 1}{E_{i}}\right) &\leq \sqrt{ N } \cdot P(E_{x})\ \forall\ x \\
 & \leq \sqrt{ N } \cdot \frac{1}{N^6} (\text{see above bound}) < \frac{1}{N^5}
\end{align}
$$

Therefore the probability that $m \geq 6\sqrt{ N }$ is less than $\frac{1}{N^5}$.
$$
\implies P(m < 6\sqrt{ N }) \geq 1-\frac{1}{N^5} \implies m \in O(\sqrt{ N }) \text{ with probability } \geq 1 - \frac{1}{N^5}
$$

This proves that, given a good random generator and a sufficiently large dataset, random partitioning is equally good in terms of local space asymptotic complexity. For a more detailed probability bounds explanation refer to [[Probability Cheat Sheet#^68c290 | the probability cheat sheet]].