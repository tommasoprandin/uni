## Outline

Suppose we want to solve a problem $\Pi$ on instances $P$ which are too large to be processed by known algorithms. 
We can try to "reduce" our input problem by extracting a smaller subset that is still representative for solving the problem. This will almost always incur in an approximation of the optimal solution, but otherwise solving the original problem would be infeasible

## Formulation

> **Coreset Technique**
> 1. _Extract_ a "small" subset $T$ from $P$, called _coreset_, making sure that it represents $P$ well with regard to the solutions to $\Pi$.
> 2. _Run_ the best known sequential algorithm for $\Pi$ on the coreset $T$, rather than the entire input $P$. Note that the sequential algorithm may be very slow.o

![[coreset-formulation.jpeg]]

To further adapt this approach to MapReduce we can further apply the concept to partitions of the original input $P$, then combine together the intermediate coresets. This will lead to less memory usage and bettery parallelization.

> **Composable Coreset Technique**
> 1. _Partition_ $P$ into $l$ subsets: $P = P_{1} \cup P_{2}, \dots, \cup P_{l}$. Then _extract_ a "small" _coreset_ $T_{i}$ for each partition, making sure that it represents $P_{i}$ "well".
> 2. _Run_ best known sequential algorithm for $\Pi$ on the union of all the _coresets_ $T = \bigcup_{i=i}^l{T_{i}}$, rather than $P$.

^9835a2

The technique is effective if
- Each $T_{i}$ can be efficiently extracted from the partition $P_{i}$ in parallel for all the partitions.
- The final coreset $T$ is still small and the solution for $\Pi$ computed on $T$ is still good for the original input set $P$.

![[coreset-composable-formulation.jpeg]]

## Case Study: Clustering

A good case study is represented by the [[Clustering]] problem, for an explanation refer to the section linked.

### k-center Clustering for Big Data

A complete discussion of the general problem for k-center clustering, and the approximate solution using Farthest-First Traversal is reported [[Clustering#k-center Clustering|here]].

We observe that the FFT algorithm requires $k-1$ scans of the pointset $P$, which is obviously impractical for massive datasets and large $k$.

To solve this problem we resort to the [[#^9835a2|composable coreset technique]].

The idea is to partition our input and solve the k-center clustering in each of the $l$ partitions. This will originate $T_{i}$ solutions for every $1\leq i\leq l$ which will be our _coresets_. Then coresets will be merged together and FFT will be executed on $T = \bigcup_{i=1}^lT_{i}$.
We will prove that this solution approximate the optimal one sufficiently well and makes the problem feasible for huge datasets.

![[coreset-fft-mr.png]]

#### MapReduce Farthest-First Traversal

Let $P$ be a set of $N$ points ($N$ very large), from a metric space $(M, d)$, and let $k>1$ be an integer.

**Algorithm**
- **Round 1**:
	- Map Phase: partition $P$ arbitrarily into $l$ subsets of equal size $P_{1}, \dots, P_{l}$.
	- Reduce Phase: execute FFT on every partition $P_{i}$ separately to determine a set $T_{i} \subseteq P_{i}$ of $k$ centers.
- **Round 2**:
	- Map Phase: empty
	- Reduce Phase: gather the _coreset_ $T = \bigcup_{i=1}^lT_{i}$ (with size $l\cdot k$) and return a solution $S = \{c_{1}, \dots, c_{k}\}$ computed by running FFT on the coreset $T$ using a _single reducer_.

##### Space Complexity

The 2-round MR-FFT can be implemented using local space $M_{L} = O(\sqrt{ Nk })$ and aggregate space $M_{A} = O(N)$, assuming an appropriate number of partitions.

- Round 1:
	- Map Phase: $O(1)$ since it just executes separately on single points for partitioning them
	- Reduce Phase: $O\left( \frac{N}{l} \right)$ since it collects together the points in the partition with size $\frac{N}{l}$
- Round 2:
	- Map Phase: empty
	- Reduce Phase: $O(kl)$ since we have $l$ solutions that have $k$ centers

Thus:
$$
\begin{align}
M_{L} &\in O\left( \max\left\{  \frac{N}{l}, kl  \right\} \right) \implies \\
&\in O\left( \frac{N}{l} + kl \right) \\
\frac{N}{l} = kl \\
N = kl^2 \\
l^2 = \frac{N}{k} \\
l = \sqrt{ \frac{N}{k} } \\
\implies  & \in O(\sqrt{ Nk })
\end{align}
$$

There is no replication in the algorithm, thus $M_{A} = O(N)$.

##### Approximation Analysis

First we will show the quality of the coreset that represents $P$:
Let $T = \bigcup_{i = 1}^lT_{i}$, $T_{i} = FFT(P_{i}, k)$.

Then we have:
$$
\forall x \in P\quad d(x, T) \leq 2\cdot \Phi_{k-center}^{opt}(P, k)
$$

This shows that $T$ represents $P$ well w.r.t. the k-center objective (because the "error" is bounded).

**Proof**:
Recall:
$$
\begin{align}
P = P_{1} \cup P_{2} \dots\cup P_{l} \\
T = T_{1} \cup T_{2} \dots\cup T_{l}
\end{align}
$$

Then, for every $r = 1, \dots, l$ we can define:
$$
q_{r} = \arg\max_{x \in P_{r}} d(x, T_{r})
$$
that is the point in $P_{r}$ farthest from the FFT solution of $P_{r} = T_{r}$.

Now we claim that $d(q_{r}, T_{r}) \leq 2\cdot \Phi_{k-center}^{opt}(P, k)$, which is the global objective function.

Let's fix $r$ arbitrarily and repeat the same argument used in the [[Clustering#Analysis|analysis of sequential FFT]].

Let:
$$
\begin{align}
&S^\star = \{ c_{1}^\star, \dots, c_{k}^\star \} &\text{be the optimal solution for the entire problem over } P \\
&C_{t}^\star &\text{be the optimal cluster around } c_{t}^\star \\ \\
& T_{r} = \{ c_{1}, \dots, c_{k} \} \\
 & T'_{r} = \{ c_{1}, \dots, c_{k}, c_{k+1} = q_{r} \} & \text{by construction of FFT}
\end{align}
$$

We have that:
- $d(q_{r}, T_{r}) \leq d(c_{i}, c_{j}) \quad \forall i\leq i<j\leq k+1$  for the same argument used in the proof of FFT
- By pigeonhole principle there must exist 2 points $c_{a}, c_{b}$ with $1\leq a<b\leq k+1$ that belong in the same optimal cluster $C_{t}^\star$.

$$
\begin{align}
\implies d(q_{r}, T_{r}) &\leq d(c_{a}, c_{b}) \\
 & \leq d(c_{a}, c_{t}^\star) + d(c_{t}^\star, c_{b}) \text{ by triangle inequality} \\
 & \leq 2 \Phi_{k-center}^{opt} (P, k)
\end{align}
$$
which proves the claim.

---

Now we will establish the approximation factor for the complete algorithm:

From the previous proof, we know that $\forall x \in P, \exists y \in T$ such that:
$$
d(x, y) \leq 2\cdot \Phi_{k_-center}^{opt}(P, j)
$$

Recall that $S$ is extracted from $T$ by running `FFT(T, k)`. Let $\bar{y}$ be the point of $T$ farthest from $S$, and recall that $T \subseteq P$.

By repeating the same argument used above we can show that:
$$
d(\bar{y}, S) \leq 2\cdot \Phi_{k-center}^{opt}(P, k)
$$

We know:
- $\forall x \in P, \exists y \in T$ such that $d(x, y) \leq \Phi_{k-center}^{opt}(P, k)$
- $\forall y \in T, \exists c \in S$ such that $d(y, c) \leq d(\bar{y}, S) \leq \Phi_{k-center}^{opt}(P, k)$

And:
$$
\begin{align}
d(x, S)  & \leq d(x, c) \\
 & \leq \underbrace{ d(x, y) + d(y, c) }_{ \text{triangle inequality} } \leq 4\cdot \Phi_{k-center}^{opt}(P, k)
\end{align}
$$

Thus the approximation factor is 4.

##### Observations

- FFT provides _good coresets_ $T_{i}$, hence a good final coreset $T$ since it ensures that any point not belonging to $T$ is well represented by some coreset point.
- MR-FFT is able to handle _very large pointsets_ and the final approximation is not too far from the best obtainable.
- Observer that for any optimization problem, **no MR algorithm can obtain better accuracy than the corresponding sequential algorithm**.
- When $P$ has _low dimensionality_ the quality of the solution returned by MR-FFT can be made arbitrarily close to 2 by selecting a _larger coreset_, while still ensuring the desired space requirement. In other words we can decide to extract coresets from the partitions with more representatives $\bar{k} > k$, this will improve quite sharply the accuracy (especially for low dimensions), while still mantaining a good local space occupation since $|T| = \bar{k}l > kl$.

##### Random Partitioning

It would be natural to wonder if random partitioning would return a good coreset for our clustering problem, simplifying the implementation.

Unfortunately this is clearly not the case, since it is intuitive to see that picking points randomly does not give any guarantee of good representation for the points in the original datasets.

For instance let's imagine a pointset $P$ with a few outliers concentrated very far from the bulk of points. For very large input sets and very low number of outliers, the probability of picking them tends to be extremely small.

The problem is that they are very important for the k-center problem, since the objective function aims at minimizing the maximum distance, hence we need to ensure we represent the farthest points in our coreset, which random partitioning does not.

This also show why k-center is very sensitive to noise, since solutions sways dramatically in presence of even a very small number of outliers.

More formally, consider this example:
![[coreset-random-fft.png]]
Let $k= 2$, and consider a 2-partition of $P = P_{1} \cup P_{2}$ where the size of $P_{2} = m$ with $m$ very small compared to the full set $(m = o(\sqrt{ N }))$. Then the size of $P_{1} = N-m$.

Also assume that these two partitions are very far off from each other, that is $P_{2}$ can be thought as noise for the point set $P$.
Formally, let $\Delta_{1}$ be the diameter of $P_{1}$, and $\Delta_{2}$ the distance between $P_{1}$ and $P_{2}$, with $\Delta_{1} \ll \Delta_{2}$. The diameter of $P_{2}$ is assumed much smaller than $\Delta_{1}$.

Now let our randomly generated coreset $T$ be a set of $\sqrt{ Nk }$ points drawn randomly with uniform probability and replacement from $P$. Then:
$$
\begin{align}
Pr(T \text{ contains some point of }P_{2}) \leq \sum_{i=1}^{\sqrt{ Nk }} \frac{m}{N} = \frac{m}{N}\sqrt{ Nk } \\
\text{assume }k = 2, m \in o(\sqrt{ N }) \implies \lim_{ N \to \infty } \frac{m}{N}\sqrt{ Nk } =0
\end{align}
$$

Thus, for large $N$ it is likely that the randomly built coreset will not include points from $P_{2} \implies T \subseteq P_{1} \implies$ the solution $S$ will be computed on a subset of $P_{1} \implies \forall x \in P_{2}: d(x, S)\geq\Delta_{2}$  by construction.

But by selecting one center in $P_{1}$ and one in $P_{2}$ the objective function becomes $\leq \Delta_{1}$:
$$
\Phi_{k-center}^{opt}(P, S') \leq \Delta_{1} \ll \Delta_{2} \text{ by hypothesis}
$$

Thus by making the distance $\Delta_{2}$ arbitrarily large we can make the approximation made by using random partitioning arbitrarily bad.

This proves that randomly constructing the coreset $T$ **does not give any quality guarantee**.

## k-center as a Coreset Primitive

In [[#MapReduce Farthest-First Traversal]], k-center is employed to extract a coreset $T$ for solving k-center itself. We will now show that it is also useful to extract representative coresets for other problems as well.

### Diameter

Given a set $P$ of $N$ points from a metric space $(M, d)$ determine its _diameter_, defined as:
$$
\Delta(P) = \max_{x, y \in P}d(x, y)
$$
that is, the _maximum distance between two points_.

It is often used as a metric for _graph analytics_.

![[coreset-diameter.png]]

Unfortunately, the computation of exact diameter of a set $P$ requires _almost_ quadratic operations ($O(N^2)$), except for low-dimensional spaces, hence it is _impractical_ for very large pointsets. Note that it is not an NP-hard problem, but even a quadratic complexity makes it unfeasible for very large inputs.

Thus we need to develop an approximation algorithm.

#### MR Exact Diameter

(TODO): write an MR algorithm for exact diameter with 2 rounds and $\sqrt{ N }$ local space

#### 2-Approximation Diameter

For any $x_{i} \in P$, define:
$$
\Delta_{i} = \max\{ d(x_{i}, x_{j}): 0 \leq j \leq N \}
$$

Then, for any $0 \leq i \leq N$ (i.e. for any point in $P$) we have:
$$
\Delta(P) \in [\Delta_{i}, 2\Delta_{i}]
$$

**Proof**

![[coreset-diameter-2-approx.png]]

- $\Delta \geq \Delta_{i}$ by the definition of $\Delta$ and $\Delta_{i}$, ($\Delta$ is the max distance between two points, and $\Delta_{i}$ is a distance between some pair of points $(x_{i}, x_{j})$)
- $\Delta \leq 2\Delta_{i}$  since (as in the example), $\Delta = d(z, w) \leq d(z, x_{i}) + d(x_{i}, w) \leq 2\Delta_{i}$ where $z, w$ are the 2 points attaining the diameter, and $x_{i}$ is arbitrary

Can we get a better approximation?

#### Coreset-based Diameter Approximation

The idea is that by computing a k-center clustering of the original set, with sufficiently large $k$, we can approximate the maximum distance computing it on the centers (used as representative for the points in its cluster), instead of the entire pointset.

The larger $k$ is the better the approximation is.

##### Steps
1. Fix a suitable granularity $k \geq 2$
2. Extract a coreset $T \subset P$ of size $k$ by running a k-center clustering algorithm on $P$ and taking the $k$ cluster center as set $T$
3. Return $d_{T} = \max_{x, y \in T}d(x,y)$ as an approximation of $d_{\max}$

If $k = O(1)$, then $d_{T}$ can be computed:
- _Sequentially_ in $O(N)$ time using [[Clustering#Farthest-First Traversal|FFT]]
- In _MapReduce_ in 2 rounds with local space $M_{L} = O(\sqrt{ N })$ and aggregate space $M_{A} = O(N)$ using [[#MapReduce Farthest-First Traversal]].

##### Approximation Analysis

Consider:
- $T = \{c_{1}, \dots, c_{k}\}$
- $q$ = point of $P$ farthest from $T$
- $R = d(q, T)$ which is the max distance of a point from its closest center (i.e. the radius of the cluster)

Let 
- $\Delta =$ true diameter, $\Delta = d(z, w)$
- $\Delta_{T} =$ coreset diameter
- $c_{i}$ = center of $T$ closest to $z$
- $c_{j}$ = center of $T$ closest to $w$

This implies:
$$
\begin{align}
\Delta  & = d(z, w) \\
 & = \underbrace{ d(z, c_{i}) }_{ \leq R } + d(c_{i}, c_{j}) + \underbrace{ d(c_{j}, w) }_{ \leq R } \\
 & \leq 2R + d(c_{i}, c_{j}) \\
 & \leq 2R + \Delta_{T}
\end{align}
$$

Then:
$$
\Delta_{T} \leq \Delta \leq \Delta_{T} + 2R \implies \text{The distance between } \Delta_{T} \text{ and } \Delta \text{ is at most } 2R
$$

This proves that the quality of the approximation can get arbitrarily close by increasing the number of clusters. In any case with even a small increase in $k$, the quality of the approximation increases significantly.

In practice the coreset $T$ computed by using FFT is a very good euristic for the problem.

### Diversity Maximization

Given a dataset $P$, determine the _most diverse_ subset of size $k$, for a given small $k$.

It is used in several fields where we aim to pick things that are far apart from each other in the solution, for instance to maximize diversity in a web query, or to select facility locations in order to minimize competition between them.
#### Max-Sum Diversification

One of the possible diversity functions is _max-sum_.
Given a set $P$ of points from a metric space $(M, d)$ and a positive integer $k < | P|$, return a subset $S \subset P$ of $k$ points ($|S| = k$), which maximises the _diversity function_:
$$
div(S) = \sum_{x, y \in S}d(x, y)
$$

This problem formulation is **NP-hard**, a c-approximation algorithm is known with $c = 2-\frac{2}{k}$; however it requires _quadratic time_ (which is again unfeasible for large inputs).

Even by modifyng the objective function the core problem still remains NP-hard.

### Coreset-based Diversity Maximization

For a given input $P$, define the value of the optimal solution as:
$$
div^{opt}(P, k) = \max_{S \subset P, |S| = k}div(S)
$$

How do we find a good quality coreset $T$ for the problem?

#### $(1 + \epsilon)$-coreset

Let $\epsilon \in (0, 1)$ be an _accuracy parameter_. A subset $T \subset P$ is an $(1 + \epsilon)$-coreset for the diversity maximization problem on $P$ if:
$$
div^{opt}(T, k) \geq \frac{1}{1 + \epsilon}div^{opt}(P, k)
$$

That is, the approximation factor of the solution $S$ to the problem on $T$ is between $0.5$ and $1$ compared to the optimal one for the original problem.
In other words, if $S$ is a c-approximation solution to diversity maximization in $T$, then $S$ will be a $c(1+ \epsilon)$-approximation solution to diversity maximization in the original set $P$.

And, if $T$ is _sufficiently small_, we can afford to run a quadratic (or even exponential) algorithm on it.

Usually, by selecting a very small $\epsilon$ we will obtain a large coreset $T$, thus it is important to have a reasonable tradeoff between accuracy and complexity.

##### k-center relation with Diversity Maximization
The following fact establishes a relation between k-center and diversity maximization:
$$
\Phi^{opt}_{k-center}(P, k) \leq \frac{div^{opt}(P, k)}{\binom{k}{2}}
$$

This gives us a way to compute a suitable coreset for the diversity maximization problem with bounded accuracy.

#### Coreset-based Algorithm for Diversity Maximization

Given $P$ and $k$:
1. Run FFT to extract a set $T'$ of $h > k$ centers from $P$, for a suitable value $h > k$.
2. Consider the $h$-clustering induced by $T'$ on $P$ and select $k$ arbitrary points from each cluster (all points if the cluster has less than $k$ points)
3. Gather the _at most_ $h\cdot k$ points selected from the $h$ clusters into a coreset $T$, and extract the final solution $S$ by running the sequential algorithm for diversity maximization on $T$

The idea is that by clustering the original pointset and extracting points from the clusters, we obtain a set of points that _proxies_ (i.e. represents) the points of the optimal solution. If we can show that these proxies are good representatives for the optimal solution, then the approximation is correct.

##### Analysis of Correctness

The idea is to show that, by using the described approach we can obtain a _coreset_ of good _proxies_ to the optimal solution, i.e. a coreset of points that represents well the points in the optimal solution.

Let $S^{\star} \equiv$ the optimal solution to diversity maximization on $P$.

We will show that each $x \in S^{\star}$ has a _nearby proxy_ $\pi(x) \in T$ such that by substituting $x$ with its proxy $\pi(x)$ the value of the objective function computed on the coreset does not change much (i.e. the quality of the approximation is bounded).

Let
- $h$ be the number of clusters found by FFT
- $C_{t} \equiv$ the cluster induced by $c_{t}$
- $C = \{ c_{1}, \dots, c_{h} \}$ be the $h$ centers found in the initial FFT execution
- $T \equiv C \cup \{ k-1 \text{ additional points from each } C_{t} \}$ the coreset.

Now for each cluster $C_{t}$ we define an **injective mapping** $\pi(\cdot)$  between $S^{\star } \cap C_{t}$ and $T \cap C_{t}$:
$$
\pi: S^{\star} \cap C_{t} \to T \cap C_{t}
$$
This is possible since $|S^{\star} \cap C_{t}| \leq |T \cap C_{t}|$ by construction of $T$: since $S^{\star} \cap C_{t}$ has at most $k$ points (only if $S^{\star} = C_{t}$), while $T \cap C_{t}$ has exactly $k$ (this is always assuming that $C_{t}$ has at least $k$ point to select, but the observation remains valid even if it has not).

Let $R = \max_{x \in P} d(x, \{ c_{1}, \dots, c_{h} \})$ the radius of the $h$-clustering induced by FFT.
Since the mapping $\pi$ is internal to the $C_{t}$ cluster for any $t$, then $\forall x \in S^{\star}\quad d(x, \pi(x)) \leq 2R$. So the proxies are _at most_ $2R$ away from the points they are representing.

We will now show that $S = \{ \pi(x) : x \in S^{\star} \} \subseteq T$ is a good solution to diversity maximization on $P$:
$$
\begin{align}
div(S) & = \sum_{x, y \in S^{\star}}d(\pi(x), \pi(y)) \\
 & \geq \sum_{x, y \in S^{\star}}[d(x, y) - \underbrace{ d(x, \pi(x)) }_{ \leq2R } - \underbrace{ d(y, \pi(y)) }_{ \leq2R }] \text{ by triangle inequality}  \\
 & \geq \sum_{x, y \in S^{\star}} [d(x, y) - 4R] \\
 & = \sum_{x, y \in S^{\star}} d(x, y) - \binom{k}{2}4R \text{ because there are pairs drawn from } k \text{ points} \\
 & = div^{opt}(P, k) - \binom{k}{2}4R
\end{align}
$$

Now by making $h$ sufficiently larger than $k$ we can make the cluster radius small: $4R \leq \epsilon \Phi^{opt}_{k-center}(P, k)$, and by the [[#k-center relation with Diversity Maximization|fact]] shown before, we have that:
$$
\binom{k}{2}4R \leq \binom{k}{2}\epsilon \Phi^{opt}_{k-center}(P, k) \leq \epsilon div^{opt}(P, k)
$$

By plugging this inequality in the above derivation:
$$
div(s) \geq (1-\epsilon)div^{opt}(P, k) = \frac{1}{1 + \epsilon'}div^{opt}(P, k)
$$
for some $\epsilon'$ close to $\epsilon$.

In conclusion, if the clustering of $P$ used to compute $T$ is good (i.e. $4R \leq \epsilon \Phi^{opt}_{k-center}(P,k)$), then $T$ contains a solution $S$ such that:
$$
div(S) \geq \frac{1}{1+\epsilon'}div^{opt}(P, k)
$$

Hence $T$ is a $(1+\epsilon')$-coreset for diversity maximization on $P$. To get closer to the desired accuracy $R$ has to decrease $\implies h$ has to increase (this is obvious since we would be picking more representatives).

### k-means / k-medians

Let's briefly review the definition of the two problems:
Given a _poinset_ $P$ of $N$ points from a metric space $(M, d)$, determine a set $S \subset P$ of $k$ centers which minimizes:
- $\Phi_{k-means} = \sum_{x \in P}d^{2}(x, S)$ (k-means)
- $\Phi_{k-medians} = \sum_{x \in P}d(x, S)$ (k-means)

Depending on the application the requirement that $S \subset P$ may be lifted, so that the centers could allowed to be any point in $M$ (this is usually the case).
k-means in practice is a _simpler_ problem to solve in terms of computational complexity.

#### Lloyd's Algorithm (k-means)

The Lloyd's algorithm is an efficient k-means clustering algorithm, that can only be used when:
- $M = \mathbb{R}^D$ (Euclidean space)
- $d(\cdot, \cdot)$ is $L_{2}$ (Euclidean distance)
- Centers can be selected outside of $P$

If the initial centers are selected well it provides _good solutions_, but if not it may be trapped in local minima which are not optimal.
It is an iterative algorithm that improves the estimate across iterations, usually a bound to the number of iterations is provided to cover cases of slow convergence.

It is suitable to be executed on massive input instances, provided a distributed implementation. However it is not suitable for _data streams_.

#### k-means++

k-means++ is a _randomized_ algorithm used to solve the k-means or k-medians problem. Alone it provides an $\alpha = \Theta(\log k)$ approximation solution.
Additionally it enforces the centers to belong in $P$ (i.e. $S \subset P$).

The standard formulation only offers guaranteed for Euclidean spaces and distances.

It is very fast, but in practice it is not used alone due to its not great accuracy. It is very often used in the initialization phase for [[#Lloyd's Algorithm (k-means)]] to select the initial centers.

It can be easily implemented in a distributed settings (where the k-means|| variant is very common), but it is not suitable for data streams.

```pseudo
\begin{algorithm}
\caption{k-means++ clustering algorithm}
\begin{algorithmic}
\Procedure{k-means++}{$P$}
	\State $c_1 \gets $ random point drawn from $P$ with uniform probability
	\State $S \gets \{ c_1\}$
	\For{$2 \leq i \leq k$}
		\ForAll{$x \in P \setminus S$}
			\State $\pi(x) \gets d^2(x, S)/\sum_{y \in P \setminus S} d^2(y, S)$
			\State $c_i \gets$ random point in $P \setminus S$ according to distribution $\pi$
			\State $S \gets S \cup \{c_i\}$
        \EndFor
    \EndFor
	\Return S
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

#### Partitioning Around Medioids (k-medians)

This algorithm finds an approximate solution to the k-medians problems, and it works by local search, starting from an arbitrary solution $S$ and progressively improving it by performing the "best swap" between a point in $S$ and a point in $P \setminus S$, until no improving swaps exisits.

It enforces $S \subset P$, and provides good quality solutions.

Unfortunately it is _very slow_ (every iteration requires $O(N\cdot k)$ comparisons and convergence is very slow) and thus it is not suitable neither for very large datasets nor data streams.

#### Challenges

This type of clustering remains very challenging when:
- The dataset is really large
- The points comes from a general metric space (not Euclidean)
- We need very accurate results, especially in _high-dimensional_ spaces.
- We need to process possibly unbounded data streams

#### Coreset-based Approach

As done previously for [[#MapReduce Farthest-First Traversal|k-center]] we will formulate a _composable coreset technique_ approach that will allow us to develop algorithms for k-means/medians that are able to both:
- handle massive data
- provide good accuracy

The general approach is to add a _weight_ measure to the representatives, in order to capture how many points they are representing from the original set. This is important for k-means (medians), since the objective function is an aggregation of distances and not only the minimum as it was for the k-center problem.
Thus the contribution of every representative depends also on how many points it represents.

![[coreset-kmeans-weights.png]]

The general steps are:
1. Each point $x_{i} \in T_{i}$ is given a weight $w(x)$ = the number of points in $P_{i}$ for which $x$ is the closest representative in $T_{i}$.
2. The local coresets $T_{i}$ are computed using a _sequential algorithm_ for k-means/medians
3. The final solution $S$ is obtained by running a _weighted sequential algorithm_ on the union of the local coresets $T = \bigcup_{i \in L}T_{i}$.

![[coreset-kmeans.png|center]]

#### Weighted k-means clustering

The following sections will focus on k-means, but with minor adaptations everything remains valid for k-medians too.

k-means is generally easier to solve, since we have valid approximation algorithms (k-means++ and Lloyd), that yield good results and are relatively fast to converge. Lloyd is not applicable to k-medians, since the method it uses to compute the centers only works for squared distances on Euclidean spaces.

**Input**:
- Pointset $P$ of $N$ points $\in \mathbb{R}^{D}$
- Integer _weight_ $w(x) > 0\ \forall x \in P$
- Target number of clusters $k$

**Output**: Set $S$ of $k$ centers in $\mathbb{R}^{D}$ minimizing the objective function:
$$
\Phi^{w}_{k-means}(P, S) = \sum_{x \in P} w(x)\cdot d^{2}(x, S) \text{ (weighted variant of k-means)}
$$

This formulation allows for centers $\in \mathbb{R}^{d}$ outside of the pointset $P$.
If the points all have unitary weights we obtain the original k-means formulation.

##### Weighted Lloyd's Algorithm

It is straightforward to modify [[#Lloyd's Algorithm (k-means)]] to work with weightes points, it is sufficient to modify the update step so that, for every cluster approximation $C_{t}$ with $m$ points:
$$
\frac{1}{m}\sum_{i = 1}^{m} x_{i} \to \frac{1}{\sum_{i=1}^{m}w(x_{i}) }\sum_{i=1}^{m} w(x_{i})^{2}(x_{i})
$$
that is we use a weighted mean in place of the arithmetic one.

##### Weighted k-means++

Also for [[#k-means++]] the modifications required are straightforward, it is sufficient to change the probability distribution $\pi$ calculation in the loop, such that it takes the weights into account:
$$
\pi(x) \to \frac{d^{2}(x, S)}{\sum_{y \in P \setminus S}d^{2}(y, S) } \to \frac{w(x)\cdot d^{2}(x, S)}{\sum_{y \in P \setminus S}w(y)\cdot d^{2}(y, S) }
$$

#### Coreset-based MR Algorithm for k-means

We will now provide a MapReduce algorithm for k-means approximation. It requires two sequential algorithms:
- $\mathcal{A}_{1}$ that solves the standard (unweighted) k-means variant
- $\mathcal{A}_{2}$ that solves the weighted variant

Both the algorithms need to require space _proportional to the input size_.
Clearly $\mathcal{A}_{2}$ could be used for both steps by assigning weight one to all the points.

##### Description
**Input**: Set $P$ of $N$ points in $\mathbb{R}^{D}$, number of target clusters $k > 1$, sequential k-means algorithms $\mathcal{A}_{1}, \mathcal{A}_{2}$ as specified.

**Output**: Set $S$ of $k$ centers $\in \mathbb{R}^{D}$ which is a "good" solution to the k-means problem on $P$ (centers are not required to be $\in P$).

**Round 1**:
- _MapPhase_: partition $P$ into $l$ subset of equal size $P_{1}, \dots, P_{l}$.
- _ReducePhase_: for every $i \in [1;l]$ separately run the unweighted sequential algorithm $\mathcal{A}_{1}$ on $P_{i}$ to determine a set $T_{i} \subseteq P_{i}$ of $k$ centers, and define:
	- For each $x \in P_{i}$ its proxy $\pi(x)$ as its closest center in $T_{i}$
	- For each $y \in T_{i}$ its weight $w(y)$ as the number of points in $P_{i}$ that are represented by $y$ (i.e $|\{x: \pi(x) = y\}|$)
	Note that the proxies **must not be stored**, instead the weigths **must be stored**.
	
**Round 2**:
- _MapPhase_: empty
- _ReducePhase_: run, collecting all the $T_{i}$ in a single reducer, the weighted algorithm $\mathcal{A}_{2}$ on $T = \bigcup_{i = 1}^{l}T_{i}$, with the weights assigned during the previous round. This determines the solution $S = \{ c_{1}, \dots, c_{k} \}$ given in output.

![[coreset-mr-kmeans.png]]

##### Space Analysis

Assume $k \in o(N)$, by setting $l = \sqrt{ \frac{N}{k} }$ it is easy to see that MR-kmeans($\mathcal{A}_{1}, \mathcal{A}_{2}$) requires:

- **Local space**: $M_{L} = O\left( \max{\frac{N}{l}, l\cdot k} \right) = O(\sqrt{ N\cdot k }) = o(N)$ 
- **Aggregate space**: $M_{A} = O(N)$

for the same reasoning used in [[#MapReduce Farthest-First Traversal#Space Complexity|MR FFT space analysis]].

##### Accuracy Analysis

In order to analyze the quality of the solutions computed by MR-kmeans we need to introduce the following notion:

>**Definition ($\gamma$-coreset)**:
Given a pointset $P$, a coreset $T \subseteq P$, and a proxy function $\pi: P \to T$, we say that $T$ is a _$\gamma$-coreset_ for $P, k$ and the k-means objective if:
>
>$$
>\sum_{p \in P}d^{2}(p, \pi(p)) \leq \gamma\cdot \Phi^{opt}_{k-means}(P, k)
>$$


![[coreset-y-coreset.png]]

We will show that the coreset used in the algorithm is a $\gamma$-coreset for the original problem.
Ideally we would like to have a small coreset $T$, so that the algorithm runs faster and in less space, and also to have a small $\gamma$ so that the error is low.

The following theorem (the proof is not provided), gives us the quality of MR-kmeans:
> **Theorem**:
> Suppose that:
> - $\mathcal{A}_{1}$ is a $\gamma$-approximation algorithm for the _unweighted_ k-means problem
> - $\mathcal{A}_2$ is an $\alpha$-approximation algorithm for the _weighted_ k-means problem
>
> Then:
> - The coreset computed in Round 1 is a $\gamma$-coreset
> - The solution $S$ computed in Round 2 is such that:
> $$
>\Phi_{k-means}(P, S) = O((1+\gamma)\cdot\alpha)\cdot \Phi^{opt}_{k-means}(P, k)
>$$

Intuitively the formula for the approximation factor shows two contributions:
1. From the fact that the subset is smaller than $P$: $T \subseteq P$, from which the term $(1+\gamma)$ comes.
2. From the approximation factor of the final algorithm $\mathcal{A}_{2}$, which contributed with the $\alpha$ term.

##### Final Observations

For k-means (as well as k-center and k-medians) good _coresets_ are obtained by combining solutions to the same problem on _smaller partitions_. However this is **not always the case** (e.g. diameter estimation, diversity maximization).

The MR algorithm explained above is in practice both _fast_ and _accurate_. Typically [[#k-means++]] is used in the first round as $\mathcal{A}_{1}$, and a combination of [[#Weighted k-means++]] and [[#Weighted Lloyd's Algorithm]] is used as $\mathcal{A}_{2}$ in round 2.

It is possible to arbitrarily increase the accuracy of the final solution by increasing the number of representatives included in the coreset $T$. To do so we could run the k-means algorithm on the partitions with a higher number of target centers $h > k$.
This tends to reduces quite significantly the value of $\gamma$, improving the quality of the final solution.