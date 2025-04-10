### Definition

Given a set of _points_ belonging to some _space_ with a notion of _distance_ between them, _clustering_ aims at partitioning the space into subsets of points called _clusters_, such that, according to some _metric_:
- Points in the same cluster are _close_
- Points in different clusters are _distant_ from one another

The notion of _distance_ captures the similarity concept between points.

A _clustering problem_ is usually defined by requiring that clusters optimize a certain _objective function_, with potentially additional properties (constaints).

Generally clustering is an NP-hard problem! This means we will very often use approximation algorithms to solve it.

![[coreset-clustering-exa.jpeg]]

### Importance and Applications

Clustering has the main purposes of:
1. Assessing the structure of data
2. Grouping data for further prediction tasks
3. Summarizing and compressing data

Applications are very wide in different fields, for example:
- _Marketing_: group together potential customers by certain properties
- _Biology_: identify protein families
- _Image processing_: perform object recognition
- _Information retrieval_: perform document categorization
- _Network analysis_: locate and identify communities
- _Facility location_: decide where to open new hubs, based on the clusters obtained

### Metric Spaces

Typically the input of a clustering problem is a set of points belonging to a _metric space_.

#### Definition

A _metric space_ is an ordered pair $(M, d)$, where 
- $M$ is a _set_
- $d: M \times M \to \mathbb{R}$ is a _metric_ (i.e. a distance function) on $M$ for which, for all $x, y, z \in M$:
	$$
		\begin{cases}
		d(x, y) \geq 0 & \text{the distances cannot be negative} \\
		d(x, y) = 0 \iff x = y \\
		d(x, y) = d(y, x) & \text{distances are symmetric} \\
		d(x, y) \leq d(x, z) + d(z, y) & \text{triangle inequality}
		\end{cases}
	$$

Note that the choice of the metric $d$ has a significant impact on the effectiveness of cluster-based analysis (i.e. if the metric does not fit well the problem at hand, analysis becomes useless).

### Distance Functions

#### Minkowski Distances (L-norm)

Let $\vec{x}, \vec{y}\in \mathbb{R}^n$ , with $\vec{x} = (x_{1}, x_{2}, \dots, x_{n})$ and $\vec{y} = (y_{1}, y_{2}, \dots, y_{n})$. For $r \in \mathbb{N}_{+}$, the $L_{r}$-norm is defined as:
$$
d_{L_{r}}(\vec{x}, \vec{y}) = \left(\sum_{i=1}^{n}{|x_{i} - y_{i}|^{r}}\right)^{1/r}
$$

This is a generalization from which common distances are derived:
- $r = 2 \implies d_{L_{2}}(\vec{x}, \vec{y}) = \sqrt{ \sum_{i=1}^{n}|x_{i} - y_{i}|^2}$ (Euclidean distance) 
- $r = 1 \implies d_{L_{1}}(\vec{x}, \vec{y}) = \sum_{i=1}^{n}|x_{i} - y_{i}|$ (Manhattan distance) 
- $r = +\infty \implies d_{L_{+\infty}}(\vec{x}, \vec{y}) = \max_{i=1}^{n}|x_{i} - y_{i}|$ (Chebyshev distance) 

![[coreset-distances-exa.jpeg]]

From the comparison image below we see how choosing different distances greatly affects the properties of the clusters formed.

![[coreset-distances-comp.jpeg]]

These metrics aggregate the gap in each dimension for points (in different ways). GPS coordinates are a natural example of this.

#### Angular Distances

Given $\vec{x}, \vec{y} \in \mathbb{R}^n$, their _angular distance_ is the angle between them:
$$
d_{angular}(\vec{x}, \vec{y}) = \arccos\left( \frac{\vec{x}\cdot \vec{y}}{||\vec{x}||\cdot ||\vec{y}||} \right) \in [0, \pi]
$$
which represents the projection of $\vec{x}$ over $\vec{y}$, scaled by their lengths.

To normalize values in the range $[0, 1]$, $d_{angular}$ is usually scaled by $\pi$ or $\frac{\pi}{2}$, depending whether vectors have arbitrary coordinates or only non-negative.

Additionally, in order to satisfy the second proper ($d(\vec{x}, \vec{y}) = 0 \iff \vec{x} = \vec{y}$), scalar multiplies of the same vector are considered the same. Thus the space could be reduced by applying to all the vectors a scaling proportional to their norm: $\vec{\dot{x}} = \frac{\vec{x}}{||\vec{x}||}$.

This distance is often used in information retrieval to measure ratios among values that may have different lengths, but where we are interested in relative measures (e.g. frequencies).


#### Hamming Distance

Let $\vec{x}, \vec{y} \in \{ 0, 1 \}^n$ (i.e. are binary sequences of length $n$). The _Hamming distance_ is the number of dimensions in which they differ:
$$
d_{Hamming}(\vec{x}, \vec{y}) = \sum_{i=1}^{n}|x_{i} - y_{i}| 
$$

It is only a special name given to the $L_{1}$-norm for binary vectors.

#### Jaccard Distance

Let $S, T$ be sets belonging to the same universe. The _Jaccard distance_ is the ratio of elements not in common between the two:
$$
d_{Jaccard}(S, T) = 1 - \frac{|S \cap T|}{|S \cup T|} = \frac{|S \cup T| - |S \cap T|}{|S\cup T|} \in[0, 1]
$$

The distance is $0 \iff S = T$  and $1 \iff S, T$  are disjoint.

![[coreset-jaccard-exa.jpeg]]

#### Distance to a Subset

Given any metric space $(M, d)$ and a set $P \subseteq M$, we can also define the distance of a point $x \in P$  to a subset $S \subseteq P$:
$$
d(x, S) = \min_{y \in S}d(x, y)
$$
that is the distance between $x$ and the closest point of $S$.

#### What should we use?

- When objects are characterized by the _numerical values_ of their features:
	- _Minkowski distance_ when we need to aggregate gaps between objects.
	- _Angular distance_ when we need to measure ratios among features, instead of absolute values
- When objects are characterized by the _presence_ of certain features:
	- _Hamming distance_ when we are interested in the _total number_ of differences
	- _Jaccard distance_ when we are interested in the _ratio of differences_ between two sets and their cardinality

### Types of Clustering

As mentioned in the [[#Definition]], a clustering problem includes an _objective function_ to optimize. Different functions lead to different solutions, hence it has to be chosen with the problem in mind. 

Thus clustering is _de-facto_ an optimization problem, for further explanation see [[Optimization Problem]].

Clustering problems can be categorized based on whether or not:
1. The target number of clusters $k$ is given as input
2. A _center_ for each cluster has to be identified
3. The clusters in output must be _disjoint_

#### Center-based Clustering

Let $P$ be a set of $N$ points in a metric space $(M, d)$, and let $k$ be the target number of clusters.
We define a _$k$-clustering of $P$_ as a tuple $\mathcal{C} = (C_{1}, \dots, C_{k})$ where:
- $(C_{1}, \dots, C_{k})$ defines a _partition_ of $P$, that is $P = \bigcup_{i=1}^kC_{i}$ and $\forall i, j; i\neq j: C_{i} \cap C_{j} = \emptyset$
- $c_{1}, \dots, c_{k}$ are suitably selected _centers_ for the clusters where $c_{i} \in C_{i}\ \forall i$

Depending on the choice of the objective function we obtain different clusters. The most common ones are:

- **k-center** clustering: minimize the _maximum distance_ from any point in the cluster to its center
![[coreset-kcenter-exa.png]]
- **k-means** clustering: minimize the _sum of squared distance_ for all the points in the cluster from the center
- **k-median** clustering: minimize the _sum of distances_ of all the points in the cluster from the center
![[coreset-kmeans-exa.png]]

Observe that for k-means and k-medians, minimizing the (squared) distances is equivalent to minimizing the average (squared) distance.

Formally:
> Let $(M, d)$ be a metric space. The _k-center_ / _k-means_ / _k-median_ clustering problems are optimization problems that, give a finite pointset $P \subseteq M$ and an integer $k \leq |P|$, require to return the subset $S \subseteq P$ of $k$ _centers_ which minimizes the following objective functions:
> - $\Phi_{k-center}(P,S) = \max_{x \in P}(d(x, S))$ (**k-center clustering**)
> - $\Phi_{k-means}(P,S) = \sum_{x \in P}d^2(x, S)$ (**k-means clustering**)
> - $\Phi_{k-median}(P,S) = \sum_{x \in P}d(x, S)$ (**k-median clustering**)

Notice how the problem only requires to return the centers and not the whole clustering solution. This is because it is trivial, given the centers, to decide where a point belong.
For any $x \in P$, the primitive that computes the cluster $i$ to which the point belongs is `Assign(P, S)`:
$$
\text{Assign}(P, S) = i : c_{i} = \arg\min_{c \in S}d(x, c)
$$

This can be executed sequentially in $O(N\cdot k)$ time. In MapReduce a 1-round algorithm can be executed with local space $M_{L} \in O(k)$ and aggregate space $M_{A} \in O(N)$

##### Observations

As mentioned before center-based clustering in **NP-hard**, this means it is infeasible to search for the optimal solution $S^\star$.

Fortunately there are several efficient approximation algorithms that in practice return good-quality solutions. However dealing with very large input sets is still a problem.

k-center is more useful when we need to guarantee that _every_ point is close to a center. It tends to be sensitive to noise in the input set.

k-means and k-median are more relaxed compared to k-center in that they only give guarantees on the average (square) distances. Compared to k-median, k-means is more sensitive to noise but it is faster to execute.

### k-center Clustering

#### Farthest-First Traversal

_Farthest-First Traversal_ (later referred to as FFT), is a popular 2-approximation sequential algorithm developed to obtain a good approximate solution to the _k-center_ algorithm.
It is simple and relatively fast when the entire input fits in main memory, for this reason it is often used as a primitive for data analysis.

**Input**: Set $P$ of $N$ points from a metric space $(M, d)$, and the target number of clusters $k \in \mathbb{N} > 1$.
**Output**: A set $S$ of $k$ centers which is a good solution to the k-center problem $P$, as we will prove $\Phi_{k-center}(P, S) \leq 2\Phi_{k-center}^{opt}(P, k)$

```pseudo
\begin{algorithm}
\caption{Farthest-First Traversal}
\begin{algorithmic}
\Procedure{FarthestFirstTraversal}{$P, k$}
	\State $S \gets \{c_1\}$
	\Comment{$c_1 \in P$ arbitrarily chosen}
	\For{$i \gets 2$ to $k$}
		\State Find the point $c_i \in P \setminus S$ that maximizes $d(c_i, S)$
		\State $S \gets S \cup \{c_i\}$
    \EndFor
	\Return $S$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

Intuitively this algorithm iteratively picks a new center furthest away from the current centers, thus greedily maximizing spacing between centers.

![[coreset-fft-exa.png]]

##### Complexity
The FFT algorithm can be implemented in $O(N\cdot k)$ time.

##### Analysis

Let's now prove that the solution returned by FFT is no worse than 2 times the optimal solution, i.e. :
$$
\Phi_{k-center}(P, S) \leq 2\cdot \Phi_{k-center}^{opt}(P, k)
$$
In other words FFT is a _2-approximation_ algorithm.

The idea is to leverage the FFT property of picking the farthest point. If we pick an additional point $k+1$ it will be the farthest from our solution of centers by construction of FFT. Then, by considering the optimal $k$ clusters, there will surely be two points in our "augmented" solution that will belong to the same optimal cluster. The maximum distance between those is clearly _at most_ twice the optimal maximum distance. This would prove that a point can be _at worst_ be twice as far from a center compared to the optimal solution. 

**Proof**:

Let $S = \{ c_{1}, \dots, c_{k} \}$ be the solution returned by FFT, with $c_{i} = i$-th center.

Define $q = \max_{x \in P}d(x, S)$, that is $\forall x \in P, d(q, S) \geq d(x, S)$.

Now consider the set $S' = \{ c_{1}, \dots , c_{k}, c_{k+1} = q \}$, which is the solution that FFT would have returned for $k' = k+1$.
$c_{k+1} = q$ comes by construction of the FFT algorithm.

We now will show that:
$$
d(c_{i}, c_{j}) \geq d(q, S)\quad \forall 1\leq i < j \leq k+1
$$
i.e. the minimum distance between any two centers is at least as large as the maximum distance from any point to its nearest center.

Fix now some arbitrary $i, j$ such that $1\leq i < j \leq k+1$.
Then:
$$
\begin{align}
d(q, S) & = d(q, \{ c_{1}, \dots, c_{k} \}) \\
 & \leq d(q, \{ c_{1}, \dots, c_{j-1} \})  &  \text{ since } j-1\leq k \\
 & \leq d(c_{j}, \{ c_{1}, \dots, c_{j-1} \}) &  \text{since FFT selects }c_{j} \text{ as the farthest} \\
 & \leq d(c_{j}, c_{i}) & \text{since } i < j \implies c_{i} \in \{ c_{1}, \dots, c_{j-1} \}
\end{align}
$$

Let now $S^\star = \{ c_{1}^\star, \dots, c_{k}^\star \}$ be the optimal centers, so
1. $\Phi_{k-center}^{opt}(P, k) = \Phi_{k-center}(P, S^\star) \leq \Phi_{k-center}(P, S)$
2. $\forall x \in P\quad d(x, S^\star) \leq \Phi_{k-center}^{opt}(P, k)$

Define $C_{t}^\star = \{x \in P : c_{t}^\star \text{ is the center of } S^\star \text{ closest to } x \}$
Thus $P = \bigcup_{i=1}^kC_{i}^\star$

By the pigeonhole principle there must be a pair of centers $c_{a}, c_{b}$ in the sub-optimal augmented solution $S'$ that are both in the same optimal cluster $C_{t}^\star$, since $S'$ is a $k+1$ clustering solution, while $S^\star$ has $k$ centers.


Note that, by how we defined $c_{a}$ and $c_{b}$ they both belong to $C_{t}^\star \implies d(c_{a}, c_{t}^\star) \leq \Phi_{k-center}^{opt}(P, k), d(c_{b}, c_{t}^\star) \leq \Phi_{k-center}^{opt}(P, k)$.
At the end, by the triangle inequality we obtain:
$$
\begin{align}
d(q, S) \leq d(c_{a}, c_{b})  & \leq d(c_{a}, c_{t}^\star) + d(c_{t}^\star, c_{b}) \\
 & \leq 2 \cdot \Phi_{k-center}^{opt}(P, k) \\
\end{align}
$$

##### Observations
- The k-center objective $\Phi$ focuses on worst-case distance of points from their _closest center_
- Farthest-First traversal's approximation guarantees are almost the best one can obtain in practice. It was proved formally that computing a c-approximate solution to k-center is _NP-hard for any $c<2$_.
- The k-center objective is _very sensitive to noise_. For noisy datasets that contain _outliers_, the clustering imposed by k-center may obfuscate the real clustering inherent in the data, as shown in the image below
![[coreset-fft-noise.png]]

### k-means/medians Clustering

Let's review the two problems:

Given a pointset $P$ of $N$ points from a _metric space_ $(M, d)$, determine a set $S \subset P$ of $k$ centers which minimizes:
$$
\begin{align}
\Phi_{k-means}(P, s) & = \sum_{x \in P} d(x, S)^{2} & \text{k-means} \\
\Phi_{k-median}(P, s) & = \sum_{x \in P} d(x, S) & \text{k-median}
\end{align}
$$

Depending on the application the requirement that $S \subset P$ may be lifted, and centers are allowed to be in $M$ and not in $P$.

#### Lloyd's Algorithm


