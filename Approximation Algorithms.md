As we have seen in the [[Complexity Classes and NP-hardness]] section, there are problems that are (currently) unsolvable in polynomial time.

Thus we need to find a way to compute _approximate solutions_ for them in polynomial time, that have a bound on the quality of the approximation (or they would be quite useless).

We will assume that $P \neq NP$.

### Optimization Problems

For a more in depth discussion of the general framework refer to [[Optimization Problem]].

To briefly recap an _approximation problem_ $\Pi$ is a relation between the set of possible input _instances_ $I$, and the set of _feasible solutions_ $S$:
$$
\Pi: I\times S
$$

We define for every solution $s \in S$ a _cost_ $c$:
$$
c: S \to \mathbb{R}^+
$$

For every instance of the problem $i \in I$, we define its _feasible solutions_:
$$
\forall i \in I\quad S(i) = \{ s \in S : s \text{ is a solution to } \Pi \text{ for } i \}
$$

We lastly define the _optimal solution_ $s^\star$ as the _minimum (maximum) cost_ feasible solution:
$$
s^\star(i) \in S(i) = \arg\min_{s \in S(i) } c(s) \text{ (same for max)}
$$

### Approximation

For an approximation of an optimization problem over $i \in I$ we will have $s \geq s^\star$ ($\leq$). It is fine to us but, as mentioned in the introduction, we need:
- guarantees on the **quality** of $s$.
- guarantees on the **complexity** of the algorithm that computes $s$ (polynomial)

#### Approximation Factor
**Definition**: Let $\Pi$ be an optimization problem and let $A_{\pi}$ be an algorithm for $\Pi$ that returns, $\forall i \in I$, a feasible solution to it: $A_{\pi}(i) \in S(i)$. 

We say that $A_{\pi}$ has an _approximation factor_ of $\rho(n) \geq 1$ if $\forall i \in I, |i|=n$, we have:
$$
\begin{align}
\text{minimization}: & \frac{c(A_{\pi}(i))}{c(s^\star(i))} \leq \rho(n) & \text{at most } \rho(n) \text{ times bigger than the optimal} \\
\text{maximization:} & \frac{c(s^\star(i))}{c(A_{\pi}(i))} \leq \rho(n) & \text{at most } \rho(n) \text{ times smaller than the optimal}
\end{align}
$$

The goal is to define an upper bound $\rho(n) = 1+\epsilon$ with $\epsilon$ as small as possible, and independent from the size of the input $n$ .

#### Approximation Scheme
**Definition**: An _approximation scheme_ for $\Pi$ is an _algorithm_ with two inputs $A_{\pi}(i, \epsilon)$, that for every $\epsilon>0$ is a $(1+\epsilon)$-approximation to $\Pi$.
An _approximation scheme_ is _polynomial_ (PTAS) if $A_{\pi}(i, \epsilon)$ is polynomial in the size of $i = |i| = n$, for any fixed $\epsilon$.

### Vertex Cover Approximation Algorithms

We have previously seen that [[Complexity Classes and NP-hardness#Vertex Cover|vertex cover is NP-hard ]].
We shall then find a suitable approximation algorithm.

One simple idea would be to use the degree of the vertices as an _euristic_ to which vertex to select first, since we assume that it would cover more of the edges in one pick.

Unfortunately, for this algorithm the approximation factor is quite bad: $\rho(n) = \Omega(\log n)$.
Note that to prove a lower bound on $\rho(n)$ ($\Omega$), it is sufficient to show one bad instance, since by definition the factor $\rho$ has to hold for all possible inputs $i \in I$.

TODO: show bad instance

A better algorithm is the following one:
1. Choose any edge at random
2. Add both its endpoints to the solution
3. Remove the covered edges from the search, so that they don't get covered again
4. Repeat

```pseudo
\begin{algorithm}
\caption{Vertex cover approximation algorithm}
\begin{algorithmic}
\Procedure{ApproxVertexCover}{$G$}
	\State $V' \gets \emptyset$
	\State $E' \gets E$
	\While{$E' \not= \emptyset$}
		\State $(u, v) = e \gets$ random edge from $E'$
		\State $V' \gets V' \cup \{u, v\}$
		\State $E' \gets E' \setminus \{\text{all edges incident to } u \text{ or } v\}$
    \EndWhile
	\Return $V'$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

#### Time Analysis

This algorithm has time complexity of
$$
O(n+m)
$$
since it traverses the graph linearly

#### Approximation Analysis

We will show that this is a 2-approximation algorithm:
$$
 \frac{|V'|}{|V^{\star}|} = 2
$$

First let $A$ be the set of selected edges, we say that $A$ **is a matching** if:
$$
\forall e, e' \in A \implies e \cap e' = \emptyset
$$
that is, there are no vertices in common. 

![[appalg-matching-exa.png]]

We observe that the algorithm selects a **maximal** matching (i.e. the one with most edges):
$$
\forall y \in E, A \cup y \text{ is not a matching}
$$
this comes naturally from the steps of the algorithm.

1. $\frac{|V'|}{|V^{\star}|} \leq 2$

	$A$ is a matching $\implies$ in $V^{\star}$ there must be at least one vertex for every edge $\in A$, or $V^{\star}$ would not cover the graph $G$:
	$\implies |V^{\star}| \geq |A|$.

	By construction of the algorithm that picks two endpoints per edge it means that:
	$\implies |V'| = 2|A|$

	Thus: $\implies |V'| = 2|A| \leq 2|V^{\star}| \implies \frac{|V'|}{|V^{\star}|} \leq 2$.
2. Now we observe very simply that if $A$ is a maximal matching, it means that the optimal solution picks exactly one endpoint per edge in $A \implies |V^{\star}| = |A|$.

	Thus: $|V'| = 2|A| = 2|V^{\star}| \implies$ Approx-Vertex Cover is exactly a $2$-approximation algorithm.

### Travelling Salesperson Problem

**Definition**: given a _complete_, _undirected_ graph $G = (V, E)$, and a positive weight for every edge: $w: E \to \mathbb{R}^{+}$, it outputs a _tour_ (cycle that touches every vertex exactly once) $T  \subseteq E$ that minimizes the total cost $\sum_{e \in T}w(e)$.
We can consider only positive weigths without loss of generality, because every TSP tour has the same number of edges ($n$), so we can add a large _offset_ to every edge such that all the edges have non-negative weights. This does not change the actual solution.

Unfortunately there is **no possible $\rho(n)$-approximation** for TSP (unless $P = NP$).

**Proof**:
We will prove this by showing that, supposing we had a $\rho$-approximate TSP in polynomial time, we could solve [[Complexity Classes and NP-hardness#^c21261|Hamiltonian Circuit]], which is NP-hard.

The reduction is almost identical to the one presented [[Complexity Classes and NP-hardness#Travelling Salesman Problem|here]], except that the weight function becomes:
$$
w(e \in E') = \begin{cases}
1 & \text{if } e \in E \\
\rho\cdot n+1 & \text{otherwise}
\end{cases}
$$
Instead of picking $+\infty$ for $e \not\in E$, we put a "bound" on how far apart the nodes are. In other words $\rho\cdot n+1$ becomes a "sentinel" value that tells us that TSP has picked an edge not in $E$ originally.

Now either:
1. $G$ has an Hamiltonian Circuit $\implies$ there is a tour of cost $n \implies$ approx-TSP on $G'$ returns a tour of cost $\leq \rho\cdot n$ (because its a $\rho$ approximation)
2. $G$ has no Hamiltonian Circuit $\implies$ approx-TSP on $G'$ returns a tour of cost $\geq \rho\cdot n + 1 > \rho \cdot n$, because it will be forced to pick at least one edge not in $E$.

This proves that, if we could approximate TSP withing a factor $\rho$ in polynomial time, we could also solve Hamiltonian Circuit in polynomial time.

#### Metric TSP

While TSP is not $\rho$-approximable in polynomial time (unless $P = NP$), there are some variants that can be.

One of the most common _special case_ is _metric TSP_, in which the weight function satisfies the **triangle inequality**:
$$
\forall u, v, z \in V: w(u, z) \leq w(u, v) + w(v, z)
$$
i.e. it is always faster to go directly instead of passing through another node.

It is called _metric_, because with this additional property it creates a _metric space_ $(V, w)$.

Unfortunately metric TSP **is still NP-hard**.
**Proof**:  TSP $\leq_{P}$ Metric TSP

The idea is pretty intuitive: as we [[#Travelling Salesperson Problem|observed previously]] for TSP to remove negative weights, we could add an offset to all the edges equally, without changing the solution.

If we take the largest weight in the graph's edges, and add it to all of them, we observe that the resulting weight function satisfies triangle inequality.

In fact let's analyze the basic triangular structure with weights $a, b, c$, and suppose $o = \max\{ a, b, c \}$. Then adding $o$ to all the edges we obtain:
$$
\begin{align}
a' = a + o \\
b' = b + o \\
c' = c + o \\
\end{align}
$$
But now for any combination we are guaranteed that they satisfy triangle inequality:
$$
\begin{align}
a' = a+o  & \leq b' + c' = b + c + 2o \\
a + \max\{ a,b, c \}  & \leq b + c + 2\max\{ a, b, c \} \\
a & \leq b + c + \max\{ a, b, c \}
\end{align}
$$
which is clearly true that for any $a \geq 0$ it will be no larger than the max between the three sides plus a non-negative amount. The same argument remains valid by considering the other edges.
This proves that modifying the weights in this way ensures that the weight function **satisfies triangle inequality**.

Now we have to prove that this modification is valid for solving TSP, but this is straightforward.
We observe that, by applying the reduction on $G$ as specified:
$$
<G = (V, E), w, k>\  \mapsto\ <G' = (V, E), w', k'>
$$
with weights updated:
$$
w'(u, v) = w(u, v) + \underbrace{ \max_{u, v \in V}w(u, v) }_{ =W }
$$
we obtain that the solution of cost $k$ in the original problem becomes:
$$
k' = k + nW
$$
since we have added the same offset $W$ in all the edges, and there exactly $n$ edges in any TSP tour in a graph with $n$ vertices.

This implies:
$$
\exists \text{ Ham. Circuit of cost } k \in G \iff \exists \text{ Ham. Circuit of cost } k' \in G'
$$
proving the reduction.

##### A 2-approx Algorithm for Metric TSP

The idea behind an approximation for metric TSP is to use an [[Graphs#Minimum Spanning Tree|MST]], which by definition connects with minimum cost all the vertices, then take shortcuts to avoid passing again through the same edges. The shortcuts are guaranteed to exist because the graph is _complete_. Furthermore, since we are in a **metric space**, going directly to a target vertex through the shortcut surely costs _no more_ than reaching it though an intermediate node.

![[aalg-metrictsp-approx.png]]

The operation of obtaining such _cycle_ from an MST is called **preorder**:
```pseudo
\begin{algorithm}
\caption{Preorder Algorithm}
\begin{algorithmic}
\Procedure{Preorder}{$T, v$}
	\State \Call{Print}{$v$}
	\If{$v$ is internal (not a leaf)}
		\ForAll{$u \in $ \Call{Children}{$v$}}
			\State\Call{Preorder}{$u$}
        \EndFor
    \EndIf
\EndProcedure
\end{algorithmic}
\end{algorithm}
```
This follows a traversal very similar to [[Graphs#Depth First Search Algorithm|DFS]].
![[aalg-preorder-exa.png]]

Then, by appending the root at the end of the list, we obtain an **Hamiltonian circuit** of the original graph.

###### Algorithm

```pseudo
\begin{algorithm}
\caption{Approx Metric TSP Algorithm}
\begin{algorithmic}
	\Procedure{ApproxMetricTSP}{$G$}
		\State $V \gets \{v_1, v_2, \dots, v_n\}$
		\State $r \gets v_1$
		\State $T^\star \gets$ \Call{Prim}{$G, r$}
		\State $H' = <v_{i_1}, \dots, v_{i_n}> \gets$ \Call{Preorder}{$T^\star, r$}
		\Comment{Cycle around the MST}
		\Return $<H^\star, v_{i_1}>$
		\Comment{Add back the starting point to create a cycle}
    \EndProcedure
\end{algorithmic}
\end{algorithm}
```

###### Analysis

As mentioned in the introduction, the intuition behind is that we start from a MST that connects all vertices with minimum cost, and then by relying on the fact that we have a metric space, we know that (by triangular inequality), the shortcuts will not increase the cost.

1. **Lower bound to the cost of** $H^{\star}$.
	We know that $H^{\star}$ is a tour with $n$ edges. Now let $T'$ be a tree obtained by removing one edge from $H$. Since $T^{\star}$ is a MST with $n-1$ edges (because its a tree), There cannot be a tree of $n-1$ edges with lower cost than $T^{\star}$ (or it would not be an MST), hence:
	$$
		w(H^{\star}) \geq w(T') \geq w(T^{\star})
	$$
2. **Upper bound to the cost of** $H^{\star}$.
	We start by defining a **full preorder chain**, which is a list with repetitions of the vertices of the tree built by adding in order the vertices reached from the recursive calls of _preorder_
	![[aalg-fullpreorder-exa.png]]
	By construction we observe that:
	$$
	w(fpc) = 2w(T^{\star})
	$$
	since every edge of $T^{\star}$ appears _exactly_ twice in a fpc.

	Now, by applying the shortcuts, and using the triangular inequality (because the space is metric), we prove the 2 approximation:
	$$
	\begin{align}
	2w(T^{\star}) & =w(<a, b, c, b, d, b, a, e, a>) \\
	& \geq w(<a, b, c, \cancel{b}, d, b, a, e, a>) \\
	& \geq w(<a, b, c, d, \cancel{b}, \cancel{a}, e, a>) \\
	& \geq w(<a, b, c, d, e, a>) \\
	& \implies 2w(T^{\star}) \geq w(H)
	\end{align}
	$$

Now, putting pieces together:
$$
\begin{align}
 & w(H^{\star}) \geq w(T^{\star}) \\
 & 2w(T^{\star}) \geq w(H) \\
 & \implies 2w(H^{\star}) \geq 2w(T^{\star}) \geq w(H) \\
 & \implies \frac{w(H)}{w(H^{\star})} \leq 2
\end{align}
$$

This instance of the problem shows that the bound is _tight_:
![[aalg-metrictspapprox-exa.png]]

##### A 3/2-approx Algorithm for Metric TSP (Christofides' Algorithm)

The reason for the 2-approximation factor of the algorithm above was that the preorder traversal of $T^{\star}$ **crossed every edge of it exactly twice**. In the worst case the shortcuts will not improve the cost of $T^{\star}$ at all (think of a degenerate triangle with the "long" edge being as long as the sum of the other two).

Obviously to improve this we need to find a way to traverse the MST edges **only once**.

The idea is to use the notion of **Eulerian cycles**: an _Eulerian cycle_ is a path that crosses every edge of the graph _exactly once_.
A connected graph in which an Eulerian cycle exists is call Eulerian.

By definition, if the MST was Eulerian (which is impossible being a tree), we would actually have the solution, since the MST covers all nodes with minimum cost, and thus by being Eulerian there would be a tour of minimum cost between all nodes.

[[#A 2-approx Algorithm for Metric TSP|Approx-metric TSP]] finds a cheap Eulerian cycle using the MST, but it needs to _double its edges_.

Can we find a cheaper Eulerian cycle?

A famous theorem gives us an important result:
> **Theorem**: A connected graph is Eulerian $\iff$ every vertex has even degree

![[aalg-odd-degrees.png]]

Thus, we need to handle the odd-degree vertices of the MST, in order to obtain an Eulerian graph.

> **Property**: In any (finite) graph the number of vertices of odd-degree is _even_

^1bd387

**Proof**:
$$
\begin{align}
 \sum_{v \in V}deg(v) &  = 2m \\
 \underbrace{ \sum_{even} deg(v) }_{ \text{even because even n of terms} } + \underbrace{ \sum_{odd}deg(v) }_{\text{must be even} } & = \underbrace{ 2m }_{ \text{even by construction} } \\
\end{align}
$$

So the **idea** is to augment the initial MST $T^{\star}$ with a minimum-weight _perfect matching_ (i.e. that includes _all_ the vertices) between all the vertices that have odd degree in the MST, so that they become even degree $\implies$ the resulting graph is Eulerian.

###### Algorithm
```pseudo
\begin{algorithm}
\caption{Christofides' Algorithm}
\begin{algorithmic}
\Procedure{Christofides}{$G$}
	\State $T^\star \gets $ \Call{Prim}{$G, r$}
	\State Let $D$ be the set of vertices of $T^\star$ with odd degree. Compute a min-weight perfect matching $M^\star$ on the graph induced by $D$.
	\State The graph $(V, E^\star \cup M^\star)$ is Eulerian. Compute an Eulerian cycle on this graph.
	\Return The cycle that visits all the vertices of $G$ in the order of their first appearance in the Eulerian cycle.
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

![[aalg-christ-exa1.png]]
![[aalg-christ-exa2.png]]

###### Analysis

1. $w(H) \leq w(T^{\star}) + w(M^{\star})$, since in the worst case we have to pick all the edges in $T^{\star}$ and $M^{\star}$, but we may have shortcuts.
2. $w(T^{\star}) \leq w(H^{\star})$ since $H^{\star}$ has $n$ edges and $T^{\star}$ is a tree of minimum cost (with $n-1$ edges).
3. $w(M^{\star}) \leq \frac{1}{2}w(H^{\star A})$.
	Consider an optimal tour of the odd-degree vertices of $T^{\star}$, we know by [[#^1bd387| this property]] that the number of vertices (and thus edges) is even.

	We also know that
	$$
	w(\text{optimal odd-degree tour}) \leq w(H^{\star})
	$$
	since it's a tour with less vertices of the graph.

	Since the number of odd-degree vertices is even, we can partition the tour in two perfect matchings, one of the two must have weigth $\leq \frac{w(H^{\star})}{2}$ (since their sum is $w(H^{\star})$ at most).

	But both have weight $\geq w(M^{\star})$ by definition, since $M^{\star}$ is a min-cost perfect matching between odd-degree vertices.
	$$
	\implies w(M^{\star}) \leq \frac{w(H^{\star})}{2}
	$$

Finally, putting pieces together:
$$
\begin{align}
w(H)  & \leq w(H^{\star}) + w(M^{\star}) \\
 & \leq w(H^{\star}) + \frac{w(H^{\star})}{2} = \frac{3}{2}w(H^{\star})
\end{align}
$$

### Set Cover

Let's define the **Set Cover** problem:
$$
\begin{align}
 & I = (X, F) \\
 & X = \text{set of elements, called universe} \\
 & F \subseteq \mathcal{B}(X) \text{ the set of all subsets of } X
\end{align}
$$

The constraint is that $\forall x \in X \ \ \exists S \in F$ such that $x \in S$, that is "$F$ covers $X$".

The optimization problem consists in finding the smallest $F' \subseteq F$ such that:
1. $F'$ covers $X$
2. $|F'|$ is minimal

**Example**:
$X = \{ 1, 2, 3, 4, 5 \}$
$F = \{ \{ 1, 2, 3 \}, \{ 2, 4 \}, \{ 3, 4 \}, \{ 4, 5 \} \}$

Hence $F' = \{ \{ 1, 2, 3 \}, \{ 4, 5 \} \}$

> **Theorem**: Set Cover (in its decision form) is NP-hard.

**Proof**: Vertex Cover $\leq_{P}$ Set Cover

The reduction is simple:
We start from the original graph for Vertex Cover $G = (V, E)$ with target $k$ vertices, and we transform it into a Set Cover problem with:
- $X = E$ (i.e. we want to cover all the edges)
- $F = \{S_{1}, \dots, S_{n} \}$, with $S_{i} = \{ (u, v) \in E \mid u =i \lor v=i \}$ (i.e. the $i$-th subset contains all the incident edges of the $i$-th node)

With this it's immediate to show that:
$\exists$ a set cover of size $k$ $\iff$ $\exists$ a vertex cover of size $k$

#### Approximation Algorithm (Greedy Approach)

One simple approximation algorithm is the following one:
1. Choose the subset $S \in F$ that contains (i.e. **covers**) the **most uncovered elements from $X$**.
2. Remove from $X$ the covered elements
3. Repeat the steps until there are no more elements to cover (i.e. $X = \emptyset$).

```pseudo
\begin{algorithm}
\caption{Set Cover Approximation Algorithm}
\begin{algorithmic}
\State $U \gets X$
\State $F' \gets \emptyset$
\While{$U \neq \emptyset$}
	\State $S' \gets \arg\max_{S \in F}{|S \cap U|}$
	\State $U \gets U \setminus S'$
	\State $F \gets F \setminus \{S'\}$
	\State $F' \gets F' \cup \{S'\}$
\EndWhile
\Return $F'$
\end{algorithmic}
\end{algorithm}
```

#### Analysis

##### Correctness
By construction at every iteration the size of the uncovered elements $|U|$ decreases by **at least one**.

##### Complexity
Again, by construction, we observe that:
- The number of iterations is $\leq |X|$, since at worst every new subset only contains one uncovered element from $X$ 
- The number of iterations is also $\leq |F|$, since at most we would need all the subsets from $F$ to cover the set
Hence, the number of iterations is surely $\leq \min{\{ |X|, |F| \}}$.

The per-iteration complexity is $\leq |X||F|$, since we need to check for every element in $X$ every subset in $F$.

Thus, the total complexity for the algorithm is:
$$
O(|X||F|\min\{ |X|, |F| \})
$$
which is **at most cubic** in the input size.

Note that with the "right" data structure it can be implemented in $O(|X| + |F|)$ (i.e. in **linear time**).

##### Approximation

We will now show that this is a $\lceil \log_{2}n \rceil + 1$-approximation algorithm:
$$
\frac{|F'|}{|F^{\star}|} \leq \lceil  \log_{2}n \rceil + 1 \quad (n = |X|)
$$

The idea is to try to bound the number of iterations such that the set of **remaining elements $U$** gets empty.

Let's define:
- $U_{0} = X$ the remaining elements at the beginning
- $U_{i}$ the remaining elements at the **end** of the $i$-th iteration
- $|F^{\star}| = k$ the size of the cover, which is unknown

> **Lemma**: After the first $k$ iterations the residual universe _at least halved_ in size, that is:
> $$|U_{k}| \leq \frac{n}{2}$$

**Proof**:
First we observe that:
$U_{k} \subseteq X \implies$ $U_{k}$ admits a cover of size $\leq k$ (being smaller of $X$), with the cover all in $F$ (i.e. **not currently selected** by the algorithm).

We can characterize such cover as:
$$
T_{1}, T_{2}, \dots, T_{k} \in F\quad \bigcup T_{i} \text{ covers } U_{k}
$$
By the _pigeonhole_ principle, since we have at least have as many elements (pigeons) as we have subsets in which they are partitioned (pigeonholes), $\exists \bar{T}$ such that $|U_{k} \cap \bar{T}| \geq \frac{|U_{k}|}{k}$.

We will now see that in the first $k$ iterations, for each iteration  at least $\frac{|U_{k}|}{k}$ elements get covered.
Let $S_{i} \in F, 1 \leq i \leq k$ be the selected subsets $\implies |S_{i} \cap U_{i}| \geq |T_{j} \cap U_{i}|, \forall 1\leq j\leq k$, since $T_{j}$ has not been selected, hence by construction it's smaller than $S_{j}$.

This also holds for $\bar{T}$, that is:
$$
|S_{i} \cap U_{i}| \geq |\bar{T} \cap U_{i}| \geq |\bar{T} \cap U_{k}| \geq \frac{|U_{k}|}{k}
$$
Hence after the first $k$ iterations the algorithm has covered $\geq \frac{|U_{k}|}{k}\cdot k = |U_{k}|$ elements.
$$
\begin{align}
\implies \underbrace{ |U_{k}| }_{\text{residual}} \le \underbrace{ n - |U_{k}| }_{ \text{covered} } \\
|U_{k}| \leq \frac{n}{2}
\end{align}
$$


This lemma implies that after $k\cdot i$ iterations the size of the uncovered set has decreased exponentially: $|U_{ki}| \leq \frac{n}{2^{i}}$.
Hence the number of necessary iterations in order to empty $U$ is $\lceil \log_{2}n \rceil\cdot k + 1$ at each iteration $|F'|++$.

Thus to conclude
$$
\begin{align}
 & \implies |F'| \leq \lceil \log_{2}n \rceil k + 1 \\
 & \implies |F'| \le \lceil \log_{2}n \rceil |F^{\star}| + 1
\end{align}
$$

**Example**
One possible input that shows that the bound is tight is the following one:
Assume $X$ has $n = 2^{(k+1)} - 2$ elements for some $k \in \mathbb{N}$.

Then $F$ has:
1. $k$ pairwise disjoint sets $S_{1}, \dots, S_{k}$ with sizes $2^{1}, \dots, 2^{k}$
2. Two additional disjoint sets $T_{0}, T_{1}$ each of which contains half of the elements from each $S_{i}$.

![[aalg-approx-setcover.png]]

The approximate greedy algorithm would choose $S_{k}, S_{k-1}, \dots, S_{0}$, while the optimal cover would be $\{ T_{0}, T_{1} \}$.

Thus the approximate algorithm has size $\Theta(\log n)$.