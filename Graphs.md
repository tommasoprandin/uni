### Basics

A _graph_ is a representation of the relationships between pairs of objects.
Formally a graph is denoted as $G = (V, E)$, where:
- $V$ = set of _vertices_ (_nodes_)
- $E \subseteq V \times V$ is a _collection_ (not a set) of _edges_, where an _edge_ is a pair of vertices: $e = (u, v)$ with $v, u \in V$.

A graph may be _directed_ if the direction of the egdes is relevant $\implies (u, v) \neq (v, u)$, in this case we often refer to edges as _arcs_.
It can also be _undirected_ if the direction of the edges is irrelevant $\implies (u, v) = (v, u)$.
![[graph-exa.png]]

A graph is said to be _simple_ if there are NO _parallel edges_ (i.e. in $E$ there is only one instance of the pair $(u, v) \ \  \forall u, v \in V$), and NO _self-loops_ (i. e. in $E \ \ \forall u\in V, (u, u) \not\in E$).
![[graph-simple-exa.png]]

#### Terminology
- Given an edge $e = (u, v)$, we say
	- $e$ is _incident_ on $u, v$
	- $u, v$ are _adjacent_ nodes.
- Given a vertex v, we define:
	- Set of _neighbors_ of a node: $N(v) = \{ u \in V \ |\ (u, v) \in E \}$
	- _Degree_ of a node: $d(v) = deg(v) = |\{ e \in E\ |\ e \text{ incident to } v \}|$.
- _Path_: sequence of vertices $v_{1}, v_{2}, \dots, v_{k}$, where $(v_{i}, v_{i+1}) \in E \ \forall \ i \in 1,\dots,k - 1$.
	- _Simple path_: a path where all the traversed nodes are _distinct_.
	- _Cycle_: a simple path where the first node is the last one ($v_{1}=v_{k}$).
- _Subgraph_: given two graphs $G = (V, E), G' = (V', E')$, $G'$ is a _subgraph_ of $G$ $\iff$ 
	- $V' \subseteq V$
	- $E' \subseteq E$
	- $\forall (u, v) \in E'; u, v \in V'$ (i.e. there are no orphan edges)
- _Spanning subgraph_: a subgraph where $V' = V$ (i.e. it covers the entirety of the original graph).
- _Connected graph_: if $\forall\ u, v \in V$, there exist a path from $u$ to $v$.
- _Connected components_: given a graph $G$, a partition of it in subgraphs $G_i = (V_{i}, E_{i})$, where, $\forall\ i \in 1,\dots k$:
	- $G_{i}$ is connected
	- $V = \bigcup_{i=1}^{k} V_{i}$
	- $E = \bigcup_{i=1}^{k} E_{i}$ (i.e. its a complete partition)
	- $\forall\ i\neq j \in 1,\dots,k$ there is no edge between $V_{i}$ and $V_{j}$ (i.e. the subgraphs of the partition are disconnected between them)
- _Tree_: a connected graph without cycles ![[graph-tree-exa.png]]
- _Forest_: a disjoint set of trees ![[graph-forest-exa.png]]
- _Spanning tree_: a spanning subgraph connected and without cycles, it exists _only if_ $G$ is connected. ![[graph-spanning-tree-exa.png]]
- _Spanning forest_: a spanning subgraph without cycles ![[graph-spanning-forest-exa.png]]

#### Notation
- Number of nodes: $n = |V|$
- Number of edges: $m = |E|$
- Size of a graph: $n + m$

#### Properties of Graphs

^39ff64

Let $G = (V, E)$ be a _simple_, _undirected_ graph with $n$ vertices and $m$ edges. Then:
1. $$
\sum_{v \in V}{d(v)} = 2m
$$
_Proof_: since the graph is simple and undirected any edge in $E$ insists on two _distinct_ nodes (no self-loops), and is unique (no parallel edges). Thus, when iterating through every vertex, any edge will be counted _exactly twice_ for every vertex end of the edge ($\forall\ e = (u, v) \in E$ it will be counted when computing $d(u)$ and $d(v)$).
2. $$m \leq \binom{n}{2}$$
_Proof_: as above, since the graph is simple and undirected, any edge in $E$ will be unique and insist on two distinct nodes. Furthermore $(u,v) = (v,u)$. Thus we can calculate the maximum number of edges in a graph with $n$ vertices by computing the number of possible combinations without repetitions of pairs in a set of size $n$, which exactly corresponds to $\binom{n}{2}$. Obviously any lower number of edges is possible.
3. $$G \text{ is a tree}\implies m=n-1$$
_Proof_: ^415fbe
- $G$ is a tree $\implies m \geq n-1$, in fact the smallest number of edges required to obtain a fully connected graph is exactly $n-1$, obtained by picking a random node and connecting it to all the other $n-1$ nodes.
- $G$ is a tree $\implies m \leq n-1$, taking the previous proof if we add one single edge to the minimal tree build before we will surely obtain a cycle in the "root" of the graph.
4. $$G \text{ is connected}\implies m\geq n-1$$
- _Proof_: derived from point 3
5. $$G \text{ is acyclic (i.e it is a forest)} \implies m\leq n-1$$
- _Proof_: derived from point 3

### Representing a Graph

We first need to decide how to encode the concept of a graph in a program. Let's first define two lists:
- $L_{V}[v]$ contains the "metadata" for every vertex $v$ in the vertex set $V$ of the graph $G$, and it is indexed by the vertex $v$ itself.
- $L_{E}[e]$ contains the "metadata" for every edge $e$ in the edge collection $E$ of the graph $G$, and it is indexed by the edge $e$ itself.

The content of those lists will depend on the data needed by the algorithm.

Since $V$ is a finite set, vertices are enumerable, so we will refer to them by a number: $1,\dots,n$. Additionally assume that $L_{V}, L_{E}, A$ are sorted by vertex number.

Now, given a vertex $v \in V$, we need a way to access directly its incident edges (i.e. its relationships with neighbors). There are two main approaches to this:

#### Adjacency List
An array $A$ of $n$ lists, one for every vertex $v \in V$, each containing the list of neighbors of $v$.
Consider this graph:
![[graph-repr-exa.png | center | 400 ]] 
$$
A = \begin{array}{|c|cccc|}
\hline
1 & 2 & 5  &  \\
2 & 1 & 3 & 4 & 5 \\
3 & 2 & 4 &  &  \\
4 & 2 & 5 & 3 &  \\
5 & 4 & 1 & 2 &  \\
\hline
\end{array}
$$

| Advantages | Disadvantages |
| -|-|
|Linear space usage: $\theta(n +m)$| No quick way to determine if given edge is in the graph |


#### Adjacency Matrix
A $n\times n$ matrix $A$ such that:
$$
A_{i,j} = \begin{cases}
1 & \text{if } edge(i, j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

Consider this graph again, its adjacency matrix $A$ is:
![[graph-repr-exa.png | center | 400]]

$$
A = \begin{array}{|c|ccccc|}
\hline
& 1 & 2 & 3 & 4 & 5 \\
\hline
1 & 0 & 1 & 0 & 0 & 1 \\
2 & 1 & 0 & 1 & 1 & 1 \\
3 & 0 & 1 & 0 & 1 & 0 \\
4 & 0 & 1 & 1 & 0 & 1\\
5 & 1 & 1 & 0 & 1 & 0 \\
\hline
\end{array}
$$

This approach works well to represent weighted edges, since the matrix definition simply becomes:
$$
A_{i,j} = \begin{cases}
w & \text{if } edge(i, j) \in E \\
- & \text{otherwise}
\end{cases}
$$
Notice we had to replace the $0$ with the $-$ symbol, in order to distinguish a missing edge from one with weight zero.

An _undirected_ graph originates a _symmetric_ adjacency matrix, this enables space reduction by storing only the upper "triangle" of the matrix, otherwise a _directed_ graph determines an _asymmetric_ matrix.

| Advantages | Disadvantages |
|-|-|
| $\theta(1)$ complexity for determining if an edge is in the graph | Space required is superlinear ($\theta(n^2)$), not good if graph is _sparse_|

### Graph Search and its Applications

A graph search algorithm (also referred as graph traversal algorithms) is a procedure to _systematically explore_ a graph $G$, starting from one of its vertices $s \in V$, then visiting all the remaining vertices.

There are two main algorithms for this problem:
- **Depth First Search**
- **Breadth First Search**

#### Depth First Search Algorithm
The idea behind DFS is to start from a vertex $s$ and then, recursively from it, start drilling down greedily exploring the first children of every vertex in the path, then backtracking when the exploration leads to an already visited node. The algorithm ONLY visits the _connected component_ $C_{s} \subseteq G$ containing $s$.

##### Hypothesis
- The implementation uses an _adjacency list_ representation
- Every vertex $v \in V$ has a boolean field $L_{V}[v].visited$ 
- Every edge $e \in E$ has a label $L_{E}[e].label$, that can assume the values:
	- Null
	- "DISCOVERY EDGE"
	- "BACK EDGE"

##### Implementation

```pseudo
\begin{algorithm}
\caption{DFS}
\begin{algorithmic}
\Procedure{DFS}{$G, v$}
	\State visit $v$
	\State $L_V[v].visited \gets$ True
	\ForAll{$e \in$ \Call{IncidentEdges}{$v$}}
		\If{$L_E[e].label$ is Null}
			\State $w \gets$ \Call{OppositeEdge}{$v, e$}
			\If{$L_V[w].visited$ = False}
				\State $L_E[e].label \gets$ "DISCOVERY EDGE"
				\State \Call{DFS}{$G, w$}
			\Else
				\State $L_E[e].label \gets$ "BACK EDGE"
            \EndIf
        \EndIf
    \EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

![[graph-dfs-exa.png]]

##### Correctness
Let's now show the correctness of the algorithm. At the end of execution:

1. All the vertices $v \in V_{s}$ have been visited, and all the all the edges $e \in E_{s}$ are labelled with either "DISCOVERY EDGE" or "BACK EDGE".

**Proof**: assume there exists a vertex $v \in V_{s}$ that has not been visited. Since $v$ is in the connected component of $s$, there exists a path from $s$ to $v$:
$$
s = v_{0} \to v_{1} \to \dots \to v_{k} = v
$$

Now let's pick $v_{i}$, which is the first vertex in the path not visited (could be $v$ itself). By construction a vertex can be visited only by calling $DFS(G, v)$ on the vertex itself.

Since $v_i$ is the first non-visited vertex in the path $\implies$ $v_{i - 1}$ has been visited $\implies DFS(G, v_{i-1})$ has been invoked. But, by construction, the algorithm iterates over all the adjacent edges, and calls $DFS$ on every unvisited neighbor $\implies DFS$  must have been called on $v_{i}$ too, this contradicts the initial hypothesis, proving that $DFS$ visits all the nodes in the connected component.

Since all edges of $C_{s}$ are visited then, by construction again, all the edges are visited (and labelled too).

2. The set of edges labeled as "DISCOVERY", forms a _spanning tree_ of $C_{s}$ (called the "DFS tree").

**Proof**: By construction, for every vertex $v \in C_{s}, DFS(G, v)$  is called just one single time (due to the checking whether the node has already been visited), and it is invoked from a node $u$ for which $\exists\ e= (u,v) \in E_{s}$, that is the "parent" of $v$ (i.e. $v$ gets "discovered" by $u$). The edge $e$ is marked as "DISCOVERY" (again by construction of the algorithm).
- $\implies \forall v \in V_{s}, v\neq s$ 
	- There exists a _unique_ parent
	- Going back parent-to-parent eventually reaches $s$ (the starting node in which the tree is rooted)
- $\implies$ the set of "DISCOVERY EDGES" is a tree rooted in $s$, with all the vertices of $V_{s}$, thus it is a spanning tree of $C_s$.

##### Complexity

Denoting as
- $n_{s}$ = the number of vertices in $C_{s}$
- $m_{s}$ = the number of edges in $C_{s}$
we observe that the complexity of $DFS$ is:
$$
\Theta\left( \sum_{v \in V_{s}}{d(v)}  \right) = \Theta(m_{s})
$$

This is given by the `for each` in the algorithm, that gets invoked on every incident edge for every node, thus giving the sum of the degrees of the nodes, which, from  [[#^39ff64|property 1]] is equal to $2m_{s}$.


##### Applications

**Graph Visiting**:
An algorithm for entire graph exploration is simply derived from $DFS$, just by iteratively calling it over all the nodes:

```pseudo
\begin{algorithm}
\caption{DFSTraversal}
\begin{algorithmic}
\Procedure{DFSTraversal}{$G$}
	\ForAll{$v \in V$}
		\If{$L_v[v].visited$ is False}
			\State \Call{DFS}{$G, v$}
        \EndIf
    \EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The complexity of the graph traversal algorithm is proportional to the size of the graph: 
$$
\Theta(n+m)
$$

**Path between two vertices**:
Given two vertices $s,t \in V$ we want to return a path from $s$ to $t$ (if it exists).
For this we will add a $parent$ field in $L_{V}$, in order to keep track of the spanning tree created starting from $s$. If $t$ is found then we can just track back to the root of the tree ($s$), finding our path. ^66bbeb
```pseudo
\begin{algorithm}
\caption{STPath}
\begin{algorithmic}
\Procedure{STPath}{$G, s, t$}
	\State visit $s$
	\State $L_V[s].visited \gets$ True
	\ForAll{$e \in$ \Call{IncidentEdges}{$s$}}
		\If{$L_E[e].label$ is Null}
			\State $w \gets$ \Call{OppositeEdge}{$s, e$}
			\If{$L_V[w].visited$ = False}
				\State $L_E[e].label \gets$ "DISCOVERY EDGE"
				\State $L_V[w].parent \gets$ s
				\If{$w$ is $t$}
					\Return \Call{Rollback}{$G, w$}
                \EndIf
				\State \Call{STPath}{$G, w$}
			\Else
				\State $L_E[e].label \gets$ "BACK EDGE"
            \EndIf
        \EndIf
    \EndFor
\EndProcedure

\Procedure{Rollback}{$G, v$}
	\State $path \gets$ \Call{EmptyStack}{}
	\State $parent \gets L_v[v].parent$
	\While{$parent$ is not Null}
		\State \Call{Push}{$path, parent$}
		\State $parent \gets L_v[parent].parent$
    \EndWhile
	\Return \Call{PopAll}{$path$}
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The _complexity_ for it is $\Theta(m_{s})$ for the execution on the connected component of $s$.

**Cycle Detection**:
Given a graph $G$, we want to return a cycle in $G$ (if any). Recall a cycle is any path such that $s = v_{1} \to v_{2} \to \dots \to v_{k} = s$. For this application we observe that as soon as we see a "BACK EDGE", we know there is a cycle in the graph. In order to return the cycle though we need to add a $parent$ field (as done [[#^66bbeb | here]]), then rolling back to the starting node of the cycle.

```pseudo
\begin{algorithm}
\caption{CycleDetection}
\begin{algorithmic}
\Procedure{CycleDetection}{$G, v$}
	\State visit $v$
	\State $L_V[v].visited \gets$ True
	\ForAll{$e \in$ \Call{IncidentEdges}{$v$}}
		\If{$L_E[e].label$ is Null}
			\State $w \gets$ \Call{OppositeEdge}{$v, e$}
			\If{$L_V[w].visited$ = False}
				\State $L_E[e].label \gets$ "DISCOVERY EDGE"
				\State $L_V[w].parent \gets$ s
				\State \Call{CycleDetection}{$G, w$}
			\Else
				\State $L_E[e].label \gets$ "BACK EDGE"
				\Return \Call{Rollback}{$G, v, w$}
            \EndIf
        \EndIf
    \EndFor
\EndProcedure

\Procedure{Rollback}{$G, v, w$}
	\State $path \gets$ \Call{EmptyStack}{}
	\State $parent \gets L_v[v].parent$
	\While{$parent$ is not Null $\land\ parent$ is not $w$}
		\State \Call{Push}{$path, parent$}
		\State $parent \gets L_v[parent].parent$
    \EndWhile
	\Return \Call{PopAll}{$path$}
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The _complexity_ for it is $\Theta(m_{s})$ for the execution on the connected component of $s$ (shown in the pseudocode), and $\Theta(n + m)$ for the entire graph $G$.

**Graph Connectivity**:
Given a graph $G$, determine if the graph is connected $\iff$ the number of connected components is 1.

This is simply done by running the $DFS$ traversal algorithm and keeping track of the explored nodes, then if $V_{s} \subset V \implies$ the graph is not connected.

Again the complexity is given by the $DFS$ traversal algorithm  $\implies \Theta(n + m)$.

**Connected Components**:
A very common problem encountered when dealing with graphs is the labeling of the connected components of $G$, so that, given two nodes $v, w \in V$, they have the same labeling $\iff$ they belong to the same connected component.

By modifying the $L_{V}.visited$ field to a numerical $L_{V}.ID$ field we can assign different labels to the various connected components.

Suppose to modify $DFS$ so that, instead of setting the $visited$ field to True on discovery, it "marks" the node with a specified $k$ parameter (this is trivial to do).
Then the solution to the problem is:

```pseudo
\begin{algorithm}
\caption{ConnectedComponents}
\begin{algorithmic}
\Procedure{ConnectedComponents}{$G$}
	\ForAll{$v \in V$}
		\State $L_V[v].ID \gets 0$
    \EndFor
	\State $k \gets 0$
	\ForAll{$v \in V$}
		\State $k \gets k + 1$
		\State \Call{LabelingDFS}{$G, v, k$}
    \EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

Again _complexity_ is linear with the size of the graph $\Theta(n+m)$.

**Summary**
All the following problems are solvable in $\Theta(n+m)$ time using a variation of the $DFS$ algorithm:

- test if $G$ is connected
- label connected components of $G$
- find a _spanning tree_ of $G$ (if connected)
- find a path between two vertices (if any)
- find a cycle (if any)

#### Breadth First Search Algorithm

An alternative approach to DFS is to instead explore the graph sistematically covering all the children of a node and then proceeding on to the next layer. As a consequence the tree built by BFS is the one with shortest paths from the root. This is a very useful property that allows us to solve some problems not solvable using DFS.

##### Hypothesis
- The implementation uses an _adjacency list_ representation
- Every vertex $v \in V$ has a boolean field $L_{V}[v].visited$ 
- Every edge $e \in E$ has a label $L_{E}[e].label$, that can assume the values:
	- Null
	- "DISCOVERY EDGE"
	- "CROSS EDGE" (because it represents the "crossing" between two branches)

##### Implementation
Contrary to DFS, the BFS algorithm is more naturally expressed using iteration. We will keep a list of nodes for every $i$-th level $L_{i}$ ($i$-th level means $i$ steps away from the root).

```pseudo
\begin{algorithm}
\caption{BFS}
\begin{algorithmic}
\Procedure{BFS}{$G, v$}
	\State visit $v$
	\State $L_V[v].visited \gets$ True
	\State $L_0 \gets [v]$
	\State $level \gets 0$
	\While{$L_{level}$ is not Empty}
		\State $L_{level + 1} \gets []$
		\ForAll{$w \in L_{level}$}
			\ForAll{$e \in $ \Call{IncidentEdges}{$w$}}
				\If{$L_E[e].label$ is Null}
					\State $u \gets$ \Call{OppositeEdge}{$G, w$}
					\If{$L_V[u].visited$ is False}
						\State visit $u$
						\State $L_V[u].visited \gets$ True
						\State $L_E[e].label \gets$ "DISCOVERY EDGE"
						\State \Call{Push}{$L_{level + 1}, u$}
					\Else
						\State $L_E[e].label \gets$ "CROSS EDGE"
                    \EndIf
                \EndIf
            \EndFor
        \EndFor
		\State $level \gets level + 1$
    \EndWhile
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

![[graph-bfs-exa.png]]
##### Correctness
Let's now show the correctness of the algorithm. At the end of execution:

1. All the vertices $v \in V_{s}$ have been visited, and all the all the edges $e \in E_{s}$ are labelled with either "DISCOVERY EDGE" or "CROSS EDGE".

**Proof**: By the nature of BFS, a vertex $v$ is discovered when it is first encountered during the exploration, which happens when an edge $(u, v)$ is traversed from a previously discovered vertex $u$. Since BFS systematically explores all vertices adjacent to discovered vertices before moving to the next level, it will eventually discover all vertices reachable from $s$. If a vertex is not visited by BFS, it must not be reachable from $s$, contradicting the definition of $V_s$.

For edge classification, BFS examines every edge $(u, v)$ where $u \in V_s$. If $v$ has not been discovered, the edge becomes a "DISCOVERY EDGE" and $v$ is added to the queue. If $v$ has already been discovered, the edge is classified as a "CROSS EDGE". Since BFS exhaustively examines all adjacencies of each vertex in $V_s$, every edge in $E_s$ must fall into one of these two categories.

2. The set of edges labeled as "DISCOVERY", forms a _spanning tree_ of $C_{s}$ (called the "BFS tree").

**Proof**: First, the discovery edges connect all vertices in $V_s$ to $s$. This is because a vertex $v$ is discovered precisely when a discovery edge $(u, v)$ is traversed, where $u$ is already discovered. By induction, this creates a path of discovery edges from $s$ to every vertex in $V_s$.

Second, the discovery edges do not form cycles. Suppose, for contradiction, that the discovery edges contain a cycle. Let $(u, v)$ be the last edge in this cycle to be labeled as a discovery edge. When BFS examines $(u, v)$, vertex $v$ must already be reachable from $s$ via another path of discovery edges. But this means $v$ would have already been discovered when $(u, v)$ is examined, so $(u, v)$ would be labeled as a cross edge, contradicting our assumption.

Thus, the discovery edges form a connected subgraph with no cycles, which is by definition a tree. Since it reaches all vertices in $V_s$, it is a spanning tree of $C_s$.

1. $\forall v \in L_{i}$, the path in $T$ from the root $s$ to $v$ is _exactly_ $i$ steps long. Every other possible path from $s$ to $v$ has $\geq$ edges. ^18efac

**Proof**: By construction, a vertex $v$ is placed in level $L_i$ if and only if it is first discovered during the exploration of vertices in level $L_{i-1}$. This means there exists a vertex $u \in L_{i-1}$ such that $(u, v)$ is a discovery edge. By induction, there is a path of $i-1$ discovery edges from $s$ to $u$, so the path from $s$ to $v$ in the BFS tree consists of exactly $i$ edges.

Now, suppose there exists a path $P$ from $s$ to $v$ with fewer than $i$ edges. Let the length of this path be $j < i$. Then, when BFS explores level $L_j$, it would discover $v$ and add it to level $L_{j+1}$. This contradicts our assumption that $v \in L_i$. Therefore, any path from $s$ to $v$ must have at least $i$ edges.
##### Complexity

For every vertex $v \in V_{S}$, one iteration of the outer "for all" is invoked and $d(v)$ invokations of the inner "for all" (all incident edges). Thus we have $\sum_{v \in V_{s}}{d(v)}$ iterations, which implies that the complexity is: $$
\Theta(m_{s})$$ identical to DFS.

##### Applications

The applications are the same previously shown for the DFS algorithm [[#Applications | here]], with the same $\Theta(n + m)$ complexity.

An additional application not possible with DFS is the _shortest path_ problem.

**Shortest Path**:
Given a graph $G$ and two nodes $s, v \in V$, return the _shortest path_ between them (if any).

**Implementation**
It follows a high-level description of the algorithm. Its correctness comes from property [[#^18efac | 3]] of the $BFS$ algorithm, which expands the exploration level-by-level in order of least hops.

- $\forall\ v in V$ add a $parent$ field to $L_{V}$.
- modify $BFS(G, s)$  such that, when $(v, u)$ is labeled "DISCOVERY EDGE", then $L_{V}[V].parent = v$
- run $BFS$ until either $v$ is found, in such case return the set of child-parent edges, or all $C_{s}$ is explored and $v$ is not found, in which case return null.

The complexity is the same as $BFS$, so $O(m_{s})$.

### Minimum Spanning Tree

Given a graph $G=(V, E)$ _undirected_, _connected_ and _weighted_; the _minimum spanning tree_ problem is to find the cheapest possible way to connect all the vertices. This problem is a very recurrent and important one in graph theory, and it is commonly used as a subroutine for other more complex algorithms.

More formally:
_Input_: a graph $G = (V,E)$ undirected, connected and weighted. Weighted means that there is a _weight function_:
$$
\begin{gather}
w: E \to \mathbb{R} \\
w(u,v) = \text{cost of the}(u,v) \text{ edge}
\end{gather}
$$

_Output_: a spanning tree $T \subseteq E$ of $G$ such that:
$$
w(T) = \sum_{e \in E_{T}}{w(e)} \text{ is minimal}
$$

![[graph-mst-exa.png]]

We observe that:
1. It only makes sense to define MST for weighted graphs, since any spanning tree in an unweighted graph has the same number of edges (see [[#^415fbe|property]]). This is actually also true for any graph where all edges have the same weight.
2. The assumption of $G$ being connected can be taken without loss of generality (or we would be considering _minimum spanning forests_).

#### Applications
There are various real-world important applications for the algorithm such as:
- Networks
- Machine Learning (clustering)
- Computer Vision (object detection)
- Data Mining
- Step for approximation algorithms

#### Generic Approach

Generally a graph have many spanning trees, in the worst case of a _fully connected_ graph (a graph with all possible $\binom{n}{2}$ edges) there are $n^{n - 2}$ different spanning trees! (see Cayley formula).

However MST can be solved in near-linear time with Greedy algorithms, which are simple to implement in practice. The main of which are:
- **Prim's Algorithm**
- **Kruskal's Algorithm**

They both apply the same generic greedy algorithm.
The idea is to iteratively add a "safe-edge" to a starting set of edges $A \subseteq V_{T}$, where $T$ is some MST of the graph. A "safe-edge" is an edge $e \not\in A$, that added to $A$ itself would maintain the property $A \subseteq V_{T}$ for some $T \implies A$ is still a subset of an MST.
After adding all the possible safe edges by induction $A$ will be an MST.

##### Generic-MST Algorithm

^b2674b

```pseudo
\begin{algorithm}
\caption{GenericMST}
\begin{algorithmic}
\Procedure{GenericMST}{$G$}
	\State $A \gets \emptyset$
	\While{$A$ does not form a spanning tree}
		\State find an edge $(u, v)$ that is safe for $A$
		\State $A \gets A \cup {(u, v)}$
    \EndWhile
	\Return A
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The crucial missing step is how to algorithmically find a safe edge for $A$. We will illustrate and prove a theorem that will allow us to systematically find safe edges for $A$.

##### Safe Edge Search

First some preliminary definitions:
- A **cut** in a graph $G = (V, E)$ is a partition of $V$ in two subsets $(S, V \setminus S)$
- An edge $e = (u, v) \in E$ **crosses** a cut $(S, V\setminus S) \iff u \in S \land v \in V \setminus S$ (or viceversa)
- Given a set of edges $A \subseteq E$ , a cut **respects** $A \iff$ no edge in $A$ crosses the cut
- Given a cut, an edge that crosses the cut and is of minimum weight among all the other edges crossing the cut, is called **light edge** (for that cut)

![[graph-light-edge.png]]

> **Theorem**:
>  Let $G=(V,E)$ be an indirect, connected and weighted graph.
>  Let $A \subseteq E$, included in some MST of $G$ ($A$ can be potentially empty).
>  Let $(S, V\setminus S)$ be a cut that _respects_ $A$, and let $e = (u, v) \in E$ be a _light edge_ for the cut.
>
> Then $e$ is safe for $A$.

The intuition behind it is that if we split the graph in two partitions not connected by A, then surely the least expensive way to connect those will belong in the final MST. Iteratively repeating this will build the final MST.

![[graph-generic-mst.png]]

**Proof**:
Let $T$ be a minimum spanning tree that includes $A$, and assume that $T$ does not contain the light edge $(u,v)$, since if it does, we are done.

We will construct another minimum spanning tree $T'$ that includes $A \cup \{ (u,v) \}$ by using a cut-and-paste technique, thereby showing that $(u,v)$ is a safe edge for $A$.

The edge $(u,v)$ forms a cycle with the edges on the simple path $p$ from $u$ to $v$ in $T$ , as the figure below illustrates. Since $u$ and $v$ are on opposite sides of the cut $(S, V \setminus S)$, at least one edge in $T$ lies on the simple path $p$ and also crosses the cut.

Let $(x,y)$ be any such edge. The edge $(x,y)$ is not in $A$, because the cut respects $A$. Since $(x,y)$ is on the unique simple path from $u$ to $v$ in $T$ , removing $(x,y)$ breaks $T$ into two components. Adding $(u, v)$ reconnects them to form a new spanning tree $T' = (T \setminus \{ (x, y) \}) \cup \{ (u, v) \}$ .

We next show that $T'$ is a minimum spanning tree. Since $(u, v)$ is a light edge crossing $(S, V \setminus S)$ and $(x, y)$ also crosses this cut, $w(u, v) \leq w(x, y)$. Therefore,
$$
\begin{align}
w(T') &= w(T) - w(x, y) + w(u, v) \\
 & \leq w(T)
\end{align}
$$
But $T$ is a minimum spanning tree, so that $w(T) = w(T')$ , and thus, $T'$ must be a minimum spanning tree as well.

It remains to show that $(u,v)$ is actually a safe edge for $A$. We have $A \subseteq T'$, since $A \subseteq T$ and $(x, y) \not\in A$, and thus, $A \cup \{ (u, v) \} \subseteq T'$. Consequently, since $T'$ is a minimum spanning tree, $(u, v)$ is safe for $A$.

![[graph-safe-edge-proof.png]]

#### Prim's Algorithm

Prim's algorithm applies the generic algorithm by starting with a source vertex and iteratively finding a safe edge by cutting the graph between the _discovered_ and the _undiscovered_ vertices. After finding the safe edge it is added to the tree and the vertex along it gets added to the discovered list.

```pseudo
\begin{algorithm}
\caption{Prim}
\begin{algorithmic}
\Procedure{Prim}{$G,s$}
	\State $X \gets \{ s \}$
	\State $A \gets \emptyset$
	\While{there is an edge $e = (u, v)$ with $u \in X$ and $v \not\in X$}
		\State $(u', v') \gets$ a minimum weight edge between $u, v$ (safe edge)
		\State $X \gets X \cup \{v'\}$
		\State $A \gets A \cup \{(u', v')\}$
    \EndWhile
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

The algorithm _grows_ a spanning tree from a source vertex $s$ by adding one edge at a time.

![[graph-prim-anim.gif]]

##### Correctness

The correctness of the algorithm follows from the proof of the "safe edge" [[#Safe Edge Search| theorem]] and the construction of the algorithm that iteratively adds a safe edge to an initially empty tree $A$.

##### Complexity

The complexity of the algorithm comes from the two cycles:
- The external `while` iterates over the nodes $\implies \in O(n)$
- The internal search for a safe edge requires iterating over the crossing edges to find the one with minimum weight, which in the worst case requires traversing all of them $\implies \in O(m)$


Thus the final complexity, by composing the two cycles is:
$$
O(m\cdot n)
$$
so this implementation has **polynomial complexity**.

In theory this is relatively efficient but in practice, for very large and dense graphs (think of the Facebook graph with 2B nodes and hundreds of incident edges per node), it is not feasible.

We observe that there actually is a repeated computation in the algorithm that is currently done _naively_: the light (safe) edge search. We can speed up this significantly by using a **priority queue**, such that finding (and extracting) the light edge will take logarithmic time.

In fact, for a heap-based priority queue the following operations:
- `Insert`: add an object to the heap
- `ExtractMin` (or `ExtractMax`): remove the min (max) element from the heap
- `Delete`: remove the specified object

All require $O(\log n)$ time ($n$ is the number of elements), given by the `heap up` and `heap down` operations. (see [Wikipedia page](https://en.wikipedia.org/wiki/Heap_(data_structure)) ).

In practice it is simpler to store _vertices_ in the heap instead of the edges directly.

#### Prim's Algorithm (efficient implementation)

^53d299

This variant of the Prim's algorithm stores the vertices in a min-priority queue, ordered by a key that contains the smallest incident edge of the vertex. The queue is updated dynamically while traversing the graph.

The algorithm is very similar in principle to the Dijkstra's algorithm.

```pseudo
\begin{algorithm}
\caption{PrimQ}
\begin{algorithmic}
\Procedure{PrimQ}{$G, s$}
	\ForAll{$v \in V$}
		\State $L_v[v].key \gets \infty$
		\State $L_v[v].parent \gets$ Null
    \EndFor
	\State $L_v[s].key \gets 0$
	\State $Q \gets \emptyset$
	\ForAll{$v \in V$}
		\State \Call{Add}{$Q, (L_v[v].key, v)$}
    \EndFor
	\While{$Q \not= \emptyset$}
		\State $u \gets $ \Call{ExtractMin}{$Q$}
		\ForAll{$v \in $ \Call{Neighbors}{$G, u$}}
			\State $e \gets (u, v)$
			\If{$v \in Q \land L_e[e].w \lt L_v[v].key$}
				\State $L_v[v].parent \gets u$
				\State $L_v[v].key \gets L_e[e].w$
				\State \Call{DecreaseKey}{$Q, v, L_e[e].w$}
	        \EndIf
        \EndFor
    \EndWhile
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

##### Complexity

Let's break down the complexity of the algorithm for each step:
1. Vertices metadata initialization step: $O(n)$
2. Priority-queue initialization: $O(n)$
3. External while iterates over all vertices: $O(n)$ iterations
	1. `ExtractMin` runs in $O(\log n)$ for binary-tree heaps $\implies$ the total cost is $O(n\log n)$
	2. Internal for loop gets executed for every incident edge: $d(v) \implies$ the number of iterations is bounded by $\sum_{v \in V}d(v) \implies O(m)$
		1. Finding if $v \in Q$ and the comparison takes constant time $O(1) \implies$  total cost is $O(m)$
		2. Decreasing the key in a heap takes $O(\log n)$ time, so a total of $O(m \log n)$

 The total complexity comes from the `ExtractMin` operation and from the `DecreaseKey`: $O(n \log n) + O(m \log n)$. Since $G$ is connected we have that $m \geq n-1 \implies$ the total complexity is
$$
O(m \log n)
$$

#### Kruskal's Algorithm

Kruskal's algorithm is a very simple and effective algorithm for building an MST of a graph $G$.
The idea is to iteratively pick the edge with lowest weight and, if it doesn't create a cycle with the other edges already in, add it to the MST under construction.

```pseudo
\begin{algorithm}
\caption{Kruskal}
\begin{algorithmic}
\Procedure{Kruskal}{$G$}
	\State $A = \emptyset$
	\State \Call{Sort}{$E$} by weight
	\State $m \gets 0$
	\ForAll{$e \in E$}
		\If{$m = n-1$}
			\Return $A$
        \EndIf
		\If{$A \cup \{e\}$ is acyclic}
			\State $A \gets A \cup \{e\}$
			\State $m \gets m + 1$
        \EndIf
    \EndFor
	\Return $A$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

##### Correctness

The correctness of the algorithm comes from the [[#^b2674b | generic MST]], since the algorithm "virtually" applies a cut that crosses the edge of least weight. It does not precisely specify the cut, but since it picks the egde with lowest weight is surely is the light edge of any cut crossing it.

##### Complexity

Let's break down all the steps:
1. Sorting all the edges by weight: $O(m \log m)$
2. For loop does $O(m)$ iterations
	1. Cheking if $A \cup \{ e = (u, v) \}$ is acyclic is equivalent to checking if $A$ already contains a path from $u$ to $v$, which is accomplished by using $DFS(V, A)$ which has complexity $O(|A|)$. Since $A$ is a forest (or at most a tree) we know that $|A| \leq n+1 \implies$ the cycle detection complexity is $O(n)$

Thus the complexity of the body of the iteration is $O(n)$ and the number of iterations is $O(m) \implies$ the total complexity is:
$$
O(n\cdot m)
$$
which is identical to Prim's algorithm (naive implementation).

Actually also for Kruskal's algorithm there is a way to speedup the computation: since we are constantly checking if a newly added edge would create a cycle, we can use a new data structure to optimize this operation: the [[Disjoint-Sets | union-find]].

##### Kruskal's Algorithm (Fast)

The idea is to store the vertices in the disjoint-set, and incrementally applying union to them in order to keep track of their connected component. In this way checking for loops is equivalent to check if two vertices already belong to the same disjoint set (i.e. the same connected component).

```pseudo
\begin{algorithm}
\caption{KruskalFast}
\begin{algorithmic}
\Procedure{KruskalFast}{$G$}
	\State $A \gets \emptyset$
	\State $U \gets$ \Call{InitUF}{$V$}
	\State \Call{Sort}{$E$} by weight
	\ForAll{$e = (u, v) \in E$}
		\If{\Call{Find}{$u$} $\not=$ \Call{Find}{$v$}}
			\State $A \gets A \cup \{ e\}$	
			\Call{Union}{$u, v$}
        \EndIf
    \EndFor
	\Return $A$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

##### Complexity

Let's break down all the steps:
1. Initializing the UF data structure: $O(n)$
2. Sorting the edges by weight: $O(m \log m)$. Since in the worst case $m = n^2$, we have $\log m \leq \log n^2 = 2 \log n \in O(\log n)$. Thus this also is $O(m \log n)$
3. For loop does $O(m)$ iterations
	1. The two find operations have $O(\log n)$ complexity and they are invoked for each edge $\implies$ total complexity is $O(m \log n)$
	2. The union operation has complexity $O(\log n)$ and it is invoked only when we add an edge to the tree. Since in a tree $m = n-1 \implies$ it is invoked $n-1$ times  $\implies$ the total complexity is $O(n \log n)$. Since for a connected graph $m \geq n-1$ it is also true that the total complexity is $O(m \log n)$
	3. Updates to $A$ take constant time and are invoked $n-1$ times as above $\implies$ total complexity is $O(n)$

Thus the total complexity is:
$$
O(m \log n)
$$
which is identical to Prim's algorithm (fast implementation).

#### Properties

##### Uniqueness of MST
> Given a weighted graph $G = (V, E)$, where $\forall e, t \in E, e \neq t \implies w(e) \neq w(t)$ (i.e. weights are all distinct), then there exists exactly one MST.

**Proof**

Let's assume a graph $G = (V, E)$ having edges with all distinct weights, and two distinct MST of $G: A, B$.
Then, by construction, there is at least one edge in $A$ that is not in $B$, since they are distinct.

Now consider the minimum weight such edge: $e_{1} = min\{ e : e\in A \land e\not\in B \}$. It connects two vertices $u, v$.

Since $A, B$ are trees, there has to be a path in both of them from $u$ to $v$; thus, if we add $e_{1}$ to $B$, we obtain a cycle in $B$ that includes $u, v$. Let's call $B' = B \cup \{ e_{1} \}$

$A$ is, by hypothesis, an MST $\implies$ it does not have cycles and contains the edges with minimum possible weights. In the newly formed cycle in $B'$, there must be at least one edge $e_{2}$ that is in $B'$ but not in $A$ (otherwise $A$ would contain a cycle). Since $A$ is an MST containing $e_{1}$​ but not $e_{2}$​, $e_{1}$ is the smallest edge $\in A, \not\in B$ and all edge weights are distinct, we must have $w(e_{2})>w(e_{1})$.

Now we remove this $e_{2}$ from $B', B'' = B' \setminus \{ e_{2} \}$ obtaining a tree again, since if we remove an edge from a cycle the vertices remain connected. Since $w(e_{2}) > w(e_{1})$ this new $B''$ has lower weight than the original $B$: $w(B'') < w(B)$.

But this is a contradiction since $B$ is by hypothesis an MST.

The converse is not generally true. Suppose the starting graph $G$ is already a tree. In that case there is only one possible spanning tree (the tree itself), and it is by necessity the minimum one even if all weights are equal.

##### Second-Best MST
> Given a weighted graph $G = (V, E)$, where $\forall e, t \in E, e \neq t \implies w(e) \neq w(t)$ (i.e. weights are all distinct), the second-best MST, that is the second smallest spanning tree is not necessarily unique.

**Proof by counterexample**:

![[graph-second-best-mst.png]]

### Shortest Path

Given a _weighted_ graph $G$, the _length_ of a path $P = v_{1}, v_{2}, \dots, v_{k}$ is defined as:
$$
len(P) = \sum_{i=1}^{k-1} w(v_{i}, v_{i+1})
$$

A _shortest path_ from a vertex $v$ to a vertex $u$ is a path with minimum length among all possible $u-v$ path.

The _distance_ between two vertices $s$ and $t$ is the length of a shortest path from $s$ to $t$. If there is no path between them (i.e. they do not belong in the same connected component), then the distance is $+\infty$.
$$
dist(s, t) = \begin{cases}
min\{ len({P}) : P \text{ is a path between }s, t \} & \text{if } s, t \in C \\
+\infty & \text{otherwise}
\end{cases}
$$

Note that, if the graph is directed, in general $dist(u, v) \neq dist(v, u)$.

![[graph-sssp-exa.png]]

#### Single-Source Shortest Path (SSSP)

An SSSP algorithm aims to provide all the distances from a vertex $s$ (source) to all the nodes in the graph.

**Input**: A _directed_, _weighted_ graph $G$ with edge weights $w: E \to \mathbb{R}$ and a source vertex $s \in V$.

**Output**: All the distances $dist(s, v)\ \forall v \in V$.

Note that no algorithms are known for the previous problem that run asymptotically faster than the best SSSP algorithms in worst case. Also we will focus on directed graphs, but adapting to undirected ones is trivial.

We will start with the special case of _non-negative_ edge weights: $w: E \to \mathbb{R}_{\geq0}$.
If all the weights are identical, then the problem reduces to finding the path with less hops. 

We already know an algorithm that solves this problem in linear time: [[#Breadth First Search Algorithm|BFS]].
Actually, assuming that the weights are integers: $w: E \to \mathbb{N}$, we can always reduce a graph into one with unitary weights by replacing every edge with weight $k$ with $k$ edges of weight $1$. Then we can apply BFS and compute the answer.

![[graph-sssp-reduction.png]]

There are two main problems with this approach:
1. We need integer weights, so it is still a special case.
2. The reduction of the graph can create a much larger graph, so even if BFS is linear in the size of it, it will still be non-linear with respect to the original one.

##### Dijkstra's Algorithm

Dijkstra's algorithm is one of the most famous ones for finding the shortest paths from a source node to the others. It is a greedy algorithm that resembles very closely Prim's one, with the difference of considering the total length from the source instead of directly picking the smallest edge.
It only works for graphs with non-negative weigths.

This algorithm can be considered as a generalization of BFS for weighted graphs, where expansion does not go in order of less hops, but in order of shortest path.

**Input**: A weighted, directed graph $G = (V, E), w:E \to \mathbb{R}_{\geq{0}}$
**Output**: The distance from the source node $s$ to all the vertices in the graph.

```pseudo
\begin{algorithm}
\caption{Dijkstra}
\begin{algorithmic}
\Procedure{Dijkstra}{$G, s$}
	\State $X \gets \{s\}$
	\State $len(s) \gets 0$
	\ForAll{$v \in V$}
		\State $len(v) \gets +\infty$
    \EndFor
	\While{$\exists\ e = (u, v)$ with $u \in X \land v \not\in X$}
		\State $e' =(u', v') \gets$ an edge that minimizes $len(u') + w(e')$
		\State $X \gets X \cup \{v'\}$
		\State $len(v') \gets len(u') + w(e')$
    \EndWhile
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

![[graph-dijkstra-exa.png]]

One of the great advantages of this algorithm is that in each iteration it computes the _final_ distance to one additional node of the graph, despite having looked only at a part of it. The drawback is that this greediness makes it fail with graphs with negative cycles, because it will constantly pick the cycle indefinitely.o

###### Complexity

The complexity of the algorithms in this simple case is:
$$
O(n\cdot m)
$$
which is given by the size of the adjacency list that will be eventually traversed completely.

###### Correctness

Let's prove its correctness by induction showing that each time we add a vertex to $X$ we will have reached it by the shortest path (i.e. we have obtained its distance from the origin).

**Invariant**:
	$\forall\ x \in X$ , $len(x)$ is $dist(s, x)$.
**Base case**:
	$|X| = 1$ at the start by construction. $X = \{ s \}$ with $len(s) = 0 \implies$ the invariant holds.
**Inductive case**:
	Let's assume that the invariant holds for any $|X| = k \in \mathbb{N}$.
	Let $v$ be the next vertex added to $X$ at the $k+1$ iteration, and let $(u, v)$ the arc by which $v$ is reached.
	By construction $u$ has to be in $X$, and by the inductive hypothesis $len(u) = dist(s, u)$.
	Now, $len(v)$ will be updated as $len(v) \gets len(u) + w(u, v)$. By construction $(u, v)$ is picked as the edge that minimizes $len(u) + w(u,v)$.
	Since we have that $len(u) = dist(s, u)$, by adding the arc that minimizes the increment in distance from the source the new $len(v)$ will be the distance of $v$ from $s$, thus the invariant holds at the end of the $k+1$ iteration.

By induction this proves the correctness of the algorithm.

##### Dijkstra's Algorithm (Heaps)

To optimize the time complexity of the original algorithm we need to find a way to speed up the minimum edge search. In a similar fashon as with [[#^53d299 | Prim's algorithm]], we will put vertices in a min-priority heap, indexed by the current estimate for the distance from the source vertex.

```pseudo
\begin{algorithm}
\caption{Dijkstra}
\begin{algorithmic}
\Procedure{Dijkstra}{$G, s$}
	\State $X \gets \emptyset$
	\State $H \gets $\Call{EmptyHeap}{}
	\ForAll{$v \in V$}
		\State $key(v) \gets +\infty$
		\State \Call{HeapInsert}{$H, (key(v), v)$}
    \EndFor
	\State $key(s) \gets 0$
	\While{$H$ is not empty}
		\State $v \gets $ \Call{ExtractMin}{$H$}
		\State $X \gets X \cup \{v\}$
		\State $len(v) \gets key(v)$
		\ForAll{$e = (v, u) \in E$ such that $u \not\in X$}
		\State $key(u) \gets min\{key(u), len(v) + w(e)\}$
		\State \Call{UpdateKey}{$H, (key(u), u)$}
        \EndFor
    \EndWhile
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

###### Complexity

This new implementation has a much better time complexity:
1. Initialization cycle is executed $n$ times and every iteration takes $O(\log n)$ time for heap insertion $\implies$ the total complexity is $O(n \log n)$.
2. The while cycle over the priority queue is executed $n$ times.
	1. Extracting the minimum element from the min-heap takes $O(\log n)$ time $\implies$ total execution takes $O(n \log n)$ time.
	2. Adding the vertex to $X$ takes constant time.
	3. The internal for cycle is executed over the crossing edges, so it is $O(d(v))$. For each iteration, key computation and distance update takes $O(\log n)$ time $\implies$ considering that the outer while explores all the nodes we can say that the total execution time is $O\left( \sum_{v \in V}d(v) \log n \right ) = O(m \log n)$

Thus in total the complexity is:
$$
O((m + n)\log n)
$$
intuitively the complexity comes from traversing the entire graph ($O(m+n)$), and from updating the min-heap ($O(\log n)$).

Note: using [Fibonacci heaps](https://en.wikipedia.org/wiki/Fibonacci_heap), the complexity drops to:
$$
O(m + n\log n)
$$

##### Generalizing the SSSP Problem

We have now seen that Dijkstra's algorithm solves the SSSP problem for graphs where _all_ the edge weights are not negative.

But we would like to solve the more general problem where weights can assume any real value. This would allow us to represent things such as financial transactions with negative amounts, or operations where costs are negative (i.e. a reward).

In some cases there are graphs where the shortest path is not defined. This happens when the graph contains _negative cycles_, that is cycles where the total cost is negative:
$$
\begin{align}
&P = v_{1}, v_{2}, \dots, v_{k} = v_{1} \\
&c(P) = \sum_{i = 1}^{k - 1}{w(v_{i}, v_{i+1})} \\
&\text{if } c(P) < 0 \implies P \text{ is a negative cycle}
\end{align}
$$

![[graph-negative-cycle.png]]

For graphs without such cycles the SSSP is well-defined, but is generally _NP-hard_, that is no polynomial-time algorithm exists for it (as far as currently known).

Now we can make a further observation: a shortest path cannot contain ever a cycle, even if it is positive weight. Intuitively this comes from the fact that looping around to get back to the same place is useless at best if the weight is zero, and surely leads to a greater cost if total cycle weight is positive.

This means that we can completely ignore cycles and assume to compute _cycle-free shortest paths_, which have $\leq n-1$ edges (cycle-free graphs are forests [[#Properties of Graphs|see]]).

##### Bellman-Ford's Algorithm

We want to update the original Dijkstra's algorithm to deal with negative-weight edges. The intuition is that weights should be continuously updated for $n-1$ times checking over all edges. At the end all vertices will have the correct distance from the source. It obviously is slower than Dijkstra, but it is the price to pay for increased flexibility.

**Input**: A directed, weighted graph $G$, $w: E \to \mathbb{R}$ and a source vertex $s \in V$.
**Output**: Either $dist(s, v)\ \forall v \in V$, or a declaration that $G$ contains a negative cycle.

```pseudo
\begin{algorithm}
\caption{Bellman-Ford}
\begin{algorithmic}
\Procedure{BellmanFord}{$G, s$}
	\ForAll{$v \in V$}
		\State $len(v) \gets +\infty$
    \EndFor
	\State $len(s) \gets 0$
	\For{$n-1$ iterations}
		\ForAll{$e = (u, v) \in E$}
			\State $len(v) \gets min\{len(v), len(u) + w(e)\}$
        \EndFor
    \EndFor
	\ForAll{$e = (u, v) \in E$}
		\If{$len(v) > len(u) + w(e)$}
			\Return There is a negative cycle
        \EndIf
    \EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
```


![[graph-bellman-ford.png]]
###### Correctness

Let $len(i,v)$ be the length of a shortest path from $s$ to $v$ that contains no more than $i$ edges (arcs).
Since the shortest path from $s$ to $v$ contains no more than $n-1$ edges (imagine a tree with all the vertices connected in line resembling a linked list. It will have $n-1$ edges because its a tree and the shortest path will necessarily pass there), it is sufficient to prove that after $i$ iterations $len(v) \leq len(i, v)$. This means that every iteration adds an edge that improves the cost of the estimate, that is the cost after relaxation at the end of the iteration is better than (or equal) at the beginning.

By induction on the iterations $i$:
- **Base case**: 
	$$
i=0 \implies \begin{cases}
len(s) = 0 \leq len(0, s) = 0 \\
len(v) = +\infty = len(0, v)  & \forall v \in V, v\neq s
\end{cases}
$$
- **Inductive hypothesis**: $len(v) \leq len(k, v)\ \forall 1 \leq k < i$
	Take $i\geq 1$ and a shortest path from $s$ to $v$ with $\leq i$ edges. Let $e = (u, v)$ be the last edge of this path. Then, by splitting the path in two parts:
	$$
	len(i, v) = len(i-1, u) + w(u, v)
	$$
	Since we are assuming that we have a shortest path from $s$ to $v$ with at most $i$ edges, all its sub-paths are shortest paths, hence removing the last edge of the path we obtain a shortest path with at most $i-1$ edges from $s$ to $u$.

	Now by the inductive hypothesis $len(u) \leq len(i-1, u)$.
	In the $i$-th iteration we update:
	$$
	\begin{align}
	len(v) & = min\{ len(v),  & len(u) + w(u, v) \} \\
	 & & \leq len(i-1, u) + w(u,v)  \\
	 & & = len(i, v)  \\
	& \leq len(i, v)
	\end{align}
	$$
###### Complexity

The complexity is clearly:
$$
O(n\cdot m)
$$

#### All-Pairs Shortest Paths (APSP)

The further extension of [[#Single-Source Shortest Path (SSSP)]] is to find _for all_ vertices in the graph the distance _for all_ the other vertices (eventually infinite if a vertex is not reachable from another).

More formally:
**Input**: A directed, weighted graph $G = (V, E)$
**Output**: One of the following:
- $dist(u, v)\ \forall$ ordered vertex pairs 
- A declarationn that $G$ contains a negative cycle

The obvious solution would be to invoke [[#Bellman-Ford's Algorithm]] once for every vertex in $V$. The resulting complexity would then be $O(mn^2)$, which is quite bad.

Using ideas from dynamic programming we can actually improve the time complexity by splitting this problem in smaller sub-problems.

##### Bellman-Ford's Algorithm (dynamic programming formulation)
We can design a variant of the BF algorithm that uses dynamic programming.
As used in the demonstration we observe that sub-paths of shortest paths are shortest paths themselves, obviously to a different destination and, more importantly, with fewer edges.

More precisely let $P$ be a shortest path from $s$ to $v$. Then all the sub-paths $P' \subset P$ are shortest paths from the starting node of $P'$ to the final node (if $P'$ was not a shortest path then neither $P$ would be, contradicting the hypothesis).

So the idea is to create new subproblems by restricting the number of edges allowed in a path, with smaller subproblems having smaller edge budgets.

In fact lets define all the subproblems as computing $len(i, v)$ for any possible $i$ and any other vertex $v \in V$. Recall that $len(i, v)$ is the length of the shortest path from $s$ to $v$ with no more than $i$ edges.
From the subproblem definition we obtain that there are $n$ vertices $\cdot$ $n-1$ edges in the shortest path at most $\implies O(n^2)$ subproblems.

We observe that this formulation doesn't reduce the size of the input for each subproblem, but only the allowable size of the output, thus its complexity.

###### Bellman-Ford Recurrence

The recurrence formulation for BF in dynamic programming form is:
$$
len(i, v) = \begin{cases}
0 & \text{if } i=0 \land v=s \\
+\infty & \text{if } i = 0 \land v\neq s \\
min\begin{cases}
len(i-1, v) & \text{adding edges is useless} \\
min_{(u, v) \in E}\{ len(i-1, u) + w(u, v) \} & \text{otherwise}
\end{cases}
\end{cases}
$$

A variant of this formulation adapted for the APSP problem has complexity (not proved here)
$$
O(n^3\log n)
$$

##### Floyd-Warshall Algorithm

The problem with the [[#Bellman-Ford Recurrence]] formulation is that is doesn't reduce the input size for the subproblems, we want to find a way to do so in order to reduce the complexity further.

The idea is to, instead of limiting the number of edges allowed in the path calculation, restrict the _identities_ of the vertices allowed in a path (i.e. the paths can only pass through only certain vertices).

Let's define the subproblems:
Assume the vertices are labelled from $1, \dots, n$.
The subproblem is to compute $dist(u, v, k)$ which is the length of the shortest path from $u$ to $v$ that uses only vertices with label $\leq k$ as internal vertices, and does not contain a directed cycle. If no such path exists the distance is $+\infty$.

With this definition we obtain a number of subproblems:
$$
O(n^3) \gets \begin{cases}
n  & \text{sources} \\
n  & \text{destinations} \\
n + 1  & \text{subsets}
\end{cases}
$$

The algorithm consists in expanding iteratively the set of allowed vertices one at a time, until all the vertices are allowed.

The payoff of this formulation is that now there are only two candidates for the optimal solution to a subproblem, which consists of the path that picks the newly added vertex $k$ and the one that does not.

![[graph-fw-subproblems.png]]

Thus the complexity is constant for all the subproblems and the total complexity is:
$$
O(1) \text{ (subproblem complexity)} \cdot O(n^3) \text{ (number of subproblems)} \implies O(n^3)
$$

```pseudo
\begin{algorithm}
\caption{Floyd-Warshall}
\begin{algorithmic}
\Procedure{FloydWarshall}{$G$}
	\State\Call{LabelVertices}{$V$}
	\Comment{Label all vertices from 1 to n}
	\State $A \gets $Tensor $\in \mathbb{R}^{n \times n \times n+1}$
	\Comment{A[u][v][k] contains dist(u, v, k)}
	\For{$u \gets 1$ to $n$}
	\Comment{Initialize distances (k = 0)}
		\For{$v \gets 1$ to $n$}
			\If{$u = v$}
				\State $A[u, v, 0] \gets 0$
			\Elif{$e = (u, v) \in E$}
				\State $A[u, v, 0] \gets w(e)$
			\Else
				\State $A[u, v, 0] \gets +\infty$
			\EndIf
        \EndFor
    \EndFor
	\For{$k \gets 1$ to $n$}
	\Comment{Solve subproblems}
		\For{$u \gets 1$ to $n$}
			\For{$v \gets 1$ to $n$}
				\State $A[u, v, k] \gets min\{A[u, v, k-1], A[u, k, k-1] + A[k, v, k-1]\}$
	        \EndFor
        \EndFor
    \EndFor
	\For{$u \gets 1$ to $n$}
		\If{$A[u, v, n] \lt 0$}
			\Return "G has a negative cycle"
        \EndIf
    \EndFor
\EndProcedure
\end{algorithmic}
\end{algorithm}
```


Finding an algorithm faster that cubic for APSP (i.e. with complexity $O(n^{3-\epsilon})$) is still an open problem!