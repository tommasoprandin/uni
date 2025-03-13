A disjoint-set is a data structured used to efficiently manage disjoint sets of objects. It is also called _union-find_.

### Operations

There are three main operations supported:
1. **`Init`**: given a collection $X$ of objects, create a union-find data structure with each object $x \in X$ in its _own_ set.
2. **`Find`**: given any object $x$, return the name of the set that contains $x$ (if any).
3. **`Union`**: given two objects $x, y$ merge the sets that contain them into a single set. If $x, y$ already belong to the same set do nothing.

### Implementation

We will use an _array_, which can be visualized as a set of _directed trees_. Each object of the array has a $parent$ field that contains the index in the array of some object $y$ (parent of $x$). We assume that the objects to be stored are numbers, so that the search is in constant time (just to to array at position $x$).

This builds a forest of _directed trees_, each representing a disjoint set.

The name of the set is given by the root of the tree.

$$
\begin{array}{c | c}
\text{Index of } x & \text{Parent of } x \\
\hline 
1 & 4 \\
2 & 1 \\
3 & 1 \\
4 & 4 \\
5 & 6 \\
6 & 6
\end{array}
\begin{array}{c c c c c}

\end{array}
$$

![[ds-forest.png | center | 400]]

#### Init

The implementation is trivial: just add all the objects in the array and make their parent field point to themselves.

This has _complexity_: $O(n)$.

![[ds-init.png]]

#### Find

To implement the find method with this data structure, we first have to find the object in the array (simple direct access $O(1)$), and then travel its tree up until the root position where $r.parent = r$.
Then simply return $r$.

The complexity of find depends on the _depth_ of the object in its set tree (i.e. the number of arcs to travel in order to reach the root). For instance in the example above

$$
\begin{align}
 & depth(4) = 0  \\
 & depth(1) = 1 \\
 & depth(2) = 2
\end{align}
$$

The complexity has an upper bound of $O(n)$ in the worst-case.
This happens if the trees are in a particularly "bad shape" (e.g. all in trees with no branches).

Thus, in order to improve the complexity of the find, the implementation of the union operation is crucial.

#### Union

In this operation, given two objects $x, y$, we need to merge their trees into a single one (if they are disjoint).

The most simple way is to point one of the two roots to another node of the other tree. But which one?

The key is to minimize the height of the resulting tree, such that the find operation cost is minimized. For this reason it is clear we need to append the root to the other root, since in this way we minimize the increase in the height of the resulting tree.

Then we have to decide which tree gets merged into the other. There are two equivalent approaches:
1. Merge the smallest tree (i.e. with less nodes) into the largest
2. Merge the shortest tree into the tallest

We decide for option 1 since it is easier to analyze.

The steps are:
1. Invoke $Find(x)$ and $Find(y)$ to obtain the names $i$ and $j$ of the sets that contain $x, y$. If $i = j$ return since there is nothing to merge.
2. We need to add a $size$ field for every set. Then pick the smallest tree between $i, j$ and make its root point the root of the other. Then update the size of the newly built tree.

#### Complexity

Let's now analyze the complexity for find and union.
When we merge the smaller tree into the larger one, we need to track what happens to individual nodes and their depths in the tree.

For any particular node:

- Initially, it has depth 0 in its single-node tree
- Its depth only increases when its tree (the smaller one) gets merged into a larger tree
- When this happens, the node becomes part of a tree that's at least twice as large as its previous tree

A node's depth can only increase when its tree gets merged into a larger tree, and this can only happen $\log_{2}n$ times.

Let's track a node through successive merges:
1. Initially the node is in a tree of size 1
2. First depth-increasing merge: now in a tree of size at least 2
3. Second depth-increasing merge: now in a tree of size at least 4
4. Third depth-increasing merge: now in a tree of size at least 8

With each depth-increasing merge, the tree size at least doubles. Since we can't have a tree larger than $n$ nodes, we can write:

- $2^k \leq n$ (where k is the number of depth-increasing merges)
- Taking $\log_{2}$ of both sides: $k \leq \log_{2}(n)$

Therefore, no node can increase its depth more than $\log_{2}(n)$ times, which bounds the maximum tree height to $O(\log n)$.

Since the maximum tree height is bounded to $O(\log n) \implies$ the complexity of both find and union is $O(\log n)$.



