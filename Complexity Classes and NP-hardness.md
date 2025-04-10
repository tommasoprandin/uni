## Preamble: NP-Hardness

In the 30's we started to _understand_ what is or isn't _**effectively** computable_ (Computability theory).

By the 60's computer scientists had developed a bunch of fast algorithms to solve some problems, while for others the only known algorithm were very slow.

In the 70's we started to _understand_ what is or isn't _**efficiently** computable_.
In 1965 Edmonds defined what efficient means: an algorithm is _efficient_ if its _running time_ is $O(n^k)$ for some constant $k$, i.e. the time complexity is _polynomial_.

Problems for which polynomial time algorithm exists are called _tractable_, otherwise if no polynomial time algorithm exists the problem is _untractable_.

The border between _tractable_ and _untractable_ problems is quite perplexing, since even small modifications to a problem formulation can drastically change its computational complexity, for instance:

**Eulerian Circuit problem**: given an _undirected_ graph, an _Eulerian circuit_ is a cycle that crosses every edge _exactly once_.

**Hamiltonian Circuit problem**: given an _undirected_ graph, an _Hamiltonian circuit_ is a cycle that passes through every vertex _exactly once_. ^c21261

While the Eulerian circuit problem has an algorithm that solves it in linear time, for the Hamiltonian circuit problem there is no polynomial time algorithm, even if its formulation is very similar to the Eulerian circuit one.

Note that this does not mean that it doesn't exist such an algorithm, but no one could find one yet; we can't prove that it is an _inherently hard problem_...

Anyway we know a way to, given a _solution_ to the HC problem, verify if the solution is correct.

### Complexity Classes

To simplify the study of the complexity of problems, we will limit our attention to _decision problems_, that is problems with a boolean answer (e.g. _Is $x$ a prime number?_).

Then we define the following _complexity classes_:
- **P class**: is the set of decision problems that can be _solved_ in polynomial time (informally, for any problem, it belongs to P if it can be solved in polynomial time)
- **NP class**: is the set of decision problems with the following property: if the answer is affirmative, then there is a _proof_ of the answer that can be _verified_ in polynomial time (informally, for any problem, it belongs to NP it it can be _verified_ in polynomial time). It is a _less stringent_ property than P.
- **co-NP class**:  is the set of decision problems with the following property: if the answer is negative, then there is a _proof_ of the answer that can be _verified_ in polynomial time. It is the opposite of NP.
- **NP-hard class**: a problem is in this class, if a polynomial time algorithm for it would imply a polynomial time solution for _every_ problem in NP. This means that all the problems in NP reduce to an arbitrary problem in NP-hard.
- **NP-complete class**: a problem is NP-complete if it is _both_ NP-hard and in NP.

This is how we currently think the relations between classes are:
![[appalg-np-diagram.png]]

The question $P=NP$ is (arguably) the most important open problem in computer science.

#### Importance of NP-hard Problems

We will focus our study on NP-hardness because if a problem is NP-hard, then it is _very likely_ (**not guaranteed**) that it is intractable, i.e. there isn't a polynomial time solution.

Thus we need to find alternative approaches to solve it, by either:
- Identifying _special cases_ that are tractable, which reduces the generalizability of the problem
- Compromising on correctness, that is finding _approximate solutions_ that have an upper bound on the "quality" of the approximation.

An algorithm that finds an approximate solution of bounded "worsening" w.r.t the optimal solution, is called an _**approximation algorithm**_.

### SAT Problem

A _SAT(isfiability)_ problem is a decision problem that, given a boolean formula in input, determines whether there exist a possible combination of boolean values that makes the formula true.

**Input**: A propositional logic formula (Boolean expression) formed by:
- variables
- logic operators
	- AND
	- OR
	- NOT
	
For example:
$$
(b \land \neg c) \lor (\neg a \land b)
$$

**Output**: A boolean value that represents whether there exists an assignment of boolean values to the variables that make the formula evaluate to true.

#### Conjunctive-Normal Form

A boolean formula is in _conjunctive-natural form_ (CNF) if it is a _conjunction_ (AND) of several _clauses_.
A _clause_ is a _disjunction_ (OR) of several _literals_.
A _literal_ is either a _variable_ or its _negation_.

For example:
$$
\underbrace{ \underbrace{ (\underbrace{ a }_{ \text{literal} } \lor b \lor c) }_{ \text{clause} } \land (b \lor \neg c \lor \neg d) \land (\neg a \lor c \lor d)  }_{ \text{CNF formula} }

$$

#### 3-SAT

A 3-SAT is a special case of SAT, where the input is a CNF formula with _exactly_ 3 _literals_ per clause. The example above is a 3-CNF formula. The formulation remains the same as SAT.

##### Cook-Levin Theorem

The Cook-Levin theorem states that 3-SAT is NP-hard

### Reductions

A technique very commonly used to prove that a problem is NP-hard is _reduction_.

Given two problem $A, B$ we say that the problem $A$ reduces to $B$ $A\leq B$ if and only if there is an _effective_ algorithm that maps instances of $A$ into instances of $B$, and solutions to $B$ into solutions to $A$.
The algorithm that implements the transformation is called _reduction_.

In the following example we show the intuition of $A \leq B$:
![[appalg-reduction.png]]

If we impose that the reduction must also be _efficient_ (i.e. with polynomial complexity), then we can prove that a problem is _as hard as one another_.

For instance let's say we have a problem $A$ that we want to show to be NP-hard, and a known NP-hard problem $H$.

If we can prove that $H \leq A$ (i.e. we can find a reduction algorithm), then we have shown that $A$ is as hard as $H \implies A$ is NP-hard.

Formally:
#### Karp Reduction
**Definition**: a problem $A$ reduces in _polynomial time_ to problem $B$:
$$
A \leq_{P}B
$$

if there exists a _polynomial-time algorithm_ that transforms an arbitrary input instance $a$ of $A$ into an input instance $b$ of $B$ such that:
$$
a \text{ is a YES instance of } A \iff b \text{ is a YES instance of }B
$$

![[appalg-karp-red.png]]

Note that this is more restrictive than the general approach:
- There can only be one "call" to $B$
- There is **no postprocessing** of the output
- It is only defined for **decision problems**

Karp reduction is **transitive**:
$$
A \leq_{P} B, B\leq_{P} C \implies A \leq_{P}C
$$
this is very important because it allows us to give a formal NP-hardness definition.

**Definition (NP-hardness)**: A problem $A$ is _NP-hard_ if every problem $B \in$  NP, reduces in polynomial time to it:
$$
A \in \text{NP-hard} \iff \forall B\in\text{NP}\quad B\leq_{P}A
$$
that is $A$ is _at least as hard_ as any problem in NP.

To prove that a problem $X$ is NP-hard, we need to find a Karp reduction of a known NP-hard problem $P$: $P \leq_{P} X$. This would show that $X$ is _at least as hard_ as $P$. But $P$ being NP-hard makes $X$ NP-hard itself.

NP-hardness **does not mean** the problem is not in P, but it provides _strong evidence_ for that:
$$
X \in \text{NP-hard} \centernot\implies X \not\in \text{P}
$$

### Some NP-hard Problems

We already know some NP-hard problems such as the [[#^c21261 | Hamiltonian circuit problem]], and the [[#3-SAT]] problem.

Actually Karp showed that 3-SAT $\leq_{P}$ HC, thus proving that HC is NP-hard. 3-SAT was already known to be NP-hard by the [[#Cook-Levin Theorem]].

#### Travelling Salesman Problem

We will prove now that the TSP is NP-hard by reducing a known NP-hard algorithm (Hamiltonian circuit) to it.

In other words suppose we could solve TSP in polynomial time. Could then we solve HC just by appropriately transforming its input **in polynomial time** into one for TSP, and applying the algorithm for TSP?

First we need a decision formulation for TSP, which is easy to define:
**Input**:
- A _complete_, _undirected_, _weighted_ graph $G = (V, E)$.
- A cost ceiling $k \in \mathbb{R}$

**Output** Is there in $G$  any Hamiltonian circuit of cost $\leq k$?

Then we define a suitable transformation from an instance of HC to an instance of TSP:

For HC the input is an _undirected_, _unweighted_ graph $G = (V, E)$.
We need to obtain a _complete_ and _weighted_ graph $G' = (V, E')$ for TSP.

So we add all the edges needed to make the graph $G'$ connected and we assign their weights in this way:
$$
w(e \in E') = \begin{cases}
1 & \text{if } e \in E \\ \\
+\infty & \text{otherwise}
\end{cases}
$$
because for HC we only care about number of steps, so we make the weights unitary, and we make the graph complete by adding edges with infinite weights, so they will never be picked by the algorithm if there are alternative options.

Lastly we need to find a suitable $k$. But this is easy to do, since the length of an HC is surely $n$ (exactly one step out of every node).

This reduction takes $O(n^{2})$ time to add the missing edges (and assign the weights). In a complete graph there are $O(n^{2} )$ edges.

Now we have to prove the correctness of the reduction. We have two cases, one for the YES instace and one for the NO.
1. (YES) If $G$ has an HC, then the TSP algorithm executed on $G'$ will return a HC of cost $n$. Thus it will only have used the edges originally in $E$, or the cost would be $+\infty$.
2. (NO) If $G$ does not have an HC, then any HC in $G'$ will have at least one edge not in $E$. But all the edges not in $E$ have cost $+\infty$. In this case TSP executed on $G'$ would return an HC of cost $>n$.

#### Maximally Independent Set

**Definition**: Given a graph $G = (V, E)$ an _independent set_ is a subset $I \subseteq V$, with _no edges_ between any node in it.

The _maximally independent set_ computes an independent set of maximum size.

We will show that 3-SAT $\leq_{P}$ IS, thus that IS is NP-hard, they may seem totally unrelated problems but there is a clever reduction between them.

Let's see the main idea:
Pick an arbitrary 3-CNF formula $f$ with $k$ clauses:
$$
f = (a \lor b \lor c) \land (b \lor \neg c \lor \neg d) \land (\neg a \lor c \lor d) \land(a \lor \neg b\lor \neg d)
$$

We will map every literal to a vertex in $G$.

Then we will add edges between _logically dependent_ literals:
1. Add an edge between every pair of edges making _conflicting_ requests, i.e. connect together literals and their negations.
2. Connect together all the edges in the same group, so that we force to choice just _one_ literal per group.

![[appalg-ind-set.png]]

Now if we can find an independent set of at least size $k$ then the formula is satisfiable, because intuitively we can assign whatever value we want to at least one literal per group. But we only need one "true" _per group_ to make the entire formula "true" (because there are only $\lor$ inside group and $\land$ between them).

The claim is that:
$$
G \text{ contains an independent set of size } k \iff \text{the formula } f \text{ is satisfiable}
$$

**Proof**:
1. $G$ contains an independent set of size $k \implies$ the formula $f$ is satisfiable.

	As mentioned before, by how the graph is constructed, having an independent set of size $k$, it means  that there is one literal per group free to be set with the desired value (true). This happens because all the literals in the same group are connected together (hence only one per group can be selected in the independent set), and also "conflicting" literals (hence only one can belong to the independent set). 
	But if we can freely set true to one literal per group than the formula is satisfiable since it is the conjunction of $k$ disjunction groups.

2. The formula $f$ is satisfiable $\implies$ $G$ contains an independent set of size $k$

	If the formula $f$ is satisfiable it means there is an assignment where at least one literal per clause is set to true, which means that we can pick exactly one vertex per group that can be set to true in the graph $G$.
	By construction of the graph the vertices selected will be independent, since there is only one per group, and conflicting literals are connected together.

#### Clique

**Definition**: Given an undirected, unweighted graph $G = (V, E)$, a _clique_ is a complete subgraph of $G: C \subseteq G, \forall v \in V_{C}\ \exists (v, u)\ \forall u \in V_{C}$.

The _clique problem_ is to find the _largest_ clique in $G$.

We will show that the CP is NP-hard by reducing the [[#Maximally Independent Set]] problem to it.

![[aalg-clique.png]]

**Claim**: $IS \leq_{P} CP$

We need to find a suitable transformation from the independent set problem to clique.

The independent set problem decides: "Is there an independent set of size $k$ in $G$?".
The clique can be formulated as a decision problem: "Is there a clique of size $k$ in $G$?".

We observe that, given an undirected, unweighted graph $G = (V, E)$:
$$
IS \subseteq V \text{ is independent} \iff IS \text{ is a clique in the complementary set } G'=(V, \bar{E})
$$
This happens because clique is the dual problem of independent set (one is the set with _all_ the edges between them, and the other is the set with _no_ edges between them).

With this observation the reduction is very straightforward:
Let $G$ be the input graph, we compute $G' = (V, \bar{E})$ and we feed it into the CP algorithm. This transformation takes polynomial time.

Hence $IS \leq_{P} CP \implies CP$ is NP-hard.

#### Vertex Cover

Given an undirected, unweighted graph $G = (V, E)$, a _vertex cover_ $VC \subseteq V$ is a set of nodes that includes at least one _endpoint_ of every edge in the graph.

The _vertex cover problem_ is to find the smallest set of vertices that covers the graph.

![[aalg-vertex-cover.png]]

We will prove that VC is NP-hard by reducing IS to it.

**Claim**: $IS \leq_{P} VC$

To find a suitable reduction we make this observation:
$$
IS \subseteq V \text{ is an independent set} \iff V \setminus IS \text{ is a vertex cover}
$$
Intuitively an independent set do not have any edge in common, this means that every edge in $E$ will have _at least_ one endpoint not in $IS \implies \in V\setminus IS$.
But the definition of vertex cover is a subset of vertices such that for every edge in the graph at least one endpoint is in the subset, which is exactly the definition of $V\setminus IS$ by construction.

So the original problem asks: "Is there an independent set of size $k$?".
We can solve it by answering: "Is there a vertex cover of size $n - k$?".

The reduction is in constant time, hence $IS \leq_{P} VC \implies$ VC is NP-hard.