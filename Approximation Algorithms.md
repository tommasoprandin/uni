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