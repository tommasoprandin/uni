**Randomized Algorithms** are algorithms that take non-deterministic choices, which may seem paradoxical, but are actually extremely useful to either:
1. **Improve performance**
2. Provide **efficiently solutions** to complex problems with some degree of possible error 

### Example 1: Randomized Quicksort

Quicksort is the de-facto best solution to sort an array. It is normally taught by choosing the pivot deterministically (e.g. the middle one). In the worst case, for specific inputs, the choice of the pivot in the middle may be very bad and make the complexity of the algorithm $O(n^{2})$.

Instead, by choosing the pivot at random, we can show that with **very high probability** the algorithm will take good decisions.
Additionally, in the deterministic case, there is a clear way to craft a "bad" input for the algorithm. For the randomized one there is no way to know the "bad" instance in advance.

### Example 2: Polynomial Identities

In this case we would like to establish if two polynomials $H(x)$ and $G(x)$ are identical:
$$
\underbrace{ (x+1)(x-2)(x+3)(x-4)(x+5)(x-6) }_{ H(x) } \equiv \underbrace{ x^{6} - 7x^{3} + 25 }_{ G(x) }
$$

The simplest way would be to convert both polynomials in the canonical form $\sum_{i=1}^{deg(P)}c_{i}x^{i}$, then compare if the coefficients are all identical.

Given $d =$ maximum degree of the polynomial, the complexity is $O(d^{2})$.

A possible randomized alternative would be to choose a number $r$ at random, compute both $H(r)$ and $G(r)$ and return if they are identical or not.

This clearly always works if the polynomials are equivalent.
If they are not we may have a _false positive_, in the case $r$ is a _root_ for $P(x) = H(x) - G(x)$ (i.e. is a solution for the equation).

Assuming $r \in \{ 1, 2, \dots, 100d \}$, then:
$$
Pr(\text{algorithm fails}) \leq \frac{d}{100d} = \frac{1}{100}
$$
since we have (at most in $\mathbb{R}$) $d$ roots for $P(x)$.

This is actually quite small, but not really satisfactory; ideally we'd like to have a way of reaching arbitrary precision.

Clearly to reduce the probability of error the straightforward way is to **run the algorithm multiple times**, for instance here we may try 10 different $r$'s and return true only if all the 10 tries are successful:
$$
Pr(\text{algorithm fails}) \leq \left( \frac{1}{100} \right)^{10} < 2^{-64}
$$

### Classification of Randomized Algorithms

Randomized algorithms are classified into two categories:
1. **Las Vegas algorithms**: algorithms that _never fail_. Their randomness has impact on their complexity, that is $T(n)$ is a random variable, and we are interested in either its expected value, or in the probability of getting a certain complexity:
	$$
	Pr(T(n) > c\cdot f(n)) \leq \frac{1}{n^{k}}
	$$ ^5fa633
2. **Monte Carlo algorithms**: algorithms that _may fail_. In this case we want to study this failure probability as function of the input size $n$
	$$
	Pr((i, s) \not\in \Pi)
	$$
	For decision problems they can be divided into:
	- _one-sided_: may fail only on one answer
	- _two-sided_: may fail on both

For instance [[#Example 1 Randomized Quicksort|randomized quicksort]] is a Las Vegas algorithm, while [[#Example 2 Polynomial Identities|polynomial identities]] is a Monte Carlo (one-sided) algorithm.

### "With High Probability"

Here we will provide a formal definition of the "high probability" term:

> **Definition**: Given a problem $\Pi \subseteq I \times S$, an algorithm $A_{\Pi}$ has complexity $T(n) = O(f(n))$ **with high probability**, if $\exists c, d > 0$ such that, $\forall i \in I, |i| = n$:
> $$\begin{gather}
Pr(A_{\Pi}(i) \text{ terminates in } > c\cdot f(n) \text{ steps}) \leq \frac{1}{n^{d}} \\
\implies A_{\Pi}(i) \in O(f(n)) \text{ with probability} > 1-\frac{1}{n^{d}}
\end{gather}$$


> **Definition**: Given a problem $\Pi \subseteq I \times S$ an algorithm $A_{\Pi}$ is correct **with high probability**, if $\exists$ a constant $d > 0$, such that $\forall i \in I, |i| = n$:
> $$\begin{gather}
Pr((i, A_{\Pi}(i)) \neq \Pi) \leq \frac{1}{n^{d}} \\
\implies (i, A_{\Pi}(i) \in \Pi) \text{ with probability} > 1-\frac{1}{n^{d}}
\end{gather}$$

The concept of high probability is directly linked with the **expectation**: with an high enough probability the expected value of  the complexity (or error rate) tends to the most favourable outcome.

To illustrate an example let's first briefly recall the **Markov's lemma**:

> **Markov's lemma**: Let $T$ be a non-negative, bounded ($\exists b \in \mathbb{N}$ such that $Pr(T > b) = 0$) and integer random variable. Then $\forall t, 0\leq t\leq b$:
> $$
\begin{gather}
t\cdot Pr(T\geq t)\leq \mathbb{E}[T] \leq t + (b-t)Pr(T \geq t) \\
\text{Common form: } Pr(T \geq t) \leq \frac{\mathbb{E}[T]}{t}
\end{gather}$$

^347b69

**Example**

Assume to have a Las Vegas algorithm $A_{\Pi}$ with $T_{A_{\Pi}}(n) = O(f(n))$ w. h. p.; and with worst-case deterministic complexity $O(n^{a}), a \leq d\ \forall n$.

We will show that $\mathbb{E}[T_{A_{\Pi}}(n)] = O(n)$.

Apply Markov's lemma:
$$
\begin{align}
\mathbb{E}[T_{A_{\Pi}}(n)] &  \leq \underbrace{ c\cdot f(n) }_{ t } + \underbrace{ (n^{a} - c\cdot f(n)) }_{ b - t } \cdot \underbrace{ \frac{1}{n^{d}} }_{ Pr(T \geq t) } \\
 & \leq c\cdot f(n) + \frac{n^{a}}{n^{d}} \\
 & \leq c\cdot f(n) + 1  & a \leq d\\
 & = O(f(n))
\end{align}
$$

### Karger's Algorithm for Minimum Cut

The _Minimum Cut_ problem consists in finding a **cut of minimum size**, or in other words the minimum number of edges whose removal disconnects the graph.

We will see the solution to a more general problem: _minimum cut on multigraphs_.

> **Definition**: A _multiset_ is a collection of objects with **repetitions allowed**:
> $S = \{ \{ \text{objects} \} \}$
> $\forall o \in S$ we can define its _multiplicity_ (i.e. number of copies in $S$) $m(o) \in \mathbb{N} \setminus \{ 0 \}$

> **Definition**: A _multigraph_ $\mathcal{G} = (V, \mathcal{E})$ is a graph with repeated edges allowed, that is $\mathcal{E}$ is a multiset of edges

![[aalg-multigraph.png]]

Clearly a simple graph $G = (V, E)$ is also a multigraph.

> **Definition**: Given a multigraph $\mathcal{G} = (V, \mathcal{E})$ connected, a cut $C \subseteq \mathcal{E}$ is a multiset of edges such that $\mathcal{G}' = (V, \mathcal{E}\setminus C)$ is _not connected_

The steps of Karger's algorithm are:
1. Choose an edge at random
2. "Contract" the two vertices of that edge, that is remove all the edges incident **both** vertices and replace the pair with a single new vertex.
3. Repeat until only 2 vertices remain
4. Return the edges between the remaining vertices.

![[aalg-karger-exa.png]]

Let's now formally define the contraction operation:
> **Definition**: Given $\mathcal{G} = (V, \mathcal{E})$ and $e = (u, v) \in \mathcal{E}$, the **contraction** of $\mathcal{G}$ with respect to $e$, $\mathcal{G}_{/e} = (V', \mathcal{E'})$, is the multigraph with:
> $$
\begin{align}
& V' = V \setminus \{ u, v \} \cup \{ z_{u, v} \}\\
& \mathcal{E'} = \mathcal{E}\setminus \{ \{ (x, y) \mid x = u \lor x= v \} \} \cup \{ \{ (z_{u, v}, y) \mid (u, y) \lor (v, y) \in \mathcal{E}, y \neq u \land y \neq v \} \} 
\end{align}$$
> The first step is to remove the original endpoints and replace it with a new collapsed vertex.
> The second step is to remove all the edges involving the old endpoints $u, v$, then restoring the connections of the new vertex to the graph.

We observe that at each contraction step:
- $|V'| = |V| - 1$
- $|\mathcal{E'}| = |\mathcal{E}|-\underbrace{ m(e) }_{ \geq 1 } \leq |\mathcal{E}| - 1$

Hence we need $n-2$ iterations to contract the entire graph.

```pseudo
\begin{algorithm}
\caption{Full-Contraction Algorithm}
\begin{algorithmic}
\Procedure{FullContraction}{$\mathcal{G}$}
	\For{$i = 1$ to $n - 2$}
		\State $e \gets$ \Call{Random}{$\mathcal{E}$}
		\State $\mathcal{G'} = (V', \mathcal{E'}) \gets \mathcal{G}_{/e}$
		\State $V \gets V'$
		\State $\mathcal{E} \gets \mathcal{E'}$
    \EndFor
	\Return $\mathcal{E}$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

![[aalg-fc-exa1.png]]
![[aalg-fc-exa2.png]]

As we can see in the example the contraction may fail, that is may not return the true min-cut.

The Karger algorithm simply repeats the full contraction $k$ times to reduce the probability of error. Obviously we have to determine the appropriate $k$ to get high probability.

```pseudo
\begin{algorithm}
\caption{Karger's Algorithm}
\begin{algorithmic}
\Procedure{Karger}{$\mathcal{G}, k$}
	\State $mincut \gets \mathcal{E}$
	\For{$i = 1$ to $k$}
		\State $t \gets$ \Call{FullContraction}{$\mathcal{G}$}
		\If{$|t| \lt |mincut|$}
			\State $mincut \gets t$
        \EndIf
    \EndFor
	\Return $mincut$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

#### Analysis

> **Property**: $\forall$ cut $C' \in \mathcal{G}_{/e}$, $\exists$ a cut $C$ in $\mathcal{G}$ of the same cardinality $\implies |mincut \in \mathcal{G}_{/e}| \geq |mincut \in \mathcal{G}|$

**Proof**:
Starting from a cut $C' \in \mathcal{G}_{/e}$ we will determine the corresponding cut $C \in \mathcal{G}$:
$$
C' \in \mathcal{G}_{/e = (u, v)} \to C \in \mathcal{G}
$$

To do this, we will substitute each edge $(z_{u, v}, y) \in C'$ with either $(u, y)$ or $(v, y)$, in order to maintain the same shape of the cut. Essentially we are undoing the contraction.
![[aalg-karger-analysis.png]]
This clearly does not change the cardinality of the cut: $|C'| = |C|$.

It remains to show that $C$ is a cut for $\mathcal{G}$.
$C' \in \mathcal{G}_{/e} \implies C'$ separates $V'$ in two _connected components_. Let $V_{1} \subset V'$ the connected component containing $z_{u,v}$, and let $x \not\in V_{1}$.
In $\mathcal{G}_{/e}$ every path from $z_{u, v}$ and $x$ must have an edge in the cut $C'$ by definition.

Now assume by contradiction that $C$ is not a cut in $\mathcal{G}$.
This would imply that there exists a path between $u$ and $x$ after the removal of $C$ in $\mathcal{G}$.
But this would mean that the path between $z_{u, v}$ and $x$ "survives" the removal of $C'$ in $\mathcal{G}_{/e}$, that is $C'$ is not a cut for $\mathcal{G}_{/e}$: contradiction.

If we continuously iterate the statement, the property remains valid for all the full contraction.
![[aalg-karger-analysis2.png]]

Now we observe that the cuts that disappear in the contraction $\mathcal{G}_{/e}$ are those hit by the random choice, i.e. the one between the contracted vertices.

The intuition is that since $|mincut|$ is small compared to the full set of edges $|\mathcal{E}|$, then the probability to hit them will be relatively small.

> **Property**: Let $\mathcal{G} = (V, \mathcal{E}), |V| = n$. If $\mathcal{G}$ has a min-cut of size $t$, then $|\mathcal{E}|\geq t \frac{n}{2}$.

**Proof**:
For all vertices $v \in V$, their degree must be $\geq t$, or the size of the min-cut would not be $t$. Then:
$$
\begin{gather}
\sum_{v \in V}d(v) = 2|\mathcal{E}| \leq \sum_{v \in V}t = nt \\
\implies |\mathcal{E}| \leq \frac{tn}{2}
\end{gather}
$$

Since the shape of the contraction depends on previous executions, we will use [[Probability Cheat Sheet#Conditional Probability|conditional probabilities]].

We will now prove that:
$$
Pr(\text{Full-contraction returns a min-cut}) \geq \frac{2}{n(n-1)}
$$

**Proof**:
Although there may be more than one min-cuts, we will prove that, for any min-cut $C$, the probability that the algorithm returns that particular min-cut $C$ is at least $\frac{2}{n(n-1)}$.

Let $C$ be some specific min-cut, and:
- $t = |C|$
- $E_{i} =$ "In the $i$-th contraction we did not hit an edge of $C$"

Now we compute the probability:
$$
\begin{align}
 & Pr(\neg E_{1}) = \frac{t}{|\mathcal{E}|} \leq \frac{t}{\frac{tn}{2}} = \frac{2}{n} \\
 & Pr(E_{1}) = 1-Pr(\neg E_{1}) \geq 1-\frac{2}{n} \\
 & Pr(E_{2} | E_{1}) \geq 1-\frac{t}{\frac{t(n-1)}{2}} = 1 - \frac{2}{n-1} \\
 & \vdots  \\
 & Pr(E_{i}|E_{1} \land E_{2} \land \dots \land E_{i-1}) \geq 1 - \frac{t}{\frac{t(n-i + 1)}{2}} = 1- \frac{2}{n - i + 1}
\end{align}
$$

This shows that by going through contractions we decrease the probability of not hitting an edge in the min-cut.

The success probability is:
$$
\begin{align}
 Pr(\text{Full-contraction succeeds})  & = Pr\left( \bigcap_{i=1}^{n-2}E_{i} \right) \\
  & \geq \prod_{i=1}^{n-2} \left( 1-\frac{2}{n-i+1} \right)   \text{(Lower bound of ind. events)} \\
 & = \prod_{i=1}^{n-2} \frac{n-i-1}{n-i+1} \\
 & = \frac{\bcancel{n-2}}{n}\cdot \frac{\bcancel{n-3}}{n-1} \dots \frac{2}{\bcancel{4}} \cdot\frac{1}{\bcancel{3}} \\
 & = \frac{2}{n(n-1)}
\end{align}
$$

Which is quite low, but with multiple tries we can obtain the desired result:
$$
Pr(\text{k runs of FC all fail}) \leq \left( 1-\frac{2}{n^{2}} \right)^{k}
$$
the goal is to make:
$$
\left( 1-\frac{2}{n^{2}} \right)^{k} \leq \frac{1}{n^{d}} \text{for some } d, k
$$

We will use this inequality:
> $$
\left( 1 + \frac{x}{y} \right)^{y} \leq e^{x}\quad y\geq 1, y\geq x$$

So:
$$
\begin{align}
\left( 1-\frac{2}{n^{2}} \right)^{k = n^{2}}  & \leq e^{-2} = \frac{1}{e^{2}} \\
\left(\left( 1-\frac{2}{n^{2}} \right)^{n^{2}}\right)^{\ln n^{d}/2}  & \leq \left(\frac{1}{e^{2}}\right)^{\ln n^{d}/2} \\
\left( 1-\frac{2}{n^{2}} \right)^{k = dn^{2}\log n/2}  & \leq \frac{1}{n^{d}} \\
\implies Pr(\text{Karger fails}) > 1-\frac{1}{n^{d}}
\end{align}
$$
assuming we make $k = \frac{dn^{2}\log n}{2} = O(n^{2}\log n)$ tries

Hence the complexity of the algorithm comes from doing $k$ times the full contraction.

Full contraction has complexity $O(n^{2})$
$\implies$ Karger has complexity $O(n^{4}\log n)$

To speed-up the algorithm the idea is to not repeat the first $\frac{n}{\sqrt{ 2 }}$ contractions, since they are the one contributing the most to the failure rate $\implies O(n^{2} \log ^{3}n)$ with the same correctness.

In 2020 an algorithm was published with complexity $O(m \log n)$.

### Chernoff Bounds

Chernoff's bounds are tools from modern probability theory allow us to analyze more tightly randomized algorithms, compared to [[#^347b69|Markov's lemma]].

They are grounded on the phenomenon of **concentration of measure**, which states that by combining a large number of sufficiently independent random variables, the results will sharply concentrate around the expectation.

In many cases, the study of either the complexity or error probability can be formalized as studying the distribution of a sum of random indicator variables.

Formally we have Bernoulli random variables representing the experiments:
$$
x_{i} \sim Ber(p)
$$
and a Binomial random variable representing the outcome of all the experiments:
$$
X = \sum_{i=1}^{n} x_{i} \sim B(p, n)
$$
We know that:
$$
\mathbb{E}[X] = np
$$

> **Definition (Chernoff bound)**: Let $x_{1}, \dots, x_{n}$ be _independent_ indicator random variables, $\forall i \in 1,\dots ,n\quad \mathbb{E}[x_{i}] = p_{i}, 0\leq p_{i} \leq 1$. Let $X = \sum_{i=1}^{n}x_{i}$ and $\mu = \mathbb{E}[X]$. Then, $\forall \delta > 0$:
> $$
Pr(X > (1+\delta)\mu) < \left( \frac{e^{\delta}}{(1+\delta)^{(1+\delta)}} \right)^{\mu}$$

**Example**
Let's suppose we want to analyze coin flips results, in particular, given $n$ coin flips, we want to calculate what's the probability of getting more that $\frac{3}{4}n$ heads.

Let:
$$
\begin{gather}
n \text{ coin flips } \to x_{1}, \dots, x_{n} \sim Ber\left( p = \frac{1}{2} \right) \\
X = \sum_{i=1}^{n} x_{i} \sim B(p, n) = \text{number of heads in } n \text{ coin flips} \\
\mathbb{E}[X] = \mu = np = \frac{n}{2}
\end{gather}
$$

By applying [[#^347b69|Markov's lemma]]:
$$
Pr\left( X > \frac{3}{4}n \right) \leq \frac{\mathbb{E}[X]}{\frac{3}{4}n} = \frac{\frac{n}{2}}{\frac{3}{4}n} = \frac{2}{3}
$$

By applying the Chernoff bound:
$$
Pr\left( X > \underbrace{ \left( 1+\frac{1}{2} \right)\mu }_{ = \frac{3}{4}n } \right) < \left( \frac{e^{1/2}}{\frac{3}{2}^{3/2}} \right)^{n/2} < 0.95^{n}
$$

Which is not only tighter, but grows exponentially with the number of tries, instead of being constant.

There are other variants of the Chernoff bound, which are in practice easier to use:
$$
\begin{align}
 & Pr(X < (1-\delta)\mu) < e^{-\frac{\mu\delta^{2}}{2}} & \delta > 0 \\
 & Pr(X < (1+\delta)\mu) < e^{-\frac{\mu\delta^{2}}{2+\delta}} & \delta > 0 \\
\end{align}
$$

#### Application of Chenoff Bound: Analysis of Randomized Quicksort

Let first illustrate the _randomized quicksort_ algorithm:
```pseudo
\begin{algorithm}
\caption{Randomized Quicksort Algorithm}
\begin{algorithmic}
\Procedure{RandQuickSort}{$S$}
	\If{$|S| \leq 1$}
		\Return $S$
    \EndIf
	\State $p \gets$ \Call{Random}{$S$}
	\Comment{Pivot is picked at random with uniform distribution}
	\State $S_1 \gets \{ x \in S \mid x < p\}$
	\Comment{$O(n)$}
	\State $S_2 \gets \{ x \in S \mid x > p\}$
	\Comment{$O(n)$}
	\State $Z_1 \gets $ \Call{RandQuickSort}{$S_1$}
	\State $Z_2 \gets $ \Call{RandQuickSort}{$S_2$}
	\Return $<Z_1, p, Z_2>$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

This is a [[#^5fa633|Las Vegas]] algorithm (always gives correct answer, complexity is not deterministic).

If we suppose that $p$ is _always_ the optimal choice (i.e. the median), then:
$$
\begin{align}
 & T_{RQS}(n) = \begin{cases}
2T_{RQS}\left( \frac{n}{2} \right) + O(n) & n > 1 \\
0 & n \leq 1
\end{cases} \\
 & \implies T_{RQS} = O(n \log n)
\end{align}
$$
since the size of the recursion tree is always $\lceil \log_{2} n \rceil$.

However, p is the median with very low probability $\frac{1}{n}$.
Note that, while its possible to find the median in $O(n)$ time, the algorithm is very complicated and with big hidden constant, hence inefficient in practice.

The _intuition_ is that we don't really need _exactly_ the median, a more relaxed bound is fine as long as the sizes of the two subarrays is not too unbalanced.

Let's try with this:
$$
\begin{cases}
|S_{1}| \leq \frac{3}{4}n \\
|S_{2}| \leq \frac{3}{4}n
\end{cases}
$$

Visually the pivot is chosen in the interval of values between $\frac{n}{4}$ and $\frac{3}{4}n$. Hence we have a range of options with $\frac{n}{2}$ values.

Now let's analyze the resulting recursion tree:
![[aalg-randqs-rectree.png]]
We notice that:
1. The total work at each level is $\leq c\cdot n$
2. The **depth** of the recursion tree is, since in the worst branch we always get the part with size $\frac{3}{4}$, $\left\lceil  \log_{\frac{4}{3}}n  \right\rceil=O(n)$.

Hence $\implies T_{RQS}(n) = O(n \log n)$
As we have seen, to get good asymptotical complexity, we don't need perfectly balanced subarrays. This leaves us with $\frac{n}{2}$ "good choices".
Now we are left to prove that the algorithm will pick the good pivots with high probability.

Let's define:
$$
\begin{align}
 & E = \text{"Pivot in good range chosen"} \\
 & Pr(E) = \frac{1}{2}
\end{align}
$$

We then fix _one_ root-leaf path $P$:
> **Lemma**: $Pr\left( |P| > a\cdot \log_{\frac{4}{3}}n \right) \leq \frac{1}{n^{3}}$

**Proof**:
![[aalg-randqs-path.png]]
We observe that the length of the path $|P|$ exceeds $l = a \log_{\frac{4}{3}}n$ only if in the first $l$ nodes of $P$ there have been $< \log_{\frac{4}{3}}n$ lucky choices.

Now define:
$$
\begin{align}
 & x_{i} = 1 & \text{ if a the }i-th \text{ vertex there was a lucky choice} \\
 & Pr(x_{i} = 1 ) = \frac{1}{2} \\
 & 0\leq i\leq l = a \log_{\frac{4}{3}}n \\
  & \text{All }x_{i} \text{ are independent} \\
 & X = \sum_{i=1}^{l} x_{i} \sim B\left( p = \frac{1}{2}, n = l \right) \\
 & \implies \mu = \mathbb{E}[X] = np = \frac{a}{2}\log_{\frac{4}{3}}n
\end{align}
$$

Then, by using Chernoff bound we can bound:
$$
\begin{align}
Pr\left( X < \log_{\frac{4}{3}}n \right)  & < e^{-\frac{\mu\delta^{2}}{2}} \\
 & < \frac{1}{n^{3}} & \text{for }a = 8, \delta=\frac{3}{4}
\end{align}
$$

Now we will apply a very popular result:
> **Lemma (Union bound)**: For any random events $E_{1}, \dots, E_{k}$:$$
Pr(E_{1} \cup E_{2} \cup \dots \cup E_{k}) \leq \sum_{i=1}^{k} Pr(E_{i})$$

![[aalg-union-bound.png]]
(i.e. the union of events may overlap, reducing total covered area)

We are now ready to prove the complexity of RQS:

$$
Pr\left( \text{all root-leaf paths have length} \leq a \log_{\frac{4}{3}}n \right) = 1 - Pr\left( \exists \text{ path} > a\log_{\frac{4}{3}}n \right)
$$

Define $E_{i} =$ path $P_{i}$ has length $> a\log_{\frac{4}{3}}n$

Now bound:
$$
\begin{align}
Pr\left( \exists \text{ path} > a\log_{\frac{4}{3}}n \right) & = Pr\left( \bigcup_{i=1}^{n}E_{i} \right) \\
 & \leq \sum_{i=1}^{n}Pr(E_{i}) & \text{(union bound)} \\
 & < n \cdot \frac{1}{n^{3}} & \text{(lemma)} \\
 & = \frac{1}{n^{2}} \\
 & \implies Pr(\text{Good choices}) \geq 1- \frac{1}{n^{2}}
\end{align}
$$

Finally:
$$
\implies T_{RQS}(n) = O(n \log n) \text{ w.h.p.}
$$