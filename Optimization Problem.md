An _optimization problem_ $\Pi$ is defined by:
- A set of _instances_ (i.e. possible inputs) for the problem $I$
- A set of _feasible solutions_ (i.e. correct) $S_{i}$ for all the instances $i \in I$
- An _objective function_ $\Phi: S \to \mathbb{R}$

The _problem_ is, given an instance $i \in I$ for the problem $\Pi$, find a _optimal solution_ to the problem $s \in S_{i}$ such that:
- $\Phi(s) = min\{ \Phi(s'): s' \in S_{i} \}$ (Minimization problem)
- $\Phi(s) = max\{ \Phi(s'): s' \in S_{i} \}$ (Maximization problem)

Note that optimal solution may not be unique.

## Approximation

Usually solving general optimization problems is NP-hard, so we sought approximate solutions that are sufficiently precise and feasible to compute. Additionally in real-world instance there is a significant amount of noise, thus the true optimum may not be well defined anymore. 

Let $\Pi$ be an optimization problem.
A _c-approximation algorithm_ $A$ for $\Pi$, with $c \geq 1$, is an algorithm that given $i \in I$, returns a solution $A(i) \in S_{i}$ such that it is at least $c$ times close to the optimal solution (possibly even better):
- $\Phi(A(i)) \leq c\cdot min\{ \Phi(s'): s' \in S_{I} \}$ for minimization problems
- $\Phi(A(i)) \geq \frac{1}{c}\cdot max\{ \Phi(s'): s' \in S_{I} \}$ for maximization problems

$c$ is called the _approximation factor_ of $A$.
$A(i)$ is called the _c-approximate solution_ to $i$.