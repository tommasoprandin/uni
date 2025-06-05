# Weak Bisimilarity is a Congruence in CCS with Guarded Sum

## Definitions and Preliminaries

**CCS with Guarded Sum Syntax:** $$P, Q ::= \mathbf{0} \mid \alpha.P \mid \sum_{i \in I} \alpha_i.P_i \mid P|Q \mid P \setminus L \mid P[f]$$

where the sum $\sum_{i \in I} \alpha_i.P_i$ is **guarded** (each process is prefixed by an action).

**Weak Transition Relation:**

- $P \Rightarrow Q$ means $P (\xrightarrow{\tau})^* Q$ (zero or more $\tau$ transitions)
- $P \xRightarrow{a} Q$ means $P \Rightarrow \xrightarrow{a} \Rightarrow Q$
- $P \xRightarrow{\tau} Q$ means $P \Rightarrow Q$

**Weak Bisimulation:** A relation $\mathcal{R} \subseteq \text{Proc} \times \text{Proc}$ is a weak bisimulation if whenever $P\ \mathcal{R}\ Q$:

1. If $P \xrightarrow{a} P'$, then $\exists Q'$ such that $Q \xRightarrow{a} Q'$ and $P'\ \mathcal{R}\ Q'$
2. If $Q \xrightarrow{a} Q'$, then $\exists P'$ such that $P \xRightarrow{a} P'$ and $P'\ \mathcal{R}\ Q'$

**Weak Bisimilarity:** $$P \approx Q \iff \exists \mathcal{R} \text{ weak bisimulation such that } P\ \mathcal{R}\ Q$$

## Main Theorem

**Theorem:** Weak bisimilarity $\approx$ is a congruence in CCS with guarded sum.

**Proof Strategy:** We need to show that $\approx$ is preserved by all CCS operators. Due to the guarded sum restriction, we can prove congruence for each operator separately.

---

## Proof for Each Operator

### 1. Action Prefix

**Claim:** If $P \approx Q$, then $\alpha.P \approx \alpha.Q$ for any action $\alpha$.

**Proof:** Let $R = {(\alpha.P, \alpha.Q) \mid P \approx Q} \cup {(P', Q') \mid P' \approx Q'}$.

We show $R$ is a weak bisimulation:

- If $\alpha.P \xrightarrow{\alpha} P'$, then $P' = P$
- Since $P \approx Q$, we have $\alpha.Q \xrightarrow{\alpha} Q$ and $P \approx Q$
- Thus $PR'Q$ where $R'$ witnesses $P \approx Q$

The symmetric case is identical. Therefore $\alpha.P \approx \alpha.Q$.

### 2. Parallel Composition

**Claim:** If $P_1 \approx Q_1$ and $P_2 \approx Q_2$, then $P_1|P_2 \approx Q_1|Q_2$.

**Proof:** Let $R = {(P_1'|P_2', Q_1'|Q_2') \mid P_1' \approx Q_1' \text{ and } P_2' \approx Q_2'}$.

We show $R$ is a weak bisimulation. Consider $(P_1|P_2, Q_1|Q_2) \in R$.

**Case 1:** $P_1|P_2 \xrightarrow{a} P_1'|P_2$ where $P_1 \xrightarrow{a} P_1'$

- Since $P_1 \approx Q_1$, there exists $Q_1'$ such that $Q_1 \xRightarrow{a} Q_1'$ and $P_1' \approx Q_1'$
- Then $Q_1|Q_2 \xRightarrow{a} Q_1'|Q_2$
- Since $P_1' \approx Q_1'$ and $P_2 \approx Q_2$, we have $(P_1'|P_2, Q_1'|Q_2) \in R$

**Case 2:** $P_1|P_2 \xrightarrow{a} P_1|P_2'$ where $P_2 \xrightarrow{a} P_2'$

- Similar to Case 1, using $P_2 \approx Q_2$

**Case 3:** $P_1|P_2 \xrightarrow{\tau} P_1'|P_2'$ where $P_1 \xrightarrow{a} P_1'$, $P_2 \xrightarrow{\bar{a}} P_2'$

- Since $P_1 \approx Q_1$, $\exists Q_1'$ such that $Q_1 \xRightarrow{a} Q_1'$ and $P_1' \approx Q_1'$
- Since $P_2 \approx Q_2$, $\exists Q_2'$ such that $Q_2 \xRightarrow{\bar{a}} Q_2'$ and $P_2' \approx Q_2'$
- Then $Q_1|Q_2 \xRightarrow{\tau} Q_1'|Q_2'$ (via synchronization)
- We have $(P_1'|P_2', Q_1'|Q_2') \in R$

The symmetric cases follow similarly.

### 3. Restriction

**Claim:** If $P \approx Q$, then $P \setminus L \approx Q \setminus L$ for any $L$.

**Proof:** Let $R = {(P' \setminus L, Q' \setminus L) \mid P' \approx Q'}$.

If $P \setminus L \xrightarrow{a} P' \setminus L$, then:

- $P \xrightarrow{a} P'$ and $a, \bar{a} \notin L$
- Since $P \approx Q$, $\exists Q'$ such that $Q \xRightarrow{a} Q'$ and $P' \approx Q'$
- Since $a, \bar{a} \notin L$, we have $Q \setminus L \xRightarrow{a} Q' \setminus L$
- Thus $(P' \setminus L, Q' \setminus L) \in R$

### 4. Relabeling

**Claim:** If $P \approx Q$, then $P[f] \approx Q[f]$ for any relabeling function $f$.

**Proof:** Let $R = {(P'[f], Q'[f]) \mid P' \approx Q'}$.

If $P[f] \xrightarrow{a} P'[f]$, then:

- $P \xrightarrow{b} P'$ where $f(b) = a$
- Since $P \approx Q$, $\exists Q'$ such that $Q \xRightarrow{b} Q'$ and $P' \approx Q'$
- Then $Q[f] \xRightarrow{a} Q'[f]$ (since $f(b) = a$)
- Thus $(P'[f], Q'[f]) \in R$

### 5. Guarded Sum (Key Case)

**Claim:** If $P_i \approx Q_i$ for all $i \in I$, then $\sum_{i \in I} \alpha_i.P_i \approx \sum_{i \in I} \alpha_i.Q_i$.

**Proof:** This is where the **guarded** property is crucial. Let:


If $\sum_{i \in I} \alpha_i.P_i \xrightarrow{\alpha_j} P_j$ for some $j \in I$, then:

- $\sum_{i \in I} \alpha_i.Q_i \xrightarrow{\alpha_j} Q_j$
- Since $P_j \approx Q_j$, we have $(P_j, Q_j) \in R$

**Crucial observation:** Since the sum is guarded, both processes can only perform the initial actions $\alpha_i$. There are no $\tau$-transitions from the sum itself, which prevents the problematic cases that arise with unguarded sums.

---

## Why Guarded Sum is Essential

Without the guarded restriction, weak bisimilarity is **not** a congruence. Here's the counterexample from your notes:

**Counterexample for Unguarded Sum:**

- $P = \tau.0 \approx 0 = Q$ (they are weakly bisimilar)
- $R = a.0$
- Consider $P + R = \tau.0 + a.0$ and $Q + R = 0 + a.0 = a.0$

From $P + R$, we can have:

- $P + R \xrightarrow{\tau} 0 + a.0 = a.0$
- $P + R \xrightarrow{a} 0$

From $Q + R = a.0$, we can only have:

- $Q + R \xrightarrow{a} 0$

The issue is that $P + R$ can reach a state (after $\tau$) that is not weakly bisimilar to any state reachable from $Q + R$.

**With guarded sum:** This problem cannot occur because we cannot have $\tau.0 + a.0$ as a guarded sum - every summand must be prefixed.

---

## Conclusion

**Theorem:** Weak bisimilarity $\approx$ is a congruence in CCS with guarded sum.

The proof works by showing that $\approx$ is preserved by all CCS operators: action prefix, parallel composition, restriction, relabeling, and crucially, guarded sum. The guarded restriction is essential to prevent the counterexamples that make weak bisimilarity fail to be a congruence in full CCS.

Therefore, in CCS with guarded sum: $$P \approx Q \implies C[P] \approx C[Q] \text{ for all contexts } C[\cdot]$$