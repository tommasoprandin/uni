## Basic Concepts

### Information Content

The information content (or self-information) of an outcome $x$ with probability $p(x)$ is:

$$I(x) = -\log_2 p(x)$$

Units: bits (when using $\log_2$), nats (when using $\ln$), or hartleys (when using $\log_{10}$)

**Intuition**: Quantifies the "surprise" or unexpectedness of an outcome. Rare events carry more information than common ones.

### Entropy

A measure of the uncertainty or randomness in a random variable.

For a discrete random variable $X$ with PMF $p(x)$: $$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x) = \mathbb{E}[I(X)]$$

For a continuous random variable $X$ with PDF $f(x)$: $$h(X) = -\int_{-\infty}^{\infty} f(x) \log_2 f(x) , dx$$

Properties:

- $H(X) \geq 0$
- For a discrete random variable with $n$ possible values, $H(X) \leq \log_2(n)$, with equality if and only if all outcomes are equally likely

**Intuition**: Entropy represents the average information content or uncertainty of a random variable. It measures the average number of bits needed to encode symbols from the distribution.

### Joint Entropy

For two discrete random variables $X$ and $Y$ with joint PMF $p(x,y)$:

$$H(X,Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_2 p(x,y)$$

**Intuition**: Measures the combined uncertainty of two random variables.

### Conditional Entropy

For two discrete random variables $X$ and $Y$:

$$H(Y|X) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_2 p(y|x) = H(X,Y) - H(X)$$

**Intuition**: Measures the remaining uncertainty about $Y$ after knowing $X$.

### Chain Rule of Entropy

For random variables $X_1, X_2, \ldots, X_n$:

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, X_2, \ldots, X_{i-1})$$

**Intuition**: The joint entropy can be decomposed as a sum of conditional entropies.

## Mutual Information

### Mutual Information

For two random variables $X$ and $Y$:

$$I(X;Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}$$

Alternative formulations: $$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$$

Properties:

- $I(X;Y) \geq 0$, with equality if and only if $X$ and $Y$ are independent
- $I(X;Y) = I(Y;X)$ (symmetry)

**Intuition**: Measures the amount of information shared between two random variables, or the reduction in uncertainty about one random variable after observing the other.

### Conditional Mutual Information

For three random variables $X$, $Y$, and $Z$:

$$I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log_2 \frac{p(x,y|z)}{p(x|z)p(y|z)}$$

Alternative formulation: $$I(X;Y|Z) = H(X|Z) - H(X|Y,Z)$$

**Intuition**: Measures the mutual information between $X$ and $Y$ conditioned on $Z$.

### Kullback-Leibler (KL) Divergence

A measure of how one probability distribution $P$ differs from a reference probability distribution $Q$:

For discrete distributions: $$D_{KL}(P||Q) = \sum_{x \in \mathcal{X}} P(x) \log_2 \frac{P(x)}{Q(x)}$$

For continuous distributions with PDFs $p$ and $q$: $$D_{KL}(p||q) = \int_{-\infty}^{\infty} p(x) \log_2 \frac{p(x)}{q(x)} , dx$$

Properties:

- $D_{KL}(P||Q) \geq 0$, with equality if and only if $P = Q$ (almost everywhere)
- $D_{KL}$ is not symmetric: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$ in general

**Intuition**: KL divergence measures the information lost when using distribution $Q$ to approximate distribution $P$. It can be interpreted as the "surprise" of seeing data from distribution $P$ when expecting distribution $Q$.

### Jensen-Shannon Divergence

A symmetrized and smoothed version of the KL divergence:

$$JSD(P||Q) = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)$$

where $M = \frac{1}{2}(P + Q)$ is the average of the two distributions.

Properties:

- $JSD(P||Q) \geq 0$, with equality if and only if $P = Q$
- $JSD(P||Q) = JSD(Q||P)$ (symmetry)
- $\sqrt{JSD(P||Q)}$ is a metric

**Intuition**: A more balanced measure of the difference between two probability distributions.

## Information Theory in Communications

### Source Coding

#### Shannon's Source Coding Theorem

For a discrete memoryless source with entropy $H(X)$ bits per symbol:

- The average number of bits per symbol required to encode the source without loss of information is at least $H(X)$
- There exists a coding scheme that uses, on average, at most $H(X) + 1$ bits per symbol

**Intuition**: Entropy establishes the fundamental limit on lossless data compression.

### Channel Coding

#### Channel Capacity

For a discrete memoryless channel, the capacity $C$ is defined as:

$$C = \max_{p(x)} I(X;Y)$$

where the maximum is taken over all possible input distributions $p(x)$.

**Intuition**: The channel capacity represents the maximum rate at which information can be reliably transmitted over a noisy channel.

#### Shannon's Noisy Channel Coding Theorem

- For any rate $R < C$, there exists a coding scheme such that the probability of error can be made arbitrarily small
- For any rate $R > C$, the probability of error is bounded away from zero

**Intuition**: The channel capacity is the maximum rate at which information can be transmitted over a noisy channel with arbitrarily small error probability.

### Example Channels

#### Binary Symmetric Channel (BSC)

A channel that flips each bit with probability $p$.

Capacity: $C = 1 - H(p) = 1 + p \log_2 p + (1-p) \log_2 (1-p)$

**Intuition**: The capacity decreases as the bit-flip probability increases, reaching zero when $p = 0.5$ (completely random channel).

#### Binary Erasure Channel (BEC)

A channel that erases each bit with probability $\epsilon$ (replacing it with a symbol indicating erasure).

Capacity: $C = 1 - \epsilon$

**Intuition**: The capacity is reduced in proportion to the erasure probability.

## Data Compression

### Huffman Coding

A variable-length prefix code that assigns shorter codes to more frequent symbols.

Properties:

- Optimal prefix code for minimizing expected code length
- Average code length is bounded by $H(X) \leq L < H(X) + 1$

**Intuition**: Huffman coding approaches the entropy bound by using shorter codes for more frequent symbols.

### Arithmetic Coding

A method for encoding a sequence of symbols as a single number in the interval $[0, 1)$.

Properties:

- Can achieve compression rates very close to the entropy bound
- Well-suited for adaptive coding where probabilities are updated as encoding proceeds

**Intuition**: Arithmetic coding encodes the entire sequence as a single number, avoiding the "one-symbol-at-a-time" limitation of Huffman coding.

### Lempel-Ziv Algorithms (LZ77, LZ78)

Dictionary-based compression methods that replace repeated patterns with references to previous occurrences.

Properties:

- Universal coding schemes that approach the entropy bound for stationary sources
- Do not require prior knowledge of symbol probabilities

**Intuition**: LZ algorithms build a dictionary of seen patterns on the fly, achieving compression by referencing repeated patterns.

## Information-Theoretic Measures

### Cross-Entropy

For two probability distributions $P$ and $Q$ on the same sample space:

$$H(P, Q) = -\sum_{x \in \mathcal{X}} P(x) \log_2 Q(x)$$

Relation to KL divergence: $H(P, Q) = H(P) + D_{KL}(P||Q)$

**Intuition**: Measures the average number of bits needed to encode data from distribution $P$ using an optimal code for distribution $Q$.

### Perplexity

For a probability distribution $P$:

$$\text{Perplexity}(P) = 2^{H(P)}$$

For cross-entropy: $\text{Perplexity}(P, Q) = 2^{H(P, Q)}$

**Intuition**: Perplexity can be interpreted as the weighted average number of choices the model is uncertain about when predicting the next element. Lower perplexity indicates better predictions.

### Pointwise Mutual Information (PMI)

For two events $x$ and $y$:

$$\text{PMI}(x; y) = \log_2 \frac{p(x, y)}{p(x)p(y)}$$

Relation to mutual information: $I(X; Y) = \mathbb{E}[\text{PMI}(X; Y)]$

**Intuition**: Measures the association between specific outcomes $x$ and $y$. Positive PMI indicates events co-occur more often than expected under independence.

## Rate Distortion Theory

### Rate-Distortion Function

For a source $X$ and a distortion measure $d(x, \hat{x})$:

$$R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

**Intuition**: Represents the minimum rate (bits per symbol) required to represent a source with average distortion not exceeding $D$.

### Shannon's Rate-Distortion Theorem

- For any rate $R > R(D)$, there exists a code with distortion at most $D$
- For any rate $R < R(D)$, all codes have distortion greater than $D$

**Intuition**: The rate-distortion function establishes the fundamental limit on lossy data compression.

## Information Theory in Machine Learning

### Maximum Entropy Principle

Among all probability distributions satisfying a set of constraints, choose the one with maximum entropy.

**Intuition**: The maximum entropy distribution incorporates precisely the information expressed in the constraints, without making additional assumptions.

### Information Bottleneck Method

For random variables $X$ (input), $Y$ (target), and a representation $Z$:

$$\min_{p(z|x)} \beta I(X; Z) - I(Z; Y)$$

where $\beta$ is a Lagrange multiplier controlling the trade-off.

**Intuition**: Creates a representation $Z$ that preserves as much information about the target $Y$ as possible while compressing the input $X$ as much as possible.

### Mutual Information Estimation

Methods for estimating mutual information from samples:

- Binning/histogram methods
- k-nearest neighbor methods
- Kernel density estimation
- Neural estimation methods (MINE, InfoNCE)

**Intuition**: Direct estimation of mutual information from samples is challenging due to the need to estimate the joint distribution.

## Quantum Information Theory

### Quantum Entropy (von Neumann Entropy)

For a quantum state described by density matrix $\rho$:

$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where ${\lambda_i}$ are the eigenvalues of $\rho$.

**Intuition**: Quantum generalization of Shannon entropy, measuring uncertainty in a quantum state.

### Quantum Relative Entropy

For density matrices $\rho$ and $\sigma$:

$$S(\rho||\sigma) = \text{Tr}(\rho(\log_2 \rho - \log_2 \sigma))$$

**Intuition**: Quantum generalization of KL divergence, measuring the distinguishability of quantum states.