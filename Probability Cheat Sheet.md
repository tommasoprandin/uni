## Basic Concepts

### Sample Space

The set of all possible outcomes of a random experiment.

- Denoted by $\Omega$
- Example: For a coin flip, $\Omega = {H, T}$

### Event

A subset of the sample space.

- Denoted by capital letters like $A$, $B$
- Example: The event of getting heads is $A = {H}$

### Probability Axioms

For a probability function $P$:

1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$
3. If $A_1, A_2, \ldots$ are mutually exclusive events, then $P(A_1 \cup A_2 \cup \ldots) = P(A_1) + P(A_2) + \ldots$

## Discrete Probability

### Probability Mass Function (PMF)

For a discrete random variable $X$, the PMF $p_X(x)$ gives the probability that $X$ equals $x$.

$$p_X(x) = P(X = x)$$

Properties:

- $p_X(x) \geq 0$ for all $x$
- $\sum_x p_X(x) = 1$

**Intuition**: The PMF gives the probability of each possible value the random variable can take.

## Continuous Probability

### Probability Density Function (PDF)

For a continuous random variable $X$, the PDF $f_X(x)$ defines probabilities via integrals.

$$P(a \leq X \leq b) = \int_a^b f_X(x)  dx$$

Properties:

- $f_X(x) \geq 0$ for all $x$
- $\int_{-\infty}^{\infty} f_X(x)  dx = 1$

**Intuition**: The PDF gives the relative likelihood of the random variable taking on a value near $x$. The probability is given by the area under the curve.

### Cumulative Distribution Function (CDF)

For any random variable $X$, the CDF $F_X(x)$ gives the probability that $X$ is less than or equal to $x$.

$$F_X(x) = P(X \leq x)$$

For discrete random variables: $$F_X(x) = \sum_{t \leq x} p_X(t)$$

For continuous random variables: $$F_X(x) = \int_{-\infty}^{x} f_X(t)  dt$$

Properties:

- $F_X(x)$ is non-decreasing
- $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$

**Intuition**: The CDF gives the cumulative probability up to a certain value.

## Expected Value and Variance

### Expectation

The expected value (or mean) of a random variable $X$ is denoted by $\mathbb{E}[X]$ or $\mu_X$.

For discrete random variables: $$\mathbb{E}[X] = \sum_x x \cdot p_X(x)$$

For continuous random variables: $$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x)  dx$$

Properties:

- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$ for constants $a$ and $b$
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ for any random variables $X$ and $Y$

**Intuition**: The expected value is the long-run average value of the random variable over many independent repetitions.

### Variance

The variance of a random variable $X$ is denoted by $\text{Var}(X)$ or $\sigma_X^2$.

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X - \mu_{X}] =  \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

For discrete random variables: $$\text{Var}(X) = \sum_x (x - \mathbb{E}[X])^2 \cdot p_X(x)$$

For continuous random variables: $$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mathbb{E}[X])^2 \cdot f_X(x)  dx$$

Properties:

- $\text{Var}(aX + b) = a^2\text{Var}(X)$ for constants $a$ and $b$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ if $X$ and $Y$ are independent

**Intuition**: The variance measures the spread or dispersion of the random variable from its expected value.

### Standard Deviation

The standard deviation of a random variable $X$ is denoted by $\text{SD}(X)$ or $\sigma_X$. $$\text{SD}(X) = \sqrt{\text{Var}(X)}$$

**Intuition**: The standard deviation provides a measure of spread in the same units as the random variable.

## Joint Probability

### Joint Probability Mass Function (Joint PMF)

For discrete random variables $X$ and $Y$, the joint PMF $p_{X,Y}(x,y)$ gives the probability that $X = x$ and $Y = y$.

$$p_{X,Y}(x,y) = P(X = x, Y = y)$$

Properties:

- $p_{X,Y}(x,y) \geq 0$ for all $x,y$
- $\sum_x \sum_y p_{X,Y}(x,y) = 1$

**Intuition**: The joint PMF describes the probability distribution of two random variables considered together.

### Joint Probability Density Function (Joint PDF)

For continuous random variables $X$ and $Y$, the joint PDF $f_{X,Y}(x,y)$ defines probabilities via double integrals.

$$P(X \in A, Y \in B) = \int_B \int_A f_{X,Y}(x,y)  dx  dy$$

Properties:

- $f_{X,Y}(x,y) \geq 0$ for all $x,y$
- $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x,y)  dx  dy = 1$

**Intuition**: The joint PDF gives the relative likelihood of the random variables taking on values near $(x,y)$.

### Marginal Distributions

The marginal PMF or PDF of $X$ can be derived from the joint distribution:

For discrete random variables: $$p_X(x) = \sum_y p_{X,Y}(x,y)$$

For continuous random variables: $$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)  dy$$

**Intuition**: Marginal distributions focus on one random variable, "summing out" the other variables.

## Conditional Probability

### Conditional Probability

The conditional probability of event $A$ given event $B$ is denoted by $P(A|B)$.

$$P(A|B) = \frac{P(A , B)}{P(B)}, \text{ where } P(B) > 0$$

**Intuition**: The conditional probability measures the probability of an event given that another event has occurred.

### Conditional PMF and PDF

For discrete random variables: $$p_{X|Y}(x|y) = \frac{p_{X,Y}(x,y)}{p_Y(y)}, \text{ where } p_Y(y) > 0$$

For continuous random variables: $$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}, \text{ where } f_Y(y) > 0$$

**Intuition**: The conditional distribution describes the probability distribution of one random variable given the value of another random variable.

### Independence

Random variables $X$ and $Y$ are independent if and only if:

For discrete random variables: $$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y) \text{ for all } x,y$$

For continuous random variables: $$f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y) \text{ for all } x,y$$

Equivalently, events $A$ and $B$ are independent if and only if: $$P(A , B) = P(A) \cdot P(B)$$

**Intuition**: Independence means that knowing the value of one random variable does not affect the probability distribution of the other.

## Bayes' Rule

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

Expanded form: $$P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A^c)P(A^c)}$$

Where $A^c$ is the complement of event $A$.

**Intuition**: Bayes' rule allows us to update our beliefs about an event $A$ after observing evidence $B$.

## Common Probability Distributions

### Bernoulli Distribution

A discrete distribution for a random variable that takes the value 1 with probability $p$ and the value 0 with probability $1-p$.

PMF: $$p_X(x) = \begin{cases} p, & \text{if } x = 1 \ 1-p, & \text{if } x = 0 \end{cases}$$

Mean: $E[X] = p$ Variance: $\text{Var}(X) = p(1-p)$

**Intuition**: Models a single trial with two possible outcomes (success/failure).

### Binomial Distribution

A discrete distribution for the number of successes in $n$ independent Bernoulli trials, each with probability of success $p$.

PMF: $$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \text{ for } k = 0, 1, \ldots, n$$

Where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

Mean: $E[X] = np$ Variance: $\text{Var}(X) = np(1-p)$

**Intuition**: Models the total number of successes in a fixed number of independent trials.

### Normal (Gaussian) Distribution

A continuous distribution with PDF:

$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation.

Standard Normal: When $\mu = 0$ and $\sigma = 1$, the distribution is called the standard normal distribution, often denoted by $Z$.

Mean: $E[X] = \mu$ Variance: $\text{Var}(X) = \sigma^2$

**Intuition**: The normal distribution naturally arises in many settings and is often used to model measurement errors and natural phenomena. The Central Limit Theorem explains why many random variables tend to be normally distributed.

### Poisson Distribution

A discrete distribution for the number of events occurring in a fixed interval of time or space, assuming these events occur with a known average rate and independently of each other.

PMF: $$p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!}, \text{ for } k = 0, 1, 2, \ldots$$

Where $\lambda > 0$ is the average number of events in the interval.

Mean: $E[X] = \lambda$ Variance: $\text{Var}(X) = \lambda$

**Intuition**: Models the number of rare events occurring in a fixed time or space interval.

### Exponential Distribution

A continuous distribution that models the time between events in a Poisson process.

PDF: $$f_X(x) = \lambda e^{-\lambda x}, \text{ for } x \geq 0$$

Where $\lambda > 0$ is the rate parameter.

Mean: $E[X] = \frac{1}{\lambda}$ Variance: $\text{Var}(X) = \frac{1}{\lambda^2}$

**Intuition**: Models waiting times between consecutive events when events occur continuously and independently at a constant average rate.

### Uniform Distribution

A continuous distribution where all intervals of the same length within the distribution's support are equally probable.

PDF: $$f_X(x) = \begin{cases} \frac{1}{b-a}, & \text{if } a \leq x \leq b \ 0, & \text{otherwise} \end{cases}$$

Mean: $E[X] = \frac{a+b}{2}$ Variance: $\text{Var}(X) = \frac{(b-a)^2}{12}$

**Intuition**: Models a random variable that is equally likely to take any value within an interval.

## Parameter Estimation

### Maximum Likelihood Estimation (MLE)

A method for estimating parameters of a statistical model, given observations, by finding the parameter values that maximize the likelihood function.

For a parameter $\theta$ and observations $x_1, x_2, \ldots, x_n$, the likelihood function is: $$L(\theta) = \prod_{i=1}^{n} f(x_i|\theta)$$

Where $f(x_i|\theta)$ is the PDF or PMF of $x_i$ given $\theta$.

The MLE of $\theta$ is: $$\hat{\theta}_{MLE} = \underset{\theta}{\arg\max} , L(\theta)$$

**Intuition**: MLE finds the parameter values that make the observed data most probable.

### Maximum Log-Likelihood

Since the logarithm is a monotonically increasing function, maximizing the log-likelihood gives the same result as maximizing the likelihood, but is often computationally easier.

The log-likelihood function is: $$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i|\theta)$$

The MLE of $\theta$ using log-likelihood is: $$\hat{\theta}_{MLE} = \underset{\theta}{\arg\max} , \ell(\theta)$$

**Intuition**: Taking the logarithm converts products to sums, making the optimization problem more tractable while yielding the same result.

## Useful Inequalities

### Markov's Inequality

For a non-negative random variable $X$ and $a > 0$: $$P(X \geq a) \leq \frac{E[X]}{a}$$

**Intuition**: Provides an upper bound on the probability that a non-negative random variable exceeds a certain value.

### Chebyshev's Inequality

^68c290

For a random variable $X$ with finite mean $\mu$ and variance $\sigma^2$, and for any $k > 0$: $$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

**Intuition**: Provides an upper bound on the probability that a random variable deviates from its mean by more than a certain amount.