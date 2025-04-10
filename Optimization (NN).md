Deep learning algorithms involve [[Optimization Problem|optimization]] in many contexts. For example, performing inference in models such as PCA involves solving an optimization
problem. We often use analytical optimization to write proofs or design algorithms.
Unfortunately, in the general case, optimization problems are NP-hard.

Of all the many optimization problems involved in deep learning, the most diﬃcult
is **neural network training**.
Because this problem is so important and so expensive, a specialized set of optimization techniques have been developed for solving it.

Generally the function to optimize that we care about is the **true error**:
$$
J^{\star}(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{x}, y) \sim p_{data}}L(f(\boldsymbol{x}; \boldsymbol{\theta}), y)
$$
which is the expected loss for the computed output $y$ with input $\boldsymbol{x}$ and weights $\boldsymbol{\theta}$ for the hypothesis $f$, for example drawn from the true data distribution $p_{data}$.
Clearly we have no way of knowing $p_{data}$, hence $J^{\star }$ is not computable.

We have to optimize this indirectly through the training set with the **empirical error**:
$$
J(\boldsymbol{\theta}) = \mathbb{E}_{(\boldsymbol{x}, y) \sim p_{training}} L(f(\boldsymbol{x}; \boldsymbol{\theta}), y) = \underbrace{ \frac{1}{n}\sum_{i=1}^{n} L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), y^{(i)}) }_{ \text{average of the loss over the training set} }
$$

Unfortunately this approach is very prone to _overfitting_, since it tends to memorize training data, thus making the model follow examples that may not be representative of the full distribution and focusing only on them; additionally many loss functions are not differentiable, which makes **gradient descent not possible**.

### Surrogate Loss

The loss function we care about cannot be generally optimized efficiently, thus we can try to find a proxy that is easier to optimize, but still represents it well. This new function is called _surrogate loss_.

A classical example is to use negative log likelihood instead of 0-1 loss (number of misclassified examples), which is not differentiable.

Another observation we make is that we don't actually want to find a local minimum since the error on training may not be representative of the true error, hence optimizing on it to reach a perfect local minima makes little sense.

### Batch/Minibatch Algorithms

As seen in the formulation above, using gradient descent the update is based on the expected loss **for all the training data**.

For instance in case of negative log likelihood loss function:
$$
J(\theta) = -\mathbb{E}_{(x, y) \sim p_{training}}\log p_{model}(y \mid x)
$$
where $p_{model}$ is the output probability distribution learned by the network.

But this is _very expensive_ computationally, and it is even really helpful in practice for these reasons:
1. The _standard error_ for the estimation of the mean scales less than linearly: $\frac{\sigma}{\sqrt{ n }}$
2. Most optimization algorithms _converge faster_ (in time) using less accurate gradient estimation
3. Small batch size have a _regularization effect_ (it is similar to add **noise** in the gradient)
4. It cannot be applied for _online algorithms_ where one example _has_ to be considered at a time (e.g. models working on time-series live data)

So we have three options:

| | (Full batch) gradient descent | (Online) stochastic gradient descent | Mini-batch gradient descent |
|-|-|-|-|
|Method| Computes the canonically correct gradient of the loss on the _whole_ training set and then updates the weights | Computes the gradient on a _single_ example of the training set and then updates the weights| Computes the gradient on a _subset_ of examples of the training set and then updates the weights |
|Advantages| Loss depends on _all_ the examples, it it the _correct_ gradient | It computes one weight update per example ($n$ weights updates per epoch) | Good tradeoff of correctness and updates number. Scales really well on distributed computing platforms and parallel devices (GPUs) |
|Disadvantages| Only one weights update per epoch | Gradient is less accurate | None (if proper batch selection) |

### Newton's Method

Newton's method is a common used to find the _roots_ of a _differentiable_ function $f$ (i.e. all the points $x_{r}$  $\{  x_{r} : f(x_{r}) = 0\}$), if there is no closed form formula for them, or if it is infeasible to compute it.

The idea is to start from an initial guess point $x_{0}$, and iteratively linerize the function, find the $x'$ where the linearize function intersects the $x$ axis, and repeat the process in the new $x'$.

![[dl-newton-method.png]]

Let's consider an arbitrary point $x_{n}$. The first-order approximation of $f$ in $x_{n}$ is given by the Taylor's series:
$$
f(x) \simeq f(x_{n}) + f'(x_{n})(x-x_{n})
$$

Now let $x_{n+1}$ be the zero of the linearized function, which will be the new guess. We can obtain it by solving the equation:
$$
\begin{gather}
f(x_{n}) + f'(x_{n})(x-x_{n})  =0 \\
f'(x_{n})(x-x_{n})   = -f(x_{n}) \\
x - x_{n}   = -\frac{f(x_{n})}{f'(x_{n})} \\

\end{gather}
$$
which results in the update step equation:
$$
x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
$$

Intuitively this means start from an initial guess, then move in the direction where the function tend to go to zero. For instance if both $f$ and $f'$ are positive, we want to move to the left, proportionally to how big the function is (how far from zero), and inversely proportional to its steepness.

This method is guaranteed to _converge_ in quadratic time only if the following conditions are satisfied:
1. The first derivative is not zero around the current guess, or it would shoot very far away
2. The second derivative is continuous around the current guess, so it doesn't have sudden jumps in curvature
3. The curvature is not excessive (i.e. smaller than the steepness of the function).

#### Newton's Method in Optimization

It is clearly useful to try and use Newton's method to solve optimization problems, since, by finding the roots of the derivative, we can find _local minima_ (_or maxima_). This is actually not always the case as explained in [[#Saddle Points]], but still a good approach.

The first order Taylor expansion of the derivative of $f$ is clearly the second order expansion of $f$ itself:
$$
f(x) \simeq f(x_{n}) + f'(x_{n})(x-x_{n}) + \frac{1}{2}f''(x_{n})(x-x_{n})^{2}
$$

By solving for the zeros of this quadratic equation (that is the parabola that fits $f$ around $x_{n}$), we obtain a very simple way to obtain the zeros of $f'$ with the same iterative steps for the previous case:
$$
x_{n+1} = x_{n} - \frac{f'(x_{n})}{f''(x_{n})}
$$

The geometric interpretation of Newton's method is that at each iteration, it amounts to the fitting of a parabola to the graph of $f(x)$ at the trial value $x_{n}$, having the same slope and curvature as the graph at that point, and then proceeding to the maximum or minimum of that parabola (in higher dimensions, this may also be a saddle point). Note that if $f$ happens to _be_ a quadratic function, then the exact extremum is found in one step.

![[dl-newton-optimization.svg|center|200]]

The image above shows a comparison between gradient descent in green (first order method), and Newton's in red (second order). The second order method approaches the minima much faster because it accounts for the curvature (because it uses $f''$) of the function.

This can generalized for multivariate functions by replacing $f'$ with the gradient $\nabla f$, and $\frac{1}{f''}$ with the _inverse_ of the Hessian matrix $\boldsymbol{H}^{-1}$:

$$
\boldsymbol{x}_{n+1} = \boldsymbol{x}_{n} - \nabla f(\boldsymbol{x}_{n})\cdot \boldsymbol{H}^{-1}(\boldsymbol{x}_{n})
$$

### Local Quadratic Approximation

A local quadratic approximation to the error function can provide useful insights into the optimization problem and to which techniques to use for solving it.

The second-order Taylor expansion for a generic error function is:
$$
J(\boldsymbol{w}) \simeq J(x_{0}) + \nabla_{\boldsymbol{w}}J(\boldsymbol{w}_{0})(\boldsymbol{w}-\boldsymbol{w}_{0}) + \frac{1}{2}(\boldsymbol{w} - \boldsymbol{w}_{0})^{T}\boldsymbol{H}(\boldsymbol{w}_{0})(\boldsymbol{w} - \boldsymbol{w}_{0})
$$

Where $\nabla_{\boldsymbol{w}}J(\boldsymbol{w}_{0})$ and $\boldsymbol{H}(\boldsymbol{w}_{0})$ are respectively the gradient and the Hessian of $J$ evaluated in $\boldsymbol{w}_{0}$.

By analyzing $\boldsymbol{H}$ (which contains information about the curvature of the error landscape), we can infer properties about the optimization problem.

### Ill-conditioning of the Hessian

Even when optimizing convex functions there are complicated challenges that can arise. Of these the most prominent is _ill-conditioning_ of the Hessian matrix $\boldsymbol{H}$.

We say that the Hessian is ill-conditioned when the Hessian matrix values have a large variance, intuitively this means that the curvature of the function is highly different across the directions (i.e. it is not smooth).

In this case first order methods (like [[#Stochastic Gradient Descent | SGD]]) may get stuck, since the gradient does not carry enough information about the curvature of the loss function, and even small steps in the gradient direction can increase the cost function.

By taking the second derivative into account, this can be solved. Even so this is very rarely used since computing the Hessian matrix is unfeasible even for small networks.

### Local Minima

Ideally we would like to work with _convex_ functions to optimize, since all of their **local minima are global minima**, so any critical point (i.e. $f'(x) = 0$) is a _good_ solution.

Unfortunately the error function of a neural network is extremely complex and can have in general an infinite number of local minima which are not optimal (thus it's _non-convex_).

An example of a very loss landscape follows (note that it is a low-dimensional visualization of a high-dimesional space):

![[dl-loss-landscape.png]]

The problem is that we don't want to find local minima which have high cost, and this was believed to be a huge problem for NN.

Actually we have seen that this is not a problem with modern large NN, since their loss landscape is so varied that there are a lot of local minima with low cost, even if not strictly optimal, which still works very well in practice.

As seen in the graph before, we can see that the error rate keeps decreasing even if the gradient norm keeps increasing. We normally expect the gradient norm to become smaller as we approach a critical point.

So a local minima is never reached even if the training process is reasonably successful.

![[dl-gradient-norm.png]]

### Saddle Points

A critical point ($f'(x) = 0$) can not only represent a local minima (maxima), but also a _saddle point_, that is a point where there is a change in the curvature.

In 1 dimension this means that the function before the saddle point is convex and after it is concave.

In many-dimensional spaces this means that the curvature is different along the dimensions, for instance along $x$ it is convex while along $y$ it is concave.

We have briefly shown [[Deep Neural Networks Training#The Hessian Matrix and Second-Order Optimization|here]] that on saddle points the Hessian matrix contains mixed _eigenvalues_ (both positive and negative).

By a simple statistic consideration in highly-dimensional spaces (like the cost function of NN), it is **exponentially unlikely** that all the eigenvalues have the same sign, so there will be extremely likely to end up in saddle points.

Eigenvalues of the Hessian are more likely to be positive on low-cost regions, thus local minima are likely to have low cost.

![[dl-saddle-points.png]]

#### Implications

For first-order optimization algorithms (gradient descent), the gradient will be small near a saddle point, if we introduce a degree of _stochasticity_ (e.g. with [[#Batch/Minibatch Algorithms]]) we are likely to escape them quickly.

Instead with second-order methods (Newton's), they will "jump" from point with zero gradient to another. So in high-dimension spaces it will very likely jump from a saddle point to another one.

This is one reason why **first-order methods are preferred in training deep NNs**.

Also flat regions where both gradient and Hessian are zero are problematic to escape (again need some randomness to escape).

![[dl-flat-regions.png]]


### Cliffs and Exploding Gradient

Deep NN tend to have extremely steep regions where the variation in the gradient is very large and the update steps can jump far away.

We can try to reduce this phenomenon with **gradient clipping** that limits the step size to a fixed bound (hyperparameter).

![[dl-cliff.png]]

#### Exploding/Vanishing Gradient

This is a very problematic behaviour, especially for RNN where the network is "unrolled", and thus much deeper and with shared weights for the recursive parts.

Suppose we repeatedly multiply the input by a weight matrix $\boldsymbol{W} = \boldsymbol{V}diag(\boldsymbol{\lambda})\boldsymbol{V}^{-1}$. After $t$ steps it is equivalent to the multiplication by $\boldsymbol{W}^{t} = \boldsymbol{V}diag(\boldsymbol{\lambda})^{t}\boldsymbol{V}^{-1}$ (supposing linear activation), with $\boldsymbol{\lambda}$ being the _eigenvalues_ and $\boldsymbol{V}$ being the _eigenvectors_ matrix (eigendecomposition).

If the _eigenvalues_ are not close to 1 they will **vanish** over the steps (if < 1), or **explode** (if > 1).

The same problem can happen with activation functions that saturates in some range (e.g. sigmoid), since in those range the gradient is very close to zero (the activation is almost flat).

Feedforward networks use different matrices for each layer, and by using non-saturating functions (e.g. ReLU) they mainly avoid the problem.

Also batch and layer normalization helps with this problem.

TODO: add link.

### Inexact Gradient

Optimization algorithms assume to have access to the true gradient and Hessian. In DL we only have access to noisy/biased estimates coming from the examples.

This means that the local structure is not representative of the global structure.

This is another reason why we don't particularly care about exact optimization and correctness, but reaching a "good enough" point is all we need.

To do so it is important to find a reasonably well-behaved region that can be descended to a low-cost solution, and to initialize our search within that well behaved region.

### Basic Optimization Algorithm

#### Stochastic Gradient Descent

It is a minibatch optimization algorithm very similar to batch gradient descent, but with the learning rate $\epsilon$ decreasing with time (steps). 

Since the random sampling introduces a degree of noise that does not decay when we arrive at a minimum, we want to gradually decrease the step size because we expect over time to reach a good solution, and we don't want noise to move away from it.

We can introduce a simple linear learning rate decay update:
$$
\epsilon_{k} = (1-\alpha)\epsilon_{0} + \alpha\epsilon_{\tau}
$$
$$
\boldsymbol{\theta} \gets \boldsymbol{\theta} - \epsilon_{k}\boldsymbol{\hat{g}}
$$
with $\alpha = \frac{k}{\tau}$, after the $\tau$ iteration it is common to leave $\epsilon$ constant.

```pseudo
\begin{algorithm}
\caption{Stochastic Gradient Descent update}
\begin{algorithmic}
\Require Learning rate schedule $\epsilon_1, \dots$
\Require Initial parameters $\boldsymbol{\theta}$
\State $k \gets 1$
\While{stopping criterion not met}
	\State Sample a minibatch of $m$ iid examples $\{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(m)}\}$ with labels $\{\boldsymbol{y}^{(1)}, \dots, \boldsymbol{y}^{(m)}\}$
	\State Compute gradient estimate $\boldsymbol{\hat{g}} \gets \frac{1}{m}\nabla_{\boldsymbol{\theta}}\sum_{i}L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
	\State Apply update: $\boldsymbol{\theta} \gets \boldsymbol{\theta} - \epsilon_{k}\boldsymbol{\hat{g}}$
	\State $k \gets k + 1$
\EndWhile
\end{algorithmic}
\end{algorithm}
```

One of the advantages of SGD is that its inherently added noise allows it to "escape" from wrong areas in the optimization space (e.g. flat regions or high-cost regions).

### Skip Connections

Many NN architectures build a main chain, and then add extra architectural features to it.

One example is **skip connections** where "blocks" are connected by both the next one in the chain, but also to a number of successive blocks directly.

This allows the gradient to "flow" much more easily from the output to the blocks (and layers) near the input.

This architectural change tend to change the loss landscape dramatically, making it much smoother and convex, thus simpler to optimize.

![[dl-skip-conn.png]]
![[dl-skip-conn-ls.png]]

### Momentum

SGD can be slow, since the update is by definition _stochastic_ it could get lost moving erratically around the "true" gradient direction.
This is more evident in situations where:
- The gradient is small, and thus the effect of the noise induced by the sampling is more evident.
- The curvature is high, and thus the direction to take is clearly defined.
- The gradient is noisy, and the randomness of SGD can accentuate this.

The idea is to introduce a _momentum_ term that forces the updates to tend towards the same direction, like a car with a certain velocity will have a momentum that will try to keep the car going in the same direction.

In practice we implement this by accumulating an _exponential moving average_ of past gradients, and keep moving in that direction. In this way we will keep updates "consistent" over time, and limit the erratic progress of SGD.

Formally, we introduce a _velocity_ parameter $\boldsymbol{\nu}$, and $\alpha \in [0; 1[$ that determines how fast old values decay in importance on the average:
$$
\begin{gather}
\boldsymbol{\nu} \gets \alpha \boldsymbol{\nu} - \epsilon \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) \\
\boldsymbol{\theta} \gets \boldsymbol{\theta} + \boldsymbol{\nu}
\end{gather}
$$

The **Nestorov's momentum** inverts the operations, so it first applies the past velocity to $\boldsymbol{\theta}$, and then updates $\boldsymbol{\nu}$.

In this way the gradient step depends on how _aligned_ the sequence of last gradients are.

The gradient in one direction is increased if all these gradients are aligned over multiple iterations, but decreased if the gradient direction repeatedly changes as the terms in the sum cancel out.

The effect is a **smoother trajectory** and reduced oscillatory behaviour of the updates (since we are actually filtering the updates with an exponential moving average).

![[dl-momentum.png]]

### Parameter Initialization

DL training algorithms are iterative and depends heavily on initialiazation. Depending on where we start gradient descent may fail to converge due to either numerical instability, or due to being in a bad (high-cost) area, from which it is hard to escape.

Additionally starting point with similar training error, may have different generalization error on the test set.

Unfortunately there is no formally correct technique for selecting initial parameters, but in practice there are some _strategies and heuristics_ that performs well.

#### Symmetries

Surely we have to **avoid symmetries** because, as also reported in the example below of backpropagation on a symmetric network, if all the weights in a layer are initialized with the same value, **each neuron will compute the same function**.

Thus the **gradient will be the same**, and the updates will be identical. In other words **each layer will be equivalent to a layer with a single neuron**.

![[dl-init-symmetry.png]]

#### Weights Initialization

The most natural way to initialize the weights in the network is to randomly select them from a certain distribution, we almost always select either **Gaussian** or _uniform_ distribution.

The choice of distribution doesn't really matter much, but its _scale_ has a very large effect on both the outcome of the optimization phase and to the ability of the network to generalize.

In particular **large weights** have a stronger _symmetry breaking effect_, but can incur in [[#Cliffs and Exploding Gradient | exploding gradients]] or saturation.

Using gradient descent and early stopping has similar effect to imposing a Gaussian prior around the initialization weights, because it forces the weights to be small since SGD will reduce the step size over time, and early stopping will limit the amount of steps.
Thus it makes sense to initialize the weights $\boldsymbol{\theta}_{0}$ to be near zero.

One common euristic used to determine the initial weights for a **fully-connected NN with $m$ inputs and $n$ outputs**:
$$
W_{i, j} \sim U\left( -\sqrt{ \frac{6}{m + n} }, \sqrt{ \frac{6}{m+n} } \right)
$$

The **biases** are initialized to pre-defined _constants_ (usually $\in [0; 0.1]$ for ReLU to avoid saturation).

The following table reports example of distribution for initializations according to the type of activation function:

![[dl-init-table.png]]

##### Pre-Training

We could also adopt ML techniques to obtain initial parameters to feed the DNN.

For instance we could use, to train an unsupervised model on the training data and use it as initialization.

We could also initialize the weights with a model trained on a similar task.

These techniques are reported in more detail [[Regularization#Transfer Learning (Pre-training)| here]].

### Adaptive Learning Rate

Gradient descent with a fixed step size has an _undesirable property_: it makes **large adjustments** when it encounters large gradients, when we may want to be more cautious to avoid "overdoing" it; and **small adjustments** with small gradients, where we may want to try and explore the area to escape an (almost) flat surface.

The problem is that when the gradient of the loss surface is irregularly curved on its directions (ill-conditioned), it is difficult to choose a _fixed_ learning rate that is both _stable_ and that makes _quick progress_ in all directions.

The idea is to use **individual learning rates for each parameter** (direction)

#### AdaGrad

As the name suggests Ada(ptive)Grad(ients) _dynamically_ scales the learning rates of individual parameters by accumulating all the squared values of the gradient.

It is a variant of the [[#Stochastic Gradient Descent]] algorithm with adaptive learning rates for each parameter.

Let $\boldsymbol{r}$ be the "accumulation" variable initialized to 0, $\epsilon$ the "global" learning rate, and let $\boldsymbol{g}$ be the gradient computed for the mini-batch. Then the update of the accumulation $\boldsymbol{r}'$ is:
$$
\boldsymbol{r}' = \boldsymbol{r} + \boldsymbol{g} \odot \boldsymbol{g} \text{ (element wise square)}
$$

The update of the step is:
$$
\Delta \boldsymbol{\theta} = - \frac{\epsilon}{\delta + \sqrt{ \boldsymbol{r} }}\odot \boldsymbol{g}
$$
Note that $\delta$ is a small constant to avoid division by zero

```pseudo
\begin{algorithm}
\caption{The AdaGrad algorithm}
\begin{algorithmic}
\Require Global learning rate $\epsilon$
\Require Initial parameters $\boldsymbol{\theta}$
\Require Small constant $\delta$ for numerical stability
\State Initialize the accumulator $\boldsymbol{r} \gets 0$
\While{stopping criterion not met}
	\State Sample a minibatch of $m$ iid examples $\{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(m)}\}$ with labels $\{\boldsymbol{y}^{(1)}, \dots, \boldsymbol{y}^{(m)}\}$
	\State Compute gradient estimate $\boldsymbol{\hat{g}} \gets \frac{1}{m}\nabla_{\boldsymbol{\theta}}\sum_{i}L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
	\State Update the accumulator: $\boldsymbol{r} \gets \boldsymbol{r} + \boldsymbol{\hat{g}} \odot \boldsymbol{\hat{g}}$
	\State Compute the update step: $\Delta \boldsymbol{\theta} = - \frac{\epsilon}{\delta + \sqrt{ \boldsymbol{r} }}\odot \boldsymbol{\hat{g}}$
	\State Apply update: $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \Delta\boldsymbol{\theta}$
\EndWhile
\end{algorithmic}
\end{algorithm}
```

#### RMSProp

AdaGrad has a problem: the accumulation of **all** the parameters from the beginning decreases the learning rate excessively. 
We can solve this by adopting an **exponential moving average**, where older history gets discarded.

In this way the convergence is much faster in situations where we reach a "wide" convex bowl with small gradients.


```pseudo
\begin{algorithm}
\caption{The RMSprop algorithm}
\begin{algorithmic}
\Require Global learning rate $\epsilon$
\Require Initial parameters $\boldsymbol{\theta}$
\Require Small constant $\delta$ for numerical stability
\State Initialize the accumulator $\boldsymbol{r} \gets 0$
\While{stopping criterion not met}
	\State Sample a minibatch of $m$ iid examples $\{\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(m)}\}$ with labels $\{\boldsymbol{y}^{(1)}, \dots, \boldsymbol{y}^{(m)}\}$
	\State Compute gradient estimate $\boldsymbol{\hat{g}} \gets \frac{1}{m}\nabla_{\boldsymbol{\theta}}\sum_{i}L(f(\boldsymbol{x}^{(i)}; \boldsymbol{\theta}), \boldsymbol{y}^{(i)})$
	\State Update the accumulator: $\boldsymbol{r} \gets \rho\boldsymbol{r} + (1-\rho)\boldsymbol{\hat{g}} \odot \boldsymbol{\hat{g}}$
	\State Compute the update step: $\Delta \boldsymbol{\theta} = - \frac{\epsilon}{\delta + \sqrt{ \boldsymbol{r} }}\odot \boldsymbol{\hat{g}}$
	\State Apply update: $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \Delta\boldsymbol{\theta}$
\EndWhile
\end{algorithmic}
\end{algorithm}
```

#### Adam

The most advanced optimization algorithm is Adam, which combines RMSprop with [[#Momentum]].

Momentum is incorporated as an estimate of first-order and second-order moments of the gradient (gradient and squared gradient, with exponential weighing).

There is also a bias correction to the moments to account for their fixed initialization.

This algorithm is generally robust to the choice of different _hyperparameters_.

### Hessian-free Optimization
As we saw in the section [[#Newton's Method in Optimization]], second-order methods are usually better then first-order one, both in terms of speed of convergence, and accuracy, especially in complex landscapes.

But they require to compute the Hessian matrix $\boldsymbol{H}$, and its inverse $\boldsymbol{H}^{-1}$ which is unfeasible, even for moderately sized networks.

Thus we want to find some alternatives that can approximate the behaviour of second-order methods, without requiring to compute explicitly $\boldsymbol{H}, \boldsymbol{H}^{-1}$.

#### Conjugate Gradients

One of the weaknesses of gradient descent is that it follows iteratively the direction of the gradient computed in successive points.

But this tend to be an inefficient approach, since the new direction may be (quasi) orthogonal to the previous one, thus inducing a _zig-zag_ pattern which wastes iterations by undoing progress done in some directions by the previous iterations. For instace in the case of a quadratic "bowl" landscape, the successive gradients are guaranteed to be orthogonal to each other, so this problem is really evident.

TODO: this section is to be finished and it's explained like shit

### Batch Normalization

Very deep neural networks involve the composition of several functions (layers). The gradient tells how to update each parameter, under the assumption that the other layers do not change. But this is not true in practice, since all the weights are updated simultaneously, hence unexpected results can happen that slow down the training of the network.

The idea is to reparametrize the inputs of some (even all) layers, by using the standard deviation and mean of the training minibatch. In this way backpropagation takes into account the normalization, and we stabilize and speed-up the training phase.

```pseudo
\begin{algorithm}
\caption{Batch normalization algorithm}
\begin{algorithmic}
\Input Values of $x$ over a mini-batch: $\mathcal{B} = \{x_1, \dots, x_m\}$
\Input Set of normalized values $\{y_1, \dots, y_m : y_i = BN(x_i)\}$
\State $\mu_{\mathcal{B}} \gets \frac{1}{m}\sum_{i=1}^{m}x_{i}$
\Comment{Compute mini-batch mean}
\State $\sigma^{2}_{\mathcal{B}} \gets \frac{1}{m}\sum_{i=1}^{m} (x_{i} - \mu_{\mathcal{b}})^{2} $
\State $\hat{x_{i}} \gets \frac{x_{i} - \mu_{\mathcal{B}}}{\sqrt{ \sigma^{2}_{\mathcal{B}} + \epsilon }} $
\Comment{Normalize input}
\Comment{Compute mini-batch variance}
\State $y_{i} \gets \gamma \hat{x_{i}} + \beta \equiv BN(x_{i})$
\end{algorithmic}
\end{algorithm}
```

Note that $\gamma$ and $\beta$ are learnable parameters from the network. Since $\hat{x_{i}}$ is a normalized value with zero mean and unitary variance, the two parameters allows the input to be shifted and scaled to better fit the network during training phase. This indirectly allow the network to control the standard deviation of the output ($\gamma$) and its mean ($\beta$) of the data originating distribution.

During inference we utilize the _global mean_ and _global standard deviation_, computed incrementally by a running average over all the examples seen during the training phase.

### Layer Normalization

An alternative to normalizing across all the mini-batch examples for each hidden unit separately it to normalize across the hidden-unit values for each data point separately using a certain function (e.g. $L_{2}$ norm).

This was introduced for RNN, since the distribution changes over the time steps making batch normalization infeasible.

![[dl-batch-layer-norm.png]]