_Deep neural networks_ with _non-linear_ activation functions can theoretically approximate any (continuous) function with typically better performances compared to other techniques. A key point is to appropriately _train_ the network, which essentially means finding the set of weights such that the loss is minimized.

## Notions of Statistical Learning Theory

The _dataset_ we have will be supposed to have random samples i.i.d. according to some probability distribution $\mathcal{D}$.

In general we are interested in _generalizing_ the training set to all unseen example.

Let's give some definitions:

### Instances and Hypothesis

**Instance Space**: the set of all possible examples that the model can encounter, or be trained on. (e.g. for an image classification task all the possible images).

**Hypothesis Space**: the set of all the functions that can be implemented by the system: $\mathcal{H} = \{ h: \boldsymbol{x} \to y\ |\ h \text{ is implementable, } \boldsymbol{x} \in \text{ Instance Space}, y \in \text{Output Space} \}$

Machine learning aims to find a function $h \in \mathcal{H}$ that replicates (or follows very closely) the function to be learned $f$. Obviously an exaustive search over $\mathcal{H}$ is infeasible.

### Errors and Overfitting

**True Error**: the _true error_ $error_{D}(h)$ of a hypothesis $h$ respect to a target concept $c$ and a distribution $D$, is the probability that $h$ will misclassify an instance drawn from the distribution $\mathcal{D}$.
$$
error_{D}(h) \equiv P_{x \in \mathcal{D}}[c(x) \neq h(x)]
$$

**Empirical Error**: the _empirical error_ $error_{Tr}(h)$ of hypothesis $h$ with respect to the set of samples $Tr$ is the number of examples of $Tr$ that $h$ misclassifies:
$$
error_{Tr}(h) = P_{(x, y) \in Tr}[y \neq h(x)]
$$

Given a hypothesis $h \in \mathcal{H}$, it overfits $Tr$ if there exists another hypothesis $h' \in \mathcal{H}$ such that: 
$$
\begin{align}
& error_{Tr}(h) < error_{Tr}(h') \\
& error_{\mathcal{D}}(h) > error_{\mathcal{D}}(h')
\end{align}
$$
which means the hypothesis performs "too good" on the training set and does not generalize well.

![[dl-underfit-overfit.jpg]]
![[dl-overfitting-bias.png]]

Thus we want a way to **estimate the true error**. We have two possible approaches:

1. Introduce a $cm(\mathcal{H})$, that gives a measure of the complexity of a particular hypothesis space (e.g. polynomial functions). This measure acts to penalize function types that are very complex and thus more prone to overfit (i.e. more likely to perform very well only over the training data).
	$$
	error_{\mathcal{D}}(h) \leq error_{Tr}(h) + cm(\mathcal{H})
	$$
	This approach is hard to define in practice and thus _rarely used_.
2. Compute the error on unseen data, which means holding out some examples from the training dataset in order to perform some unbiased evaluation after learning. This approach is the most commonly used in practice.

### Model Selection and Validation Set

The hold-out procedure is the technique used to evaluate the true error of a newly trained machine learning model.

It is composed of four parts:
1. **Split**: Divide you dataset in three parts:
	- _Training Set ($Tr$)_ used to train the model
	- _Validation Set ($Va$)_: a held-out portion used to _evaluate_ model performance and _tune_ hyperparameters.
	- _Test Set($Te$)_: a separately held-out set used for a _final, unbiased_ evaluation of the trained model.
2. **Train and Evaluate**: Train multiple models with different hyperparameters, iteratively evaluating them only using the validation set.
3. **Hyperparameter Selection**: Compare the results of phase 2 and determine the best hyperparameters.
4. **Test**: Perform final evaluation of the model with the chosen hyperparameters over the held-out test set.

This process allows us to evaluate the unbiased _true error_ of the model over the given input distribution.

Unfortunately often we don't have access to enough data to implement this approach.

#### K-Fold Cross Validation

A possible solution to this, which is the currently _state-of-the-art_ solution, is to iteratively apply the steps of the held-out procedure over a training set partitioned in $k$ subsets, one of those acting as the validation set, while the others actually used for training.

The steps are:
1. **Split**: Split your dataset into a training set and a test set for final evaluation
2. **Partition**: Partition the training set into $k$ different subsets $P = \{  P_{1}, \dots, P_{k}\}$.
3. **Training**: From the partitions created before apply the hold-out procedure $k$ times using, for the $i$-th iteration, the set $Tr_{i} = \bigcup_{i = 0}^k (P \setminus P_{i})$ for training, and the set ${Va}_{i} = P_{i}$ for the validation.
4. **Final Error Estimation**: After this procedure there will be $k$ different hypothesis emerged from the previous iterations $h_{1}, \dots, h_{k}$. The final error will be obtained by averaging the errors:
	$$
	E_{final} = \sum_{i=1}^{k} E_{i}
	$$

The above procedure is repeated for different values of the parameters and the model with smaller final error is selected. Finally evaluation is performed on the held-out test set.

![[dl-kfold-validation.png]]
## Gradient Descent

(For a mathematical reference for the following section see the [[Multivariate Calculus Cheat Sheet | multivariate calculus cheat sheet]])

Gradient descent is the optimization algorithm used to train neural networks by iteratively adjusting weights to minimize the loss function.

The basic idea is to compute the gradient (direction of steepest increase) of the loss function with respect to each weight, then move in the opposite direction. The size of each step is determined by the learning rate.

![[dl-gradient.png]]

The weight update rule is:

$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

Where:

- $\alpha$ is the learning rate
- $\frac{\partial L}{\partial w}$ is the partial derivative of the loss function with respect to the weight

Variants of gradient descent include:

1. **Batch Gradient Descent**: Computes gradients using the entire dataset
    
    - Provides the most accurate gradient direction
    - Computationally expensive for large datasets
2. **Stochastic Gradient Descent (SGD)**: Updates parameters after each training example
    
    - Faster but with noisy updates
    - Can help escape local minima
3. **Mini-batch Gradient Descent**: Updates parameters after computing gradients on small batches of data
    
    - Balances accuracy and computational efficiency
    - Most commonly used approach

Modern optimizers like Adam, RMSprop, and AdaGrad build on these concepts with adaptive learning rates and momentum to improve convergence.


![[dl-gradient-descent.png]]

### Challenges of Gradient Descent

#### Local Minima Problem

The loss function of a neural network typically has multiple local minima—points where the function value is lower than all nearby points, but not necessarily the lowest possible value globally.

Key challenges:

- Gradient descent can get trapped in local minima, preventing the model from reaching better solutions
- In high-dimensional spaces, distinguishing between local and global minima becomes difficult
- The quality of the final model depends on where the optimization process converges

Recent research suggests that in very high-dimensional spaces (as in deep networks), true local minima may be less problematic than previously thought. Many apparent local minima are actually saddle points, and most local minima tend to have similar loss values.

#### Saddle Points and Plateaus

Saddle points are locations where the gradient is zero but the point is neither a maximum nor a minimum. They have positive curvature in some directions and negative curvature in others.

Plateaus are flat regions with very small gradients where:

- Learning slows dramatically
- Training appears to stall
- The optimizer makes minimal progress for many iterations

The prevalence of saddle points increases exponentially with the dimensionality of the parameter space, making them more common than local minima in deep networks.

#### Ill-Conditioning and Ravines

Optimization landscapes often contain ravines—steep, narrow valleys with a gently sloping floor:

- Gradients point more strongly in the direction of steep walls
- Standard gradient descent oscillates across the ravine rather than moving efficiently along it
- Learning becomes slow and unstable

This problem is mathematically described as ill-conditioning, where the condition number (ratio of largest to smallest eigenvalues of the Hessian matrix) is large. The condition number can also be seen as the ratio between the maximum and minimum curvature in the objective function.

![[dl-gd-problems.png]]

### The Jacobian Matrix

The Jacobian matrix represents the first-order partial derivatives of a vector-valued function with respect to its inputs.

For a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with input $x \in \mathbb{R}^n$ and output $y \in \mathbb{R}^m$, the Jacobian matrix $J$ is defined as:

$$J = \begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n}  \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ 
 \frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix} = 
 \begin{bmatrix}
\nabla_{\boldsymbol{x}}y_{1} \\
\nabla_{\boldsymbol{x}}y_{2} \\
\vdots \\
\nabla_{\boldsymbol{x}}y_{3}
\end{bmatrix}$$

#### Utility of the Jacobian

In neural networks, the Jacobian is particularly useful for:

1. **Backpropagation**: The chain rule for computing gradients relies on Jacobian matrices of each layer
2. **Sensitivity analysis**: Understanding how changes in inputs affect outputs
3. **Network behavior**: Analyzing properties like robustness and stability

For classification problems, the Jacobian shows how changes in input features affect class probabilities, providing insight into feature importance and model behavior.

### The Hessian Matrix and Second-Order Optimization

The Hessian matrix encapsulates second-order derivative information, providing crucial insights about the curvature of the loss landscape.

For a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ with input $x \in \mathbb{R}^n$, the Hessian matrix $H \in \mathbb{R}^{n\times n}$ is defined as:

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n}  \\
 \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n}  \\
 \vdots & \vdots & \ddots & \vdots  \\
 \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$

For a vector field $f: \mathbb{R}^n \to \mathbb{R}^m$ the hessian matrix is a _tensor_: $H \in \mathbb{R}^{n\times n\times m}$

#### Utility of the Hessian

The Hessian provides valuable information for optimization:

1. **Characterizing critical points**:
    
    - Positive definite Hessian (i.e. eigenvalues are positive) → local minimum
    - Negative definite Hessian (i.e. eigenvalues are negative) → local maximum
    - Mixed eigenvalues → saddle point
2. **Improving optimization**:
    
    - Second-order methods like Newton's method use the Hessian to determine step direction and size
    - Enables faster convergence in well-behaved regions
3. **Understanding landscape geometry**:
    
    - Eigenvalues reveal the curvature in different directions
    - Large eigenvalue ratios indicate ill-conditioning
4. **Analyzing model robustness**:
    
    - Sharp minima (large Hessian eigenvalues) typically generalize worse than flat minima (small eigenvalues)
    - Models converging to flatter regions tend to perform better on unseen data

#### Condition Number

The ratio between the maximum and minimum non-zero eigenvalues of the Hessian matrix is called the _condition value_. 

It represents the _sensitivity_ of the original functions to small changes in input.

A problem for which the condition number is very high is said to be _ill-defined_, this means that small changes in the input greatly impact the answer, thus making the solution harder to find.

In machine learning context we want to have _well-defined_ loss functions (i.e. "stable" w.r.t the input) so that it is easier to find their solution. The graph for a _well-defined_ loss function is "smooth" and without steep transitions.

![[dl-high-cn.gif | center ]]
Function with high condition number. Notice how it is hard to find the minima, imagine dropping a ball in the valley it will struggle to stop in the minima.

![[dl-low-cn.gif | center]]
Function with low condition number. The minima is very well defined and a ball dropped in there will stop very soon and precisely.

#### Practical Considerations

While the Hessian offers valuable insights, its direct computation presents challenges:

- For a network with $n$ parameters, the Hessian requires $O(n^2)$ storage
- Computing all elements requires $O(n^2)$ backpropagation passes
- This becomes prohibitive for modern networks with millions of parameters

Hence, practical applications often use approximations:

- Diagonal approximations consider only second derivatives with respect to the same parameter
- Low-rank approximations capture dominant curvature directions
- Hessian-vector products can be computed efficiently without forming the full Hessian

## Backpropagation

While training a deep neural network we want to find the weights that minimize the loss function. Let's denote the parameters (weights) of tha network as $\boldsymbol{\theta}$, the loss function as $J(\boldsymbol{\theta})$, and the output of the network as $\hat{y}$.

Then two steps are needed for training:
1. A _forward pass_ where the input is propagated to the network to produce the output $\hat{y}$, and thus the cost $J(\boldsymbol{\theta})$ (**forward propagation**)
2. A _backwards pass_ where, starting from the loss value, we roll back  to compute the gradient for each parameter (i.e. determine the contribute of each weight to the final loss).

The second step is called **back propagation** and it is a method for computing the gradient. It is DOES NOT perform the learning phase (that is done by an algorithm that USES the gradient computed by back propagation).

Back propagation is in practice just an implementation of the chain rule of calculus (ref. to [[Multivariate Calculus Cheat Sheet#Chain Rule for Multivariate Functions | here]]).

To optimize the computation is uses _dynamic programming_ (table filling) to avoid recomputing repeated expressions, thus speeding up the algorithm by using some memory.

![[dl-backpropagation-arch.png]]

There it follows an example of gradient computation on a feedforward neural network with sigmoidal activation function.

### Example for Sigmoidal Units

![[dl-backpropagation-exa.pdf]]

### Computational Graph

To simplify the visualization and representation of computations we introduce a notation that encodes variables as nodes and edges as operations between them:

![[dl-computation-graph.png]]

### Automatic Differentiation

Automatic differentiation (AD) is a computational technique that efficiently calculates the derivatives needed during the backpropagation phase of neural network training.
Automatic differentiation tracks derivatives through all computational operations by applying the chain rule systematically. Unlike numerical differentiation (which approximates derivatives) or symbolic differentiation (which manipulates mathematical expressions), AD computes exact derivatives efficiently by breaking calculations into elementary operations and applying differentiation rules.

#### Neural Network Implementation

1. **Forward Pass**: During the forward pass, AD builds a computational graph that records all operations performed to compute the loss function.
    
2. **Reverse-Mode AD**: Backpropagation uses reverse-mode AD, which efficiently calculates gradients of a scalar output (loss) with respect to many inputs (model parameters). It's called reverse-mode because it starts computation from the output layer back to the input.
    
3. **Chain Rule Application**: Starting from the loss function, AD propagates derivatives backward through the computational graph, applying the chain rule at each step.
4. **Gradient Accumulation**: As backpropagation traverses the graph, gradients are accumulated for parameters used multiple times.
    
Frameworks like TensorFlow and PyTorch implement AD using:

1. **Dual Number Representation**: Each value tracks both its primal value and its gradient.
    
2. **Operator Overloading**: Mathematical operations are overloaded to compute both the result and the necessary information for gradient calculation.
    
3. **Computational Graph**: Either static (TensorFlow 1.x) or dynamic (PyTorch, TensorFlow Eager) graphs record the operations.