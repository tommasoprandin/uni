## ML as Inverse Problem

In machine learning we have to solve the problem of, given a set of training examples $Tr = \{ (x_{1}, t_{1}), \dots, (x_{n}, t_{n}) \}$, infer the conditional distribution $p(y|x)$. Note that, while we have only access to a finite set of samples we would ideally want to infer the original distribuition for the observed phenomenon.

This is intrinsically _ill-posed_, since there are _infinitely many_ distributions that could have generated the observed training data. Potentially any probability density function with non-zero value in the observed $x$, could have generated the data.

To make the machine learning model useful we need a way to choose a specific distribution among all the possible ones.
This is achieved by prioritizing a certain category of distributions, considering the _domain knowledge_ of the problem to solve.

For instance, if we are trying to design an inference model for predicting the height of a person, we know that the samples are coming from a normal (gaussian) distribution.

## Inductive Bias

The preference for one choice over the other is called **inductive bias** (or prior knowledge), and plays a pivotal role for machine learning.

As the [[Deep Learning Introduction#No Free Lunch Theorem|no free lunch theorem]] states, it is not possible to learn purely from data, but the developers need to bring in some bias derived from domain knowledge.

Bias may be encoded _implicitly_, for instance the number of parameters of a network limits the complexity of functions that can be represented; or can be encoded _explicitly_ as reflection of prior knowledge.

Stronger biases tend to _improve performance_ on a more restricted set of tasks, but worsen the generalizability of the model.

Bias can be incorporated in the architecture of the model, for instance by specifying a certain kind of output layer activation function, or more generally by the nature of the model (e.g. NN are biased to the composition of lots of simple functions).
It can also be incorporated via a _regularization term_ to the error function using during training. This term penalizes hypothesis that do not fit well our bias.

## Regularization

**Regularization** is defined as any _modification_ made to a learning algorithm intended to reduce its _generalization error_, that is a technique used to trade _bias_ with _variance_.

In other words regularization aims to introduce some bias to limit overfitting which, in case of DNN is often a problem given their complexity. This gives better performance over the more general task (i.e. lower true error, see [[Deep Neural Networks Training#Errors and Overfitting|errors and overfitting]]).

### Parameter Norm Penalties

One of the oldest regularization strategies is to limit the capacity of the model by adding a _norm penalty_ to the parameters. In the case of NN parameters are represented by weights, and we want to favour solutions that pick smaller weights.

More formally, let $J(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y})$ be the loss function over the batch of examples $\boldsymbol{X}$, the label $\boldsymbol{y}$ and the parameters $\boldsymbol{\theta}$. We can favour smaller parameters with:
$$
\tilde{J}(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) = J(\boldsymbol{\theta}, \boldsymbol{X}, \boldsymbol{y}) + \alpha\Omega(\boldsymbol{\theta})
$$

Here $\alpha$ is the penalty factor and $\Omega$ is the norm function. In general only weights are regularized but not the bias terms in the neurons. This is because weights do influence multiple variables while bias only one so their influence is limited and normalizing them would incur the risk of underfitting.
Different norm functions give different properties, that is favour different solutions.

As seen in the image below applying norm penalties "raises" local minima that don't respect the bias condition, which is to have small weights. Thus only local minima that were close to the origin are now considered.
This has the effect of penalizing unlikely weights, thus potentially reducing overfitting.

![[dl-parameter-norm-penalty.png]]

#### $L^{2}$ Norm (Weight Decay)

The $L^2$ norm drives the weigths close to the origin with this norm function:
$$
\Omega(\boldsymbol{\theta}) = \frac{1}{2} {||\boldsymbol{w}||_{2}}^2
$$
where $||\boldsymbol{w}||_{2}$ is the Euclidean norm $= \boldsymbol{w}^T\boldsymbol{w}$.
Assuming there are no bias terms we have $\boldsymbol{\theta} = \boldsymbol{w}$, thus the regularized loss function becomes:
$$
\tilde{J}(\boldsymbol{w}; \boldsymbol{X}, \boldsymbol{y}) = \frac{\alpha}{2}\boldsymbol{w}^T\boldsymbol{w} + J(\boldsymbol{w}; \boldsymbol{X}, \boldsymbol{y})
$$

Then, compuing its gradient we obtain:
$$
\nabla_{\boldsymbol{w}}\tilde{J} = \alpha \boldsymbol{w} + \nabla_{\boldsymbol{w}}J
$$
so for a single SGD step (single example) with learning rate $\epsilon$, the weight update would be:
$$
\begin{align}
&\boldsymbol{w} \gets \boldsymbol{w} - \epsilon \nabla_{\boldsymbol{w}}\tilde{J} \\
&\boldsymbol{w} \gets \boldsymbol{w} - \epsilon(\alpha \boldsymbol{w} + \nabla_{\boldsymbol{w}}J) \\
&\boldsymbol{w} \gets (1-\epsilon \alpha)\boldsymbol{w} - \epsilon \nabla_{\boldsymbol{w}}J
\end{align}
$$

The weights get shrinked on each step by a constant factor $(1-\epsilon\alpha) < 1$ before applying the usual gradient update, that's why it is also called weight decay.

This regularization function can assume a probabilistic interpretation by observing that this converts _maximum likelihood_ to _maximum a-posteriori probability_, assuming weights follow a normal distribution (i.e. we introduce an inductive bias over the weights).

![[dl-l2-norm.png]]

##### Linear Regression
We will now see an example that illustrates how regularization is necessary in some ill-posed ML problems, where there are less examples than features. More generally this happens when the solution _is not unique_, and thus the problem is _underconstrained_.

Consider the general linear regression problem, where we want to find some weights $\boldsymbol{w} \in \mathbb{R}^m$ (vector) that minimizes the over the examples $\boldsymbol{X} \in \mathbb{R}^{n\times m}$ with labels $\boldsymbol{y} \in \mathbb{R}^n$. We will consider MSE loss function.
(Note that $n$ is the number of examples and $m$ the number of features)
$$
\begin{align}
 & \boldsymbol{X}\boldsymbol{w} = \boldsymbol{y} \\
 & J(w) = \frac{1}{n}\sum_{i = 1}^{n} ||X_{i;} \cdot \boldsymbol{w} - y_{i}||^2 = \frac{1}{n}||\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y}||^2
\end{align}
$$

Let's now compute the gradient of the loss:
$$
\nabla_{\boldsymbol{w}}J = \frac{1}{n}\cdot 2 \cdot (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})^T\boldsymbol{X} = \frac{2}{n}\cdot (\boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X} - \boldsymbol{y}^T\boldsymbol{X})
$$

Now, by imposing the gradient to zero, we can find the critical point(s) which could be local minima.
$$
\begin{align}
 & \frac{2}{n}\cdot (\boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X} - \boldsymbol{y}^T\boldsymbol{X}) = 0 \\
 & \boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X} = \boldsymbol{y}^T\boldsymbol{X} \\
 & \boldsymbol{w}^T = \boldsymbol{y}^T\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1} \\
 & \boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}
\end{align}
$$

But now we have a problem: to solve the equation for $\boldsymbol{w}$ we $(\boldsymbol{X}^T\boldsymbol{X})$ must be invertible! This is only true where the matrix is square and, more importantly _full rank_ (i.e. all the rows are linearly independent and all the columns).

This is impossible in the case where $n < m$, because we have less examples than features (i.e. less equations than variables), and so the problem is _underconstrained_, thus infinitely many solutions exists.

By applying regularization we can constrain the problem such that it is always solvable for any number of example $n$. (Of course the real-world accuracy would depend on how much example there are but a line fitting the examples is always findable).

Now the problem becomes:
$$
\begin{align}
 & \boldsymbol{X}\boldsymbol{w} = \boldsymbol{y} \\
 & J(w) = \frac{1}{n}\sum_{i = 1}^{n} ||X_{i;} \cdot \boldsymbol{w} - y_{i}||^2 + \alpha ||\boldsymbol{w}||^2= \frac{1}{n}||\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y}||^2+\alpha ||\boldsymbol{w}||^2
\end{align}
$$

The gradient of the loss becomes:
$$
\nabla_{\boldsymbol{w}}J = \frac{1}{n}\cdot 2 \cdot (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})^T\boldsymbol{X} + 2\alpha \boldsymbol{w}^T = \frac{2}{n}\cdot (\boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X} - \boldsymbol{y}^T\boldsymbol{X}+2\alpha \boldsymbol{w}^T)
$$

Finding the closed form solution:
$$
\begin{align}
 & \frac{2}{n}\cdot (\boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X} - \boldsymbol{y}^T\boldsymbol{X} + \alpha \boldsymbol{w}^T) = 0 \\
 & \boldsymbol{w}^T\boldsymbol{X}^T\boldsymbol{X}+\alpha \boldsymbol{w}^T = \boldsymbol{y}^T\boldsymbol{X} \\
 & \boldsymbol{w}^T(\boldsymbol{X}^T\boldsymbol{X}+\alpha \boldsymbol{I}) = \boldsymbol{y}^T\boldsymbol{X} \\
 & \boldsymbol{w}^T = \boldsymbol{y}^T\boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X}+\alpha \boldsymbol{I})^{-1} \\
 & \boldsymbol{w} = (\boldsymbol{X}^T\boldsymbol{X}+\alpha \boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}
\end{align}
$$

Now the matrix $(\boldsymbol{X}^T\boldsymbol{X} + \alpha \boldsymbol{I})^{-1}$ is full rank $\implies$ it is always invertible.
Notice that:
$$
\boldsymbol{X}^+ = \lim_{ \alpha \to 0 }(\boldsymbol{X}^T\boldsymbol{X} + \alpha \boldsymbol{I})^{-1}
$$
is the Moore-Penrose pseudo-inverse, commonly used to compute a "best fit" ([least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares "Ordinary least squares")) approximate solution to a system of linear equations that lacks an exact solution, which very closely map to our original problem with less examples than features.

#### $L_{1}$ Regularization

Now our regularization function becomes the $L_{1}$ norm of the weights:
$$
\begin{align}
 & \Omega(\boldsymbol{\theta}) = ||w||_{1} = \sum_{i=1}^{n} |w_{i}| \\
 & \tilde{J}(\boldsymbol{w}, \boldsymbol{X}, \boldsymbol{y}) = \alpha ||w||_{1} + J(\boldsymbol{w}, \boldsymbol{X}, \boldsymbol{y})
\end{align}
$$

Then the gradient of the regularized loss function $\tilde{J}$ becomes:
$$
\nabla_{\boldsymbol{w}}\tilde{J} = \alpha\ sign(\boldsymbol{w}) + \nabla_{\boldsymbol{w}}J(\boldsymbol{w}, \boldsymbol{X}, \boldsymbol{y})
$$
where the sign is applied element-wise.

This type of regularization forces solutions to be sparse, that is it performs _feature selection_.

Let's pick again an underconstrained linear regression task with less examples than features $\implies$ it is ill-defined since there are infinitely many solutions.

Now with regularization the problem becomes well-defined, by the properties and shape of the $L_{1}$ norm the solutions will tend to be on the "tips" of the regularization term, this is especially true in _high-dimensional spaces_.

Below there is a comparison between $L_{1}$  and $L_{2}$ norm, that highlights how $L_{1}$ regularization favours sparsity.

![[dl-l1-l2-comparison.PNG]]

#### Parameter Norm Regularization Summary

We have seen that regularization is necessary in some ill-posed ML problems, for instance when we have **less examples than features**.
In general this is true when the problem is underconstrained, that is the solution _is not unique_.

Regularization (especially $L_{2}$) guarantees that the optimization problem is well-defined. This is done in practice using the Moore-Penrose pseudoinverse instead of the normal inverse, thus stabilizing underconstrained problems (i.e. generalizing the inverse matrix notion even when the matrix should not be invertible by getting the best possible fit).

Regularization is very useful to _avoid overfitting_ by penalizing weights with high norm, thus unlikely to be generally a good solution.


### Data Augmentation

The best way to make a ML model generalize better is to train it on more data, but in practice the amount of available examples is limited.

One way to get around this is to generate "fake" training data by applying some transfomations to the training instances.

For instance in object detection we would like the model to be invariant to translations, rotations or scaling of the original image. Thus we could apply some of this transformation to generate new training examples starting from the existing ones. This will train the network to "ignore" certain undesired transformations.

Also **injecting noise** at some point in the neural network pipeline can be an effective data augmentation techinique. Noise can be added:
- In the input examples
- In the hidden representations (neuron dropout)
- In the target (label smoothing where, for categorization tasks with [[Deep Learning Introduction#^a39cb4|softmax]], "hard" 0 and 1 target labels are replaced with "soft" targets $\frac{\epsilon}{k-1}$ and $1 - \epsilon$, in order to avoid increasing weights when convergence is impossible to reach for hard targets).

### Semi-Supervised Learning

In the context of deep learning, _semi-supervised learning_ usually refers to learning a representation $\boldsymbol{h} = f(\boldsymbol{x})$. The goal in this case is to learn a _representation_ so that examples from the same class have similar representations.

To learn this _hidden representation_ $\boldsymbol{h}$, the models exploits both _unlabelled examples_ drawn from $P(\boldsymbol{x})$, and _labelled examples_ from $P(\boldsymbol{x}, y)$, in order to estimate the conditional probability $P(y \mid \boldsymbol{x})$.

This new representation in a new space may allow classifiers to achieve better generalization.
An example of this is using an unsupervised generative model modeling $P(\boldsymbol{x})$ that shares parameters with a discriminative model estimating $P(y \mid \boldsymbol{x})$ (i.e. it tries to predict the labels from unlabeled examples).

This can be used to label unlabeled examples or to generate new ones to train a new network.
![[dl-semisup-learning.png]]

### Transfer Learning (Pre-training)

Another option widely used when training data for the specific task is limited is to leverage other datasets related to similar tasks to improve performance.

The strategy is called _transfer learning_ and it works as follow:
1. The network is trained to perform a _secondary related task_ for which more training data is available
2. The resulting network is adapted to the original task by replacing the output layer (possibly adding more) to match the required output
3. Then there are two possible options:
	1. Keep the original model weight fixed and only train the newly added layer(s)
	2. Fine tune the entire new network

This approach works because the network trained on the related task will have learned an internal representation that works well also for the task at hand (since they are similar). It can be viewed as "initializing" most of the parameters of the final network into a space that is likely to work well for the original task, given the similarity of them.

![[dl-transfer-learning.png]]

### Multi-Task Learning

One idea similar to transfer learning is to segment a neural network in a _shared_ part and a _specialized_ one.

The _shared_ part is used across different (related) tasks and the goal is to learn a "high level" representation for data that is valid across all the specialized tasks. This has a good **regularization effect** since the shared part has to work well for every task, thus it needs to be general.

Then on top of the shared part there is a branch for every _specialized_ task, where the network diverges to implement different tasks.
The loss functions for each task are combined together into a joint loss function for training.

Compared to transfer learning the training phase is done simultaneously for every subtask in the network.

![[dl-multi-task.jpg]]

### Self-Supervised Learning

When absolutely _no labeled data_ is available, we can _create_ large amounts of "free" labeled data using _self-supervised learning_ and use this for transfer learning.

There are two main categories of self-supervised learning:
- **Generative** self-supervised learning where part of each data example is _masked_ and the goal is to predict the missing part.
- **Contrastive** self-supervised learning is an approach that trains models to recognize what makes examples similar or different. The core principle involves:
	- Creating multiple **augmented views** of the same example (positive pairs) through transformations such as cropping, rotation, color jittering, etc.
	- Bringing the representations of positive pairs closer together in the embedding space
	- Pushing away representations of different examples (negative pairs)
	The model learns to create embeddings where:
	- Different views of the same object are close together
	- Views of different objects are far apart

![[dl-regularization-tech.png]]

### Early Stopping

In order to reduce the tendency to overfit in large models (like deep neural networks), we can try to "stop early" in the case where, even after multiple epochs of training, the _validation loss_ does not decrease (i.e. improvement hits a _plateau_).

This has a **regularization effect** that limits the parameter space to be close to the initial parameters. In practice it can have a very similar effect as [[#$L {2}$ Norm (Weight Decay)]] regularization, that penalizes large weights. So if weights starts small (as usually is the case), then the effects will be very similar.

![[dl-early-stopping-1.png]]
![[dl-early-stopping-2.png]]

### Parametery Tying and Sharing

Another way we can introduce _bias_ in our network architecture is to impose that some weights (parameters) should be close one another, or exactly the same.

For instance let's pick two models $A$ and $B$ that solve similar tasks. We can impose that their weights should be similar (because of the similar tasks) by adding a penalty on the distance between their parameters, for instance on their quadratic Euclidean distance:
$$
\Omega(w^{(A)}w^{(B)}) = ||w^{(A)}w^{(B)}||^{2}_{2}
$$

We can also impose that parameters are identical simply by sharing them across multiple neurons. This is very commonly done in [[Convolutional Neural Networks]], where weights for the kernels are shared across the image.

These techniques are examples of incorporating _domain knowledge_ in the network architecture.

### Ensemble Methods

_Ensemble_ methods are based on the idea of training different models, in which every of them contribute to the final output (e.g. by combining them with a majority vote).

There are two possible approaches:
#### Bagging

With _bagging_ (bootstrap aggregating) the idea is to combine multiple _highly powerful_ models that tend to overfit a lot, and to combine their results with majority vote, in order to reduce total variance. This works really well for neural networks since they are a notoriously powerful model.
Since, even by training on the same examples, neural networks tend to be very sensible to hyperparameters and random selections, they will often make partially independent errors.

To perform training, let $k$ be the number of different models used, we construct $k$ different datasets with the same number of samples from the original dataset, with samples picked with replacement. This will make so that approximately $\frac{1}{3}$ of the examples will ve repeated.

Then a model is trained for each dataset, differences in training sets result in differences in the resulting models.

This approach can be quite expensive, since now we need to train $k$ different models.

![[dl-bagging.png]]
#### Boosting

_Boosting_ is the dual approach to bagging, where multiple _weak_ models are combined to build a more powerful one. This approach is very seldomly used in deep learning due to the characteristics of deep neural networks.

### Dropout

We have seen that _bagging_ is a very powerful generalization tool for neural networks. Unfortunately the cost of training $k$ different neural networks makes it difficult to apply in practice for $k \geq 10$.

_Dropout_ represents a very good approximation to training and evaluating a bagged ensemble of _exponentially many_ neural networks.
Specifically dropouot trains an ensemble consisting of all subnetworks that can be constructed by removing non-output units from a base network.

To train with dropout we use a minibatch-based learning algorithm (like SGD) that makes small steps. For every example loaded we randomly sample a **binary mask** $\mu$ to "shut down" some neurons. The mask for each unit is sampled independently from all the others, with probability of selecting a neuron representing an _hyperparameter_.

More formally, suppose the binary mask vector $\boldsymbol{\mu}$, and the loss fuction of parameters and mask $J(\boldsymbol{\theta, \mu})$.
Dropout training consist of minimizing $\mathbb{E}_{\boldsymbol{\mu}}J(\boldsymbol{\theta}, \boldsymbol{\mu})$, this term contains exponentially many terms but, by taking different samples of $\boldsymbol{\mu}$ we can estimate that by taking the mean over all masks tried.

Dropout training differs from bagging because here the models are not independent from each other, but instead they share parameters inherited from the parent neural network. This sharing makes possible to represent an exponential number of models with a tractable amount of memory. Additionally in dropout each model is not trained to reach the convergence on its training set, but it is only used to train on a subset of examples, and then moves on to the next model.

![[dl-dropout.png]]

To make prediction a bagged architecture takes the mean from all its $k$ members:
$$
p(y \mid \boldsymbol{x}) = \frac{1}{k}\sum_{i=1}^{k} p^{(i)}(y \mid \boldsymbol{x})
$$

In the case of dropout the arithmetic mean over all the masks becomes:
$$
p(y \mid \boldsymbol{x}) = \sum_{\boldsymbol{\mu}}p(\boldsymbol{\mu})p(y \mid \boldsymbol{x}, \boldsymbol{\mu})
$$
with $p(\boldsymbol{\mu})$ deriving from the chosen probability distribution for mask sampling.
This would be infeasible to compute, because there is $2^{d}$ possible masks for a network with $d$ units that can be dropped. But we can approximate this with good accuracy by sampling a number of masks and then take the average result (usually 20 is enough).

#### Weight Scaling

If, instead of using arithmetic mean to compute the inference result, we used _geometric mean_ we could greatly improve performance.

The geomean of results from all the masks is:
$$
\tilde{p}_{ensemble}(y \mid \boldsymbol{x}) = \sqrt[2^{d}]{ \prod_{\boldsymbol{\mu}} p(y \mid \boldsymbol{x}, \boldsymbol{\mu}) }
$$
this is not normalized so it needs to be renormalized
$$
p_{ensemble}(y\mid \boldsymbol{x})= \frac{\tilde{p}(y \mid \boldsymbol{x})}{\sum_{y'}\tilde{p}(y' \mid \boldsymbol{x}) }
$$

The key "trick" is that we can approximate $p$ by only evaluating $p(x \mid \boldsymbol{x})$ on the network with all units, but with weights going out of each uni multiplied by the probability of including it in the mask. This is called **weight scaling inference rule**.
This trick has an exact formulation for certain classes of models, but not for all of them; in any case it has been shown to work well in practice.

For a simple example consider a softmax regression classifier with $n$ input variables represented by vector $\boldsymbol{\mathbf{v}}$. Notice that there is one layer, thus the mask only acts on the inputs.
$$
P(\mathbf{y} = y \mid \mathbf{v}) = \text{softmax}(\boldsymbol{W}^{T}\mathbf{v} + \boldsymbol{b})_{y} \text{ (softmax is $y$ indexed for the $y$-th entry)}
$$
We can index into the family of submodels by element-wise multiplication with the mask:
$$
P(\mathbf{y} = y \mid \mathbf{v}; \boldsymbol{\mu}) = \text{softmax}(\boldsymbol{W}^{T}\left( \mathbf{v} \odot \boldsymbol{\mu}) + \boldsymbol{b} \right)_{y}
$$

Substituting this result into the geomean ensemble prediction to obtain the probability distribution for the entire ensamble:
$$
\tilde{P}_{ensemble}(\mathbf{y} = y \mid \mathbf{v}) = \sqrt[2^{n}]{ \prod_{\boldsymbol{\mu} \in \{ 0,1 \}^{n}} p(\mathbf{y} = y \mid \boldsymbol{x}, \boldsymbol{\mu}) }
$$
then renormalize it:
$$
P_{ensemble}(\mathbf{y = }y\mid \boldsymbol{x})= \frac{\tilde{P}(\mathbf{y} = y \mid \boldsymbol{x})}{\sum_{y'}\tilde{P}(\mathbf{y} = y' \mid \boldsymbol{x}) }
$$

TODO: complete...

### Adversarial Training

By how ML models are trained there is the possibility for malicious actors to specifically craft inputs that are indistinguishable for humans, but that damage significantly the quality of the prediction of the network, as in the following example where a panda image gets augmented with some special noise in order to "confuse" the network.

By including there _adversarial examples_ in the training phase, we can not only improve security, but also provide generic regularization, since it is still a form of noise introduction. 

![[dl-adversarial-training.png]]