## What are Deep Neural Networks?

Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. For example, in image recognition, lower layers might identify edges, while higher layers might identify concepts relevant to human understanding like digits, letters, or faces.

The power of deep learning comes from:

1. **Automatic feature extraction**: Unlike traditional machine learning that requires manual feature engineering, deep neural networks automatically discover the representations needed for feature detection or classification directly from raw data.
    
2. **Universal function approximation**: Deep networks can theoretically approximate any continuous function to arbitrary precision, making them incredibly versatile.
    
3. **Hierarchical learning**: The layered structure allows the model to learn increasingly complex abstractions, mimicking how humans understand the world.
    
![[dl-deepnetwork-arch.png | center | 500]]

## Common Use Cases

Deep learning has transformed numerous fields by achieving breakthrough performance:

- **Computer Vision**: Image classification, object detection, facial recognition, medical imaging analysis
- **Natural Language Processing**: Machine translation, sentiment analysis, text generation, question answering
- **Speech Recognition**: Voice assistants, transcription services, real-time translation
- **Recommendation Systems**: Product recommendations, content personalization, advertisement targeting
- **Generative Modeling**: Image generation, text-to-image synthesis, artistic style transfer
- **Reinforcement Learning**: Game playing, autonomous vehicles, robotic control
- **Scientific Applications**: Protein folding prediction, drug discovery, weather forecasting

## The Perceptron: Building Block

The perceptron is the fundamental building block of neural networks, inspired by biological neurons. It takes several binary inputs and produces a single binary output.

The perceptron computes a weighted sum of inputs, adds a bias term, and then applies a step function:

$$output = \begin{cases}
1 & \text{if } \sum_i w_i x_i + b > 0 \\
0 & \text{otherwise}
\end{cases}$$
Where:

- $x_i$ are the inputs
- $w_i$ are the weights associated with each input
- $b$ is the bias term

The perceptron can learn to classify linearly separable patterns by adjusting weights and bias.

![[dl-perceptron-arch.png | center | 500]]

## Linear Neural Networks

Linear neural networks consist of layers of neurons where each neuron computes a linear combination of its inputs:

$$\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}$$

Where:

- $\boldsymbol{W}$ is a weight matrix
- $\boldsymbol{x}$ is the input vector
- $\boldsymbol{b}$ is the bias vector
- $\boldsymbol{y}$ is the output vector

For a multi-layer linear network, the output of each layer becomes the input to the next. However, stacking multiple linear layers is mathematically equivalent to having just one linear layer, as the composition of linear functions is still a linear function.

This limitation highlights a critical problem: linear models can only learn linear relationships. Many real-world problems, however, involve complex non-linear patterns.

## The Need for Non-linearity

A classic example demonstrating the limitations of linear models is the XOR problem. The XOR (exclusive OR) function outputs 1 only when exactly one of its inputs is 1:

|x₁|x₂|XOR(x₁, x₂)|
|---|---|---|
|0|0|0|
|0|1|1|
|1|0|1|
|1|1|0|

No single linear boundary can separate the output classes (0s and 1s), making it impossible for a linear model to learn this function.

By introducing non-linearity between layers, neural networks can approximate complex functions and solve problems like XOR. This is what gives neural networks their expressive power.

![[dl-xor-problem.jpg | center | 500]]

## Activation Functions

Activation functions introduce non-linearity into the network. They determine whether and to what extent a neuron's signal should be transmitted to the next layer.

Common activation functions include:

1. **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
    
    - Outputs between 0 and 1
    - Historically popular but suffers from vanishing gradient problems

1. **Hyperbolic Tangent (tanh)**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
    
    - Outputs between -1 and 1
    - Zero-centered, which helps with optimization
3. **Rectified Linear Unit (ReLU)**: $f(x) = \max(0, x)$
    
    - Outputs the input if positive, otherwise 0
    - Computationally efficient and helps mitigate vanishing gradient problems
    - Most widely used in modern networks
4. **Leaky ReLU**: $f(x) = \max(\alpha x, x)$ where $\alpha$ is a small constant
    
    - Addresses the "dying ReLU" problem by allowing a small gradient when inactive
5. **Softmax**: $\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$
    
    - Converts a vector of values to a probability distribution
    - Commonly used in the output layer for multi-class classification

![[dl-activation-functions.png | center | 500]]

## Loss Functions and Training

Neural networks learn by adjusting their weights and biases to minimize a loss function that quantifies the difference between predicted outputs and desired targets.

Common loss functions include:

1. **Mean Squared Error (MSE)**: $MSE = \frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2$
    
    - Used for regression problems
2. **Cross-Entropy Loss**: $L = -\sum_{i=1}^n y_i \log(\hat{y}_i)$
    
    - Used for classification problems
    - Measures the difference between two probability distributions

The training process follows these steps:

1. **Forward pass**: Input data is fed through the network to produce predictions
2. **Loss calculation**: The difference between predictions and targets is quantified
3. **Backward pass (backpropagation)**: Gradients of the loss with respect to each weight are calculated
4. **Parameter update**: Weights and biases are adjusted to reduce the loss

This process is repeated iteratively with batches of training data until the model converges to a satisfactory solution.

## Regression vs Classification

Deep learning can be applied to both regression and classification problems:

### Regression

- **Task**: Predict continuous numerical values
- **Output layer**: Typically has linear activation (no activation function)
- **Loss function**: Often Mean Squared Error (MSE) or Mean Absolute Error (MAE)
- **Examples**: Housing price prediction, temperature forecasting, age estimation

### Classification

- **Task**: Assign inputs to discrete categories or classes
- **Output layer**:
    - Binary classification: Sigmoid activation with threshold
    - Multi-class classification: Softmax activation
- **Loss function**: Binary Cross-Entropy or Categorical Cross-Entropy
- **Examples**: Spam detection, image recognition, sentiment analysis

The architecture of the network (number of output neurons, activation functions, loss function) is tailored to the specific type of problem.

## Universal Approximation Theorem

> A _feedforward network_ with a linear input layer and _at least_ one hidden layer with any non-linear _squashing_ activation function, can approximate any continuous function with an arbitrarily small error, given sufficient hidden units.

This means that for any possible continuous function there is a feedforward neural network that can approximate it with arbitrary precision.
A single layer is potentially sufficient but is usually infeasibly large for any meaningful function and, since it cannot represent hierarchical relations in data, it struggles to generalize well.

## No Free Lunch Theorem

> Averaged over _all possible_ data-generating distributions, every classification algorithm performs the same (i.e. has the same error rate), when classifying previously unobserved points.

This means that there is not a universally better machine learning algorithm, that can cover all the possible data distributions, but we need to create custom models for any task at hand, that is strictly optimized fot that data distribution.

## Advantages of Deep Neural Networks

Neural networks with many layers (deep networks) often outperform those with few layers (shallow networks), even when the total number of parameters is similar. This section explores the theoretical and practical advantages that depth provides.

### Representational Efficiency

The key advantage of deep networks lies in their representational efficiency. While shallow networks can theoretically approximate any continuous function (a property known as universal approximation), they often require an exponentially larger number of neurons to represent functions that deep networks can represent more compactly.

Consider a parity function on $n$ bits, which outputs 1 if an odd number of bits are 1, and 0 otherwise. A shallow network would need approximately $2^n$ neurons to represent this function, while a deep network could represent it with $O(n)$ neurons arranged in $O(\log n)$ layers.

### Hierarchical Feature Learning

Deep networks learn features in a hierarchical manner:

- Early layers learn simple, local features
- Middle layers combine these into more complex patterns
- Later layers assemble these patterns into abstract concepts

This hierarchy mirrors how many natural processes are organized and allows the network to build increasingly sophisticated representations.

![[dl-hierarchical-learning.png]]

### Parameter Sharing

Deep architectures, particularly convolutional neural networks (CNNs), benefit from parameter sharing. The same filter parameters are applied across different parts of the input, which:

- Dramatically reduces the number of parameters needed
- Enforces translation invariance (recognizing patterns regardless of position)
- Improves generalization

## Empirical Evidence

Research has consistently shown that, given the same parameter budget, deeper networks generally achieve higher performance than wider, shallower networks. This has been demonstrated across various domains including image recognition, natural language processing, and speech recognition.

The breakthrough performance of networks like AlexNet, VGGNet, and ResNet was largely attributed to their increased depth, establishing a trend toward deeper architectures in the field.

## Comparison: Deep vs. Shallow Networks

|Aspect|Shallow Networks|Deep Networks|
|---|---|---|
|**Number of layers**|Few (typically 1-3)|Many (can be hundreds)|
|**Representational capacity**|High width needed for complex functions|More efficient representation of complex functions|
|**Parameter efficiency**|Less efficient (may require exponentially more parameters)|More efficient for many problem classes|
|**Feature abstraction**|Limited hierarchy of features|Rich hierarchy from simple to complex features|
|**Training difficulty**|Easier to train (fewer vanishing/exploding gradient issues)|More challenging (requires techniques like batch normalization, residual connections)|
|**Overfitting risk**|Can overfit with too many parameters in a single layer|Better generalization when properly regularized|
|**Computational requirements**|Can be parallelized more easily|Sequential nature of layers can limit parallelization|
|**Expressivity**|Limited ability to capture compositional patterns|Naturally captures compositional structure|
|**Scalability**|Less scalable to very complex problems|Scales better to complex problems with appropriate architecture|
|**Interpretability**|Potentially more interpretable|Individual layers may be less interpretable, but can reveal hierarchical concepts|

### Vanishing and Exploding Gradients

As networks get deeper, gradients can either vanish (become too small) or explode (become too large) during backpropagation, making training difficult. Modern solutions include:

- **Skip connections** (as in ResNets): Allow gradients to flow directly through the network
- **Batch normalization**: Normalizes activations to prevent extreme values
- **Careful initialization**: Strategies like He and Xavier initialization help maintain appropriate gradient magnitudes
- **Alternative activation functions**: ReLU and its variants help maintain gradient flow

### Computational Efficiency

Deep networks require more sequential processing, which can limit parallelization. Architectures like Transformers address this by enabling more parallel computation while maintaining the benefits of depth.

### When Shallow Networks Might Be Preferable

Despite the advantages of deep networks, shallow networks are sometimes preferable:

- **Simple problems**: When the underlying function is simple, adding depth adds unnecessary complexity
- **Small datasets**: Deep networks may overfit on small datasets where shallow networks generalize better
- **Computational constraints**: When inference speed or memory is strictly limited
- **Interpretability requirements**: When understanding the model's decision process is critical

The power of depth in neural networks stems from the ability to learn hierarchical representations that efficiently capture complex patterns in data. While shallow networks have their place in machine learning, the success of deep learning has demonstrated that for many complex real-world problems, depth provides a fundamental advantage in representational capacity, parameter efficiency, and generalization ability.

The optimal architecture—whether deep or shallow—ultimately depends on the specific problem, available data, and computational constraints. Modern neural network design often involves finding the right balance between depth, width, and specialized architectural components to match the task at hand.

## Building Blocks of Modern Deep Learning

While the concepts above form the foundation, modern deep learning incorporates several additional components:

- **Convolutional Neural Networks (CNNs)**: Specialized for grid-like data such as images
- **Recurrent Neural Networks (RNNs)**: Process sequential data with memory of previous inputs
- **Transformers**: Handle sequential data using attention mechanisms
- **Regularization techniques**: Dropout, batch normalization, and weight decay to prevent overfitting
- **Transfer learning**: Leveraging pre-trained models on new tasks

These advanced architectures and techniques have enabled deep learning to achieve remarkable performance across diverse domains, pushing the boundaries of artificial intelligence.