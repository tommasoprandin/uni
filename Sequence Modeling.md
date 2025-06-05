## Learning in Sequential Domains

Why learning in sequential domains is different than static domains? Because successive points in sequential data are **strongly correlated**. Machine learning models and algorithms for sequence learning have to consider that data points are not independent, deal with sequential distortions and/or variations (e.g. In speech, variations in speaking rate) and make use of **contextual information**.

With static data we usually learn: $$P(\mathbf{o}|\mathbf{x})$$ where $\mathbf{x}$ is a fixed-size tuple of predictive attributes and $\mathbf{o}$ is a classification/regression task.

With **sequential data**, instead, $\mathbf{x}$ is a **sequence** $x^{(1)}, ..., x^{(t)}, ...$ where each $x^{(t)}$ has a static type. $\mathbf{o}$ may be either static (e.g., sequence classification) or a sequence.

Using mathematical induction, a **sequence** is either an external vertex, or an ordered pair $(t, h)$ where the head $h$ is a vertex and the tail $t$ is a sequence.

![[dl-seq-exa.png]]

### Sequential Transductions

Sequence Transduction is a machine learning task that involves converting an input sequence into an output sequence, potentially of different lengths.

Let $X$ and $O$ be the **input** and **output** label spaces. We denote by $X^*$ the set of all sequences with labels in $X$. $X^*$ is basically the set of all possible input sequences using elements from $X$. We can define a general transduction $T$ as a function $$T : X^* \rightarrow O^*$$

- $T(\cdot)$ has **finite memory** $k \in \mathbb{N}$ if $\forall \mathbf{x} \in X^*$ and $\forall t$, $T(x^{(t)})$ only depends on ${\mathbf{x}^{(t)}, \mathbf{x}^{(t-1)}, ..., \mathbf{x}^{(t-k)}}$ The output at time $t$ depends only on a fixed number $k$ of past inputs.
- $T(\cdot)$ is **algebraic** if it has 0 finite memory (i.e., no memory at all)
- A transduction $T(\cdot)$ is **causal** if the output at time $t$ does not depend on future inputs (at time $t + 1, t + 2,...$ )

### Learning Sequences

Sequences have variable length but typical machine learning models have a fixed number of inputs. In order to solve this problem we can:

- Limit context to a **fixed-size window**.
- Use **recurrent** models. (causality)
- Use **transformers** for non-causal sequences (e.g. text).

Note that causal means that the model can only see the past and not the future. This is important because we want to predict the future based on the past. For example, in a speech recognition task, we want to predict the next word based on the previous words, not the other way around. More precisely: A transduction $T$ is **causal** if the output at time $t$ does not depend on future inputs (at time $t + 1, t + 2,...$ ).

## Recursive State Representation

In order to represent a recursive state we can use the following equations: $$\begin{align} h^{(t)} & = f(h^{(t-1)}, x^{(t)}, t) \\
 o^{(t)} & = g(h^{(t)}, x^{(t)}, t) \end{align}$$

where $f$ is the _state transition function_ and $g$ is the _output function_. The function $f$ updates the system's memory. It combines the past hidden state $h^{(t-1)}$—which holds information from previous time steps—with the current input $x^{(t)}$ to produce the new hidden state $h^{(t)}$. This allows the model to build a memory over time. In other words the state transition fuction tells how the model "evolves" over time given the previous state and the new inputs.

$h^{(t)}$ is called the state of the system and it's defined by a **recursive equation**. Indeed, the definition of $h$ at time $t$ refers back to the same definition at time $t - 1$. It contains information about the whole past sequence. For a finite number of time steps $\tau$, the graph can be unfolded by applying the definition $\tau - 1$ times. **Unfolding** the equation by repeatedly applying the definition yields an expression that does not involve recurrence. Such an expression can now be represented by a traditional directed acyclic computational graph.

The state transition function can be represented using the time shift operator $q^{-1}$: $$q^{-1}h^{(t)} = h^{(t-1)}$$

![[graph-dnn-recursive-state.png]]

The unfolding process thus introduces two major advantages:

1. Regardless of the sequence length, the learned model always has the same input size, because it is specified in terms of transition from one state to another state, rather than specified in terms of a variable-length history of states.
    
2. It is possible to use the same transition function $f$ with the same parameters at every time step.
    

These two factors make it possible to learn a single model that operates on all time steps and all sequence lengths. Learning a single, shared model allows generalization to sequence lengths that did not appear in the training set, and allows the model to be estimated with far fewer training examples than would be required without parameter sharing.

Given a sequence $s \in X^{*}$ and a recursive transduction $T$, the _encoding network_ associated to $s$ and $T$ is formed by unrolling (time unfolding) the recursive network of $T$ through the input sequence $s$.

![[dl-seq-recform.png]]
![[dl-seq-transductions.png]]

$T$ is **stationary** (**time-invariant**) if $f(\cdot)$ and $g(\cdot)$ do not depend on $t$.

There are different ways in which we can implement $f(\cdot)$ and $g(\cdot)$. There are two general families of models:

- Linear:
    - Kalman Filter
    - Hidden Markov Models
    - Linear Dynamical Systems
    - ...
- Nonlinear
    - Recurrent Neural Networks
    - ...

Non-linear techniques are much more powerful than the linear counterparts, because they can represent a much wider class of systems.

## Shallow Recurrent Neural Networks

Armed with the graph unrolling and parameter sharing ideas, we can design a wide variety of recurrent neural networks. In general we have: $$\begin{align} \mathbf{h}^{(t)} & = f(\mathbf{U}\mathbf{x}^{(t)} + \mathbf{W}\mathbf{h}^{(t-1)} + \mathbf{b}) \\
 \mathbf{o}^{(t)} & = g(\mathbf{V}\mathbf{h}^{(t)} + \mathbf{c}) \end{align}$$

where $f()$ and $g()$ are non-linear functions (e.g. $tanh()$ and $softmax$), and $h^{(0)} = 0$ (or can be learned jointly with the other parameters). $\mathbf{U}$ and $\mathbf{W}$ are weight matrices which parametrize **input-to-hidden**connections and **hidden-to-hidden** recurrent connections respectively. **Hidden-to-output** connections are parametrized by the weight matrix $\mathbf{V}$.

An example of RNN for IO-transduction with discrete outputs: $$\begin{align}
 & \mathbf{h}^{(t)} = tanh(\mathbf{U}\mathbf{x}^{(t)} + \mathbf{W}\mathbf{h}^{(t-1)} + \mathbf{b}) \\
  & \mathbf{o}^{(t)} = \mathbf{V}\mathbf{h}^{(t)} + \mathbf{c} \\
 & \hat{\mathbf{y}} = softmax(\mathbf{o}^{(t)}) \\
 &  L = \sum_t L^{(t)} = - \sum_t log,p_{model}(\mathbf{y}^{(t)} | {\mathbf{x}^{1}, ..., \mathbf{x}^{(t)}}) \end{align}$$

where:

- $o^{(t)}$ is the unnormalized log probabilities at time $t$
- $\mathbf{y}^{(t)}$ is the target vector at time $t$
- $p_{model}(\mathbf{y}^{(t)} | {\mathbf{x}^{1}, ..., \mathbf{x}^{(t)}}$ is given by reading the entry for $\mathbf{y}^{(t)}$ from the model's output vector $\hat{\mathbf{y}}^{(t)}$, that is, the loss $L$ **internally computes** $\hat{\mathbf{y}}$.
- $L$ is the loss function

The corresponding computation graph is the following:

![[dl-rnn-compgraph.png]]

Some examples of important design patterns for recurrent neural networks include the following:

- Recurrent networks that produce an output at each time step and have recurrent connections between hidden units (IO-transduction).
- Recurrent networks that produce an output at each time step and have recurrent connections only from the output at one time step to the hidden units at the next time step.
- Recurrent networks with recurrent connections between hidden units, that read an entire sequence and then produce a single output (e.g. for classification).

There are a lot of possible additional architectural features, such as short-cut connections, higher-order states, feedback from output, teacher forcing, bidirectional RNN, etc[^1]. All these architectural features (and others...) are orthogonal, i.e. they can be combined together.

[^1]: See slides for further information

### Teacher Forcing

The network with recurrent connections only from the output at one time step to the hidden units at the next time step is strictly less powerful because it lacks hidden-to-hidden recurrent connections. Therefore, it requires that the output units capture all of the information about the past that the network will use to predict the future. Because the output units are explicitly trained to match the training set targets, they are unlikely to capture the necessary information about the past history of the input.

For this reason, models that have recurrent connections from their outputs leading back into the model may be trained with **teacher forcing**. Teacher forcing is a procedure in which during training the model receives the ground truth output $\mathbf{y}^{(t)}$ as input at time $t + 1$. When the model is deployed, the true output is generally not known. In this case, we approximate the correct output $\mathbf{y}^{(t)}$ with the model's output $\mathbf{o}^{(t)}$, and feed the output back into the model. $$\begin{align}
 & \mathbf{h}^{(t)}  = tanh(\mathbf{U}\mathbf{x}^{(t)} + \mathbf{W}\mathbf{y}^{(t-1)} + \mathbf{b}) \\
  & \mathbf{o}^{(t)} = \mathbf{V}\mathbf{h}^{(t)} + \mathbf{c} \\
 &  \hat{\mathbf{y}} = softmax(\mathbf{o}^{(t)}) \\
 &  L = - \sum_t log,p_{model}(\mathbf{y}^{(t)} | \mathbf{y}^{(1)}, ..., \mathbf{y}^{(t-1)}, \mathbf{x}^{(1)}, ..., \mathbf{x}^{(t)}) \\
\end{align}$$

![[dl-rnn-teacherforcing.png]]

The advantage of eliminating hidden-to-hidden recurrence is that, for any loss function based on comparing the prediction at time $t$ to the training target at time $t$, all the time steps are decoupled. Training can thus be parallelized, with the gradient for each step $t$ computed in isolation.

### Bidirectional RNNs

In many applications we want to output a prediction of $\mathbf{y}^{(t)}$ which may depend on the whole input sequence. For example, if there are two interpretations of the current word that are both plausible, we may have to look far into the future (and the past) to disambiguate them. Bidirectional recurrent neural networks (or bidirectional RNNs) were invented to address that need.

As the name suggests, bidirectional RNNs combine an RNN that moves forward through time, beginning from the start of the sequence, with another RNN that moves backward through time, beginning from the end of the sequence.

![[dl-rnn-bidi.png]]

### 1 to _n_ transduction

Previously, we have discussed RNNs that take a sequence of vectors $\mathbf{x}^{(t)}$ for $t = 1, ..., \tau$ as input. Another option is to take only a single vector $\mathbf{x}$ as input. When $\mathbf{x}$ is a fixed-size vector, we can simply make it an extra input of the RNN that generates the $\mathbf{y}$ sequence. The interaction between the input $\mathbf{x}$ and each hidden unit vector $\mathbf{h}^{(t)}$ is parametrized by a newly introduced weight matrix $\mathbf{R}$.

![[graph-dnn-1ntrans.png]]

Each element $\mathbf{y}^{(t)}$ of the observed output sequence serves both as input (for the current hidden unit at time $t$) and, during training, as target (for the previous output unit at time $t-1$).

This RNN is appropriate for tasks such as image captioning, where a single image is used as input to a model that then produces a sequence of words describing the image.

### Encoder-Decoder Sequence-to-Sequence Architectures

Here we discuss how an RNN can be trained to map an input sequence to an output sequence which is not necessarily of the same length. This comes up in many applications, such as speech recognition, machine translation, etc.

An encoder-decoder RNN architecture is composed of an encoder RNN that reads the input sequence and a decoder RNN that generates the output sequence. The final hidden state of the encoder RNN is used to compute a generally fixed-size context variable $C$ which represents a semantic summary of the input sequence and is given as input to the decoder RNN.

![[dl-enc-dec.png]]

If the context $C$ is a vector, then the decoder RNN is simply a vector-to-sequence RNN.