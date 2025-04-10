Images are a very important kind of data on which we desire to train neural networks. They have three particular properties that justify the need for a specialized architecture:
1. They are **highly-dimensional**, for instance even a relatively small image of 256x256 colored pixels has 196,608 input dimensions, which for a fully connected network would make the _number of weights enormous_.
2. There is a **strong spatial correlation** between pixels. Fully connected networks have no notion of "nearby" and treat the relationship between every input equally.
3. The interpretation of an image is **stable under geometric transformations**. Clearly an image of a cat remains an image of a cat even if rotated or shifted. But for a network even a small shift would completely change the inputs, thus the learning phase would require examples for every possible transformation

### Translation Invariance

For a task to be _translational invariant_ it means that the output remains the same for inputs that are obtained from translating one another.

For example a translational invariant classification task would classify both these translated images as "mountain".

CNNs are _not_ translation invariant out of the box, but we aim to make them as much as possible using techniques such as pooling.

![[cnn-trans-inv.png|center|200]]

### Translation Equivariance

For a task to be _translational equivariant_ it means that for every input $i$ and its output $o$, $f(i) = o$, whatever translation applies to the input is reflected in the output: $f(t(i)) = t(f(i))$.

Consider this segmentation task in which every pixel in the image is classified according to the object it belongs to:

![[cnn-trans-equiv.png|center]]

Every layer in CNNs is translational equivariant.

### CNNs

_Convolutional Neural Networks_ are specialized networks for processing grid-like data structures, such as time-series data (1D), **images**(2D), videos (3D); in which there are strong correlations between close elements, but little or no correlation between far elements.

Their architecture is inspired by the frontal visual cortex of the brain and mathematically it is based on the **convolution** operation (hence the name).

The idea is to replace the classical matrix multiplication with convolution, while maintaining the same other aspects.

#### Convolution Operator

Given two functions $f, g \in \mathbb{R} \to \mathbb{R}$, the convolution operation ($*$) is defined as (it is commutative):
$$
\begin{align}
(f * g)(t) & = \int_{-\infty}^{+\infty}f(t-a)g(a) \, da \\
& = \int_{-\infty}^{+\infty}g(t-a)f(a) \, da \\
\end{align}
$$

We will mainly focused on discrete signals:
For _discrete time signals_ $w, u$:
$$
(w * u)(t) = \sum_{a=-\infty }^{+\infty}w(t-a) u(a)
$$

Since usually the filters are non-zero only on a finite set of points:
$$
(w * u)(t) = \sum_{a=0 }^{m}w(t-a) u(a)
$$

In images, given a **kernel** $\boldsymbol{K}$ and an image $\boldsymbol{I}$:
$$
(\boldsymbol{K} * \boldsymbol{I})(i, j) = \sum_{a = -m}^{m}\sum_{b = -n}^{n}K(a, b) I(i-a, j-b)
$$

![[cnn-convolution.gif|center]]

##### Cross-correlation

In practice the _cross-correlation_ operator is used, which is identical to the convolution, except that the "sliding" function does not get flipped (i.e. the minus sign becomes a plus).

For 2D images:
$$
(\boldsymbol{K} * \boldsymbol{I})(i, j) = \sum_{a = -m}^{m}\sum_{b = -n}^{n}K(a, b) I(i+a, j+b)
$$

So intuitively we can think of "sliding" a kernel over the image, performing a point-wise matrix multiplication between the kernel and an equally-sized section of the image, then summing the elements of the result together.


![[cnn-convolution-example-2.png]]
![[cnn-convolution-example.gif]]

In the following sections we will use convolution even if the underlying operation is cross-correlation. They are practically interchangeable in this field, but cross-correlation is more natural to implement and visualize.

#### Sparse Interactions and Parameter Sharing

One of the main problem mentioned in the introduction was the absurd number of weights required.

Since now we are applying a sliding kernel over the input image, the inputs are connected only to $n$ neurons in the hidder layer (with $n$ being the size of the kernel).

Additionally we will (almost always) keep the weights of the kernel constant throughout the convolution over the input image, thus weights are shared, in fact there will be only $n$ weights per convolution layer.

This dramatically reduces the number of weights to learn during training, and dramatically increases efficiency.
Consider this example with:
- Input image: 320x280
- Kernel size: 2x1
- Output size: 319x280 (because of TODO: link padding)

| | Convolution | Dense layer | Sparse Matrix (Conv. without shared weights) |
| - | - | - | - |
| Stored parameters | 2 | 319x280x320x280 > 8e9 | 2x319x280 = 178,640 |

By adding more convolutional layers we further enlarge the _receptive field_ of the deeper layers' neurons (i.e. the amount of pixels that contribute to its output).


![[cnn-sparse-interactions.png]]
![[cnn-parameter-sharing.png]]

#### Strided and Dilated Convolution

The definition we gave up until here makes so that the output is _as big as_ the input, this happens because the kernel is shifted by one place at a time, and slid across the entire image.

Intuitively we can imagine to move the kernel by more than one step at a time. This would reduce the size of the resulting feature map compared to the original map (downsampling). This technique is called **strided convolution**.
The main advantages are:
- The feature map becomes smaller, thus **improved efficiency**.
- The receptive layer of the following layer is **larger** without using a larger kernel.

The number of places that the kernel shift to in a step is called _stride_. The resulting size of the feature map is $\left\lfloor  \frac{n}{s}  \right\rfloor$, with $n$ the original image size and $s$ the stride.

In the NN the effect of the stride is to reduce the number of neurons (circa $s$ times) in the convolutional layer.

Another modification is to "sparsify" the kernel matrix, with the goal of **enlarging** the receptive field without requiring more weigths and operations.
This technique is called **dilated convolution**, in which kernel values are spaced with zeros.

The effect on the NN is that some input neurons are "skipped" in the connection to the neuron the the convolutional layer, as shown in the image.

![[cnn-stride-dilation.png]]