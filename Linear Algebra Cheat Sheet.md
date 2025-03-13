### Basic Nomenclature
- **Scalar**: a _scalar_ is a single number (can be rational, natural, real, complex, etc...), it is denoted with italic lowercase: $$
a = 3.14
$$
- **Vector**: a _vector_ is an array of numbers belonging to the same set. We denote it with bold lowercase, eventually adding the upper arrow. $n$ is the length of the vector, and they are laid out as _columns_: $$
\vec{\boldsymbol{x}} \in \mathbb{R}^n = \begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{bmatrix}
$$ They are used to represent the position of a point in an $n$-dimensional space.
- **Matrix**: a _matrix_ is a two-dimensional array (grid) of numbers belonging to the same set. We denote it with bold uppercase letter, while the components are normal uppercase. $$
\boldsymbol{A} = 
\begin{bmatrix}
A_{1, 1} & A_{1, 2} \\
A_{2, 1} & A_{2, 2} \\
A_{3, 1} & A_{3,2}
\end{bmatrix} \in \mathbb{R}^{3\times 2}
$$
- **Tensor**: a _tensor_ is the generalization of a matrix for $n$-dimensions, so a 3-dimensional tensor $\boldsymbol{A} \in \mathbb{R}^{n \times m \times z}$ can be thought as a "cube" of numbers.

### Matrices Definitions and Operations

- **Transposition**: given a matrix $\boldsymbol{A}$, its _transpose_ $\boldsymbol{A}^T$ is the matrix obtained by swapping the rows and columns of $\boldsymbol{A}$ (i.e. mirroring over the diagonal): $$
\boldsymbol{A} = \begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2, 1} & A_{2,2}  \\
a_{3, 1} & A_{3,2} \\
\end{bmatrix}
\implies
\boldsymbol{A}^T = \begin{bmatrix}
A_{1, 1} & A_{2,1} & A_{3, 1}  \\
A_{1, 2} & A_{2, 2} & A_{3, 2}
\end{bmatrix}
\implies
(\boldsymbol{A}^T)_{i, j} = \boldsymbol{A}_{j,i}
$$
- **Symmetric Matrix**: a matrix $\boldsymbol{A}$ is _symmetric_ iff it is identical to its transpose:$$
\boldsymbol{A} = \boldsymbol{A}^T
$$
- **Diagonal Matrix**: a matrix $\boldsymbol{A}$ is _diagonal_ iff all the elements NOT on its diagonal are null: $$
\boldsymbol{A} = \begin{bmatrix}
A_{1, 1} & 0 & 0 & \dots & 0 \\
0 & A_{2,2} & 0 & \dots & 0 \\
0 & 0 & A_{3,3} & \dots & 0  \\
\vdots & \vdots  & \vdots & \ddots & 0 \\
0 & 0 & 0 & 0 & A_{n,n}
\end{bmatrix}
\implies \forall\ i, j\ |\ j \neq j\ A_{i,j} = 0
$$
- **Identity Matrix**: the _identity_ matrix $\boldsymbol{I}_{n}$ is the diagonal square matrix of size $n$, where all the elements on the diagonal are $1$. $$
\boldsymbol{I}_{n} = \begin{bmatrix}
1 & 0 & 0 & \dots & 0 \\
0 & 1 & 0 & \dots & 0 \\
0 & 0 & 1 & \dots & 0 \\
\vdots & \vdots  & \vdots & \ddots & 0 \\
0 & 0 & 0 & \dots & 1
\end{bmatrix} 
\implies \boldsymbol{I}_{i,j} = \begin{cases}
1 &\text{ if } i=j \\
0 &\text{ otherwise}
\end{cases}
$$
- **Sum**: given two matrices $\boldsymbol{A}, \boldsymbol{B} \in \mathbb{R}^{n \times m}$ (so of the same shape), we can define the sum as: $$
\boldsymbol{C} = \boldsymbol{A} + \boldsymbol{B} \text{ where } C_{i, j} = A_{i, j} + B_{i,j} \implies \boldsymbol{C} \in \mathbb{R}^{n \times m}
$$
- **Product**: given two matrices $\boldsymbol{A} \in \mathbb{R}^{n \times m}, \boldsymbol{B} \in \mathbb{R}^{m \times p}$ (so $\boldsymbol{A}$ must have the same number of columns as $\boldsymbol{B}$ has rows), we can define the product as: $$
\boldsymbol{C} = \boldsymbol{A}\boldsymbol{B} \text{ where } C_{i, j} = \sum_{k=1}^{m} A_{i, k}B_{k, j}\\
$$ Example:  $$
\begin{align}
\boldsymbol{C} &= \boldsymbol{A}\boldsymbol{B} = \begin{bmatrix}
A_{1,1} & A_{1, 2} & A_{1, 3} \\
A_{2,1} & A_{2,2} & A_{2,3}
\end{bmatrix} 
\begin{bmatrix}
B_{1,1} & B_{1,2} \\
B_{2,1} & B_{2,2} \\
B_{3,1} & B_{3,2}
\end{bmatrix}\\
&=\begin{bmatrix}
A_{1,1}B_{1,1} + A_{1,2} B_{2, 1} + A_{1,3}B_{3,1} & A_{1,1}B_{1,2} + A_{1,2} B_{2, 2} + A_{1,3}B_{3,2}  \\
A_{2,1}B_{1,1} + A_{2,2} B_{2, 1} + A_{1,3}B_{3,1} & A_{2,1}B_{1,2} + A_{2,2} B_{2, 2} + A_{2,3}B_{3,2}  \\
\end{bmatrix}
\end{align}$$ 
This extends to any matrix size, thus also vectors. Product has the following properties:
	- _Distributive_: $\boldsymbol{A}(\boldsymbol{B} + \boldsymbol{C}) = \boldsymbol{A}\boldsymbol{B} + \boldsymbol{A}\boldsymbol{C}$
	- _Associative_: $\boldsymbol{A}(\boldsymbol{B}\boldsymbol{C}) = (\boldsymbol{A}\boldsymbol{B})\boldsymbol{C}$
	- _Transposition_: $(\boldsymbol{A}\boldsymbol{B})^T = \boldsymbol{B}^T\boldsymbol{A}^T$
	- _Neutral element_: $\boldsymbol{A}\boldsymbol{I} = \boldsymbol{A}$
	- _**NOT** commutative_: $\boldsymbol{A}\boldsymbol{B} \neq \boldsymbol{B}\boldsymbol{A}$ ^4ce54e
- **Power**: given a _square_ matrix $\boldsymbol{A}$, its power is defined as:$$
\boldsymbol{A}^k = \boldsymbol{A}\boldsymbol{A}\dots \boldsymbol{A} \text{ } k \text{ times}
$$
- **Hadamard product**: Given two matrices $\boldsymbol{A}, \boldsymbol{B}$ of the _same shape_, the Hadamard (element-wise) product is the one obtained by multiplying element-by-element the two matrices: $$
\begin{align}
\boldsymbol{C} = \boldsymbol{A} \odot \boldsymbol{B} &= \begin{bmatrix}
A_{1, 1} & A_{1,2} \\
A_{2,1} & A_{2,2}
\end{bmatrix} \odot
\begin{bmatrix}
B_{1, 1} & B_{1,2} \\
B_{2,1} & B_{2,2}
\end{bmatrix} \\
&= \begin{bmatrix}
A_{1,1}B_{1,1} & A_{1,2}B_{1,2} \\
A_{2,1}B_{2,1} & A_{2,2}B_{2,2} 
\end{bmatrix}\\
\\
&\boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C} \in \mathbb{R}^{n \times m}
\end{align}
$$
- **Dot Product**: given two vectors $\vec{\boldsymbol{x}}, \vec{\boldsymbol{y}} \in \mathbb{R}^n$, its _dot product_ is defined as: $$
<\vec{\boldsymbol{x}}, \vec{\boldsymbol{y}}> = \vec{\boldsymbol{x}} \cdot \vec{\boldsymbol{y}} = \boldsymbol{x}^T\boldsymbol{y}
$$ that follows the normal matrix multiplication [[#^4ce54e | here]].
- **Trace**: the _trace_ of a matric $\boldsymbol{A \in \mathbb{R}^{n \times m}}$ is the sum of all its diagonal entries: $$
Tr(\boldsymbol{A}) = \sum_{i} {A_{i, i}}
$$
- **Matrix Inversion**: given a _square_ matrix $\boldsymbol{A} \in \mathbb{R}^{n \times n}$, where all its columns are _linearly independent_, its _inverse_ is a matrix $\boldsymbol{A}^{-1}$, for which: $$
\boldsymbol{A}^{-1}\boldsymbol{A} = \boldsymbol{I}_{n}
$$ Finding the inverse of a matrix is equivalent to finding the solutions for a system of linear equations, represented by the matrix. To be able to find unique solutions there must be $n$ equations for $n$ variables (thus the matrix must be square), and the equations must not be redundant (hence the linear independence of the columns).
- **Orthogonal Matrix**: a _square_ matrix $\boldsymbol{A}$ is _orthogonal_ if: $$
\boldsymbol{A}^T\boldsymbol{A} = \boldsymbol{A}\boldsymbol{A}^T = \boldsymbol{I}a \implies \boldsymbol{A}^{-1} = \boldsymbol{A}^T
$$
- **Orthogonal Vectors**: two vectors $\vec{\boldsymbol{x}}, \vec{\boldsymbol{y}} \in \mathbb{R}^n$ are called _orthogonal_ if their dot product is null: $$
<\vec{\boldsymbol{x}}, \vec{\boldsymbol{y}}> = \boldsymbol{x}^T\boldsymbol{y} = 0
$$

### Linear Dependence, Span and Spaces

Given a set of vectors $\{ \vec{\boldsymbol{x}}_{1}, \dots, \vec{\boldsymbol{x}}_{n} \}$, a _linear combination_ of those is obtained by scaling each vector with some coefficient $c_i$ and then add the result together: $$
\sum_{i=1}^{n}  c_{i}\vec{\boldsymbol{x}}_{i}$$

A set of vectors is said to be _linearly independent_ iff no vector in the set is obtainable by linear combination of the others (i.e. there are no redundant vectors).

The _span_ of a set of vectors is the set of all obtainable vectors by linear combination of the original vectors. It can be said that a set of vector originates a vector space with dimension equal to the number of _linearly independent_ vectors in the set (i.e. not redundant).

Given a matrix $\boldsymbol{A}$ the space originated by linear combination of its columns is called _column space_ of $\boldsymbol{A}$.

A set of $n$ linearly independent vectors is called a _base_ of an $n$-dimensional vector space.

### Norm

The norm of (typically) a vector is used to measure some kind of distance from the origin to the point represented by the vector.

The $L^p$ norm is defined as: $$
||\boldsymbol{x}||_{p} = \left( \sum_{i=1}^{n}{|x_{i}|^p}  \right)^{\frac{1}{p}}$$

For $p = 2$ we have the "classical" euclidean norm (i.e. the length of the vector, or the distance from the origin).

A vector with unitary Euclidean norm: $||\boldsymbol{x}||_{2} = 1$ is called _unit vector_.

We can extend this concept to matrices: $$
||\boldsymbol{A}||_{p} = \left( \sum_{i,j}{|A_{i}|^p}  \right)^{\frac{1}{p}}$$

For $p = 2$ we have the Frobenius norm: $||\boldsymbol{A}||_{F} = \sqrt{ \sum_{i,j}{A_{i,j}^2} }$

### Eigenvectors, Eigenvalues, Eigendecomposition

Given a square matrix $A \in \mathbb{N}^{n\times n}$, an _eigenvector_ $v \in \mathbb{N}^n$ is a vector which direction doesn't change under the linear application of $A$, formally:
$$
\begin{gather}
\boldsymbol{A}\vec{\boldsymbol{v}} = \lambda \vec{\boldsymbol{v}} \\
\iff \\
(\boldsymbol{A} - \lambda \boldsymbol{I})\vec{\boldsymbol{v}} = 0
\end{gather}
$$

The scalar $\lambda$ is called _eigenvalue_.

The equation above has (infinite) solutions $\iff$ 
$$
\det(\boldsymbol{A} - \lambda \boldsymbol{I}) = 0
$$
Then solving for $\lambda$ gives the _eigenvalues_ of $\boldsymbol{A}$.
To finally find the _eigenvectors_ it is sufficient to plug the newly found eigenvalues in the original equation, then solve for $\vec{\boldsymbol{v}}$.

If $A$ is _diagonalizable_, we can represent it as a product of another matrix $V$, where $V$ is a linear map with the eigenvectors of $A$ as columns; and a matrix $\boldsymbol{\lambda}$ formed by the eigenvalues in its diagonal.
$$
\begin{gather}
\boldsymbol{V} = [\vec{\boldsymbol{v}_{1}},\dots, \vec{\boldsymbol{v}_{n}}] \\
 \\
\boldsymbol{\lambda} = \begin{bmatrix}
\lambda_{1} & \dots & 0 \\
0 & \ddots & 0 &  \\
0 & \dots & \lambda_{n} \\
\end{bmatrix} \\
\implies \\
\boldsymbol{A} = \boldsymbol{V}\boldsymbol{\lambda}\boldsymbol{V}^{-1}
\end{gather}
$$

### Hyperplanes

Given a vector $\boldsymbol{w} \in \mathbb{R}^n$, an _hyperplane_ in $\mathbb{R}^n$ is composed by the set of points that satisfy the equation:
$$
\begin{gather}
\boldsymbol{w}^T\boldsymbol{x} = b \\
\equiv \\
w_{1}x_{1} + \dots + w_{n}x_{n} = b
\end{gather}
$$

The hyperplane is _orthogonal_ to the $\boldsymbol{w}$ vector and has an offset equal to $\frac{|b|}{||\boldsymbol{w}||}$.
An hyperplane divides the space of $\mathbb{R}^n$ is two parts called _halfspaces_.
This property is commonly used for classification tasks, where examples are classified by checking in which subspace they are.