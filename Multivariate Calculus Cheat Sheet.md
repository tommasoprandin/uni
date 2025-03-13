## Partial Derivatives

### Definition and Notation

If $z = f(x,y)$ then the partial derivatives are defined to be: $$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h,y) - f(x,y)}{h}$$ $$\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x,y+h) - f(x,y)}{h}$$

If $z = f(x,y)$ then all of the following are equivalent notations for partial derivatives: $$\frac{\partial f}{\partial x} = f_x = \frac{\partial z}{\partial x} = \frac{\partial}{\partial x}f(x,y) = D_x f(x,y)$$ $$\frac{\partial f}{\partial y} = f_y = \frac{\partial z}{\partial y} = \frac{\partial}{\partial y}f(x,y) = D_y f(x,y)$$

For a function of $n$ variables $f(x_1, x_2, \ldots, x_n)$, the partial derivative with respect to $x_i$ is: $$\frac{\partial f}{\partial x_i} = f_{x_i} = \frac{\partial}{\partial x_i}f(x_1, x_2, \ldots, x_n)$$

### Interpretation of Partial Derivatives

If $z = f(x,y)$ then:

1. $f_x(a,b)$ is the slope of the tangent line to the curve formed by the intersection of the surface $z = f(x,y)$ and the plane $y = b$ at point $(a,b,f(a,b))$.
2. $f_y(a,b)$ is the instantaneous rate of change of $f$ with respect to $y$ at $(a,b)$.

## Basic Properties and Formulas

If $f(x,y)$ and $g(x,y)$ are differentiable functions, $c$ and $n$ are any real numbers:

1. $\frac{\partial}{\partial x}(cf) = c\frac{\partial f}{\partial x}$
2. $\frac{\partial}{\partial x}(f \pm g) = \frac{\partial f}{\partial x} \pm \frac{\partial g}{\partial x}$
3. $\frac{\partial}{\partial x}(fg) = f\frac{\partial g}{\partial x} + g\frac{\partial f}{\partial x}$ – Product Rule
4. $\frac{\partial}{\partial x}\left(\frac{f}{g}\right) = \frac{g\frac{\partial f}{\partial x} - f\frac{\partial g}{\partial x}}{g^2}$ – Quotient Rule
5. $\frac{\partial}{\partial x}(c) = 0$
6. $\frac{\partial}{\partial x}(x^n) = nx^{n-1}$, $\frac{\partial}{\partial y}(x^n) = 0$ – Power Rule

## Chain Rule for Multivariate Functions

For a composite function $F(x,y) = f(g(x,y), h(x,y))$:

$$\frac{\partial F}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial g}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial h}{\partial x}$$ $$\frac{\partial F}{\partial y} = \frac{\partial f}{\partial u}\frac{\partial g}{\partial y} + \frac{\partial f}{\partial v}\frac{\partial h}{\partial y}$$

Where $u = g(x,y)$ and $v = h(x,y)$.

### Chain Rule Special Cases

1. $\frac{d}{dt}f(x(t), y(t)) = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$
2. $\frac{d}{dt}[f(g(t))]^n = n[f(g(t))]^{n-1}f'(g(t))g'(t)$
3. $\frac{d}{dt}e^{f(g(t))} = f'(g(t))g'(t)e^{f(g(t))}$

## Common Multivariable Derivatives

- $\frac{\partial}{\partial x}(x^ny^m) = nx^{n-1}y^m$
- $\frac{\partial}{\partial y}(x^ny^m) = mx^ny^{m-1}$
- $\frac{\partial}{\partial x}(e^{xy}) = ye^{xy}$
- $\frac{\partial}{\partial y}(e^{xy}) = xe^{xy}$
- $\frac{\partial}{\partial x}(\ln(xy)) = \frac{1}{x}$
- $\frac{\partial}{\partial y}(\ln(xy)) = \frac{1}{y}$

## Higher Order Partial Derivatives

Second-order partial derivatives: $$\frac{\partial^2 f}{\partial x^2} = f_{xx} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$$ $$\frac{\partial^2 f}{\partial y \partial x} = f_{yx} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$$ $$\frac{\partial^2 f}{\partial x \partial y} = f_{xy} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)$$ $$\frac{\partial^2 f}{\partial y^2} = f_{yy} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right)$$

If $f$ has continuous second-order partial derivatives, then $f_{xy} = f_{yx}$ (Clairaut's theorem).

## Directional Derivatives

The directional derivative of $f(x,y,z)$ in the direction of a unit vector $\mathbf{u} = (u_1, u_2, u_3)$ is: $$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}u_1 + \frac{\partial f}{\partial y}u_2 + \frac{\partial f}{\partial z}u_3$$

## Gradient, Divergence, Curl and Laplacian

### Gradient

For a scalar function $f(x,y,z)$: $$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right)$$

Properties:

- $\nabla(f + g) = \nabla f + \nabla g$
- $\nabla(fg) = f\nabla g + g\nabla f$
- The gradient points in the direction of maximum rate of increase of $f$
- The magnitude of the gradient is the maximum rate of change

### Divergence

For a vector field $\mathbf{F} = (F_1, F_2, F_3)$: $$\nabla \cdot \mathbf{F} = \frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z}$$

### Curl

For a vector field $\mathbf{F} = (F_1, F_2, F_3)$: $$\nabla \times \mathbf{F} = \left(\frac{\partial F_3}{\partial y} - \frac{\partial F_2}{\partial z}, \frac{\partial F_1}{\partial z} - \frac{\partial F_3}{\partial x}, \frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y}\right)$$

### Laplacian

For a scalar function $f(x,y,z)$: $$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

## Implicit Differentiation

For a function $F(x,y) = 0$, the derivative $\frac{dy}{dx}$ can be found by: $$\frac{dy}{dx} = -\frac{F_x}{F_y} = -\frac{\partial F/\partial x}{\partial F/\partial y}$$

## Critical Points and Optimization

### Critical Points

$(a,b)$ is a critical point of $f(x,y)$ if $\nabla f(a,b) = (0,0)$, meaning: $$\frac{\partial f}{\partial x}(a,b) = 0 \text{ and } \frac{\partial f}{\partial y}(a,b) = 0$$

### Second Derivative Test

At a critical point $(a,b)$, let: $$D = f_{xx}(a,b)f_{yy}(a,b) - [f_{xy}(a,b)]^2$$

1. If $D > 0$ and $f_{xx}(a,b) > 0$, then $f$ has a local minimum at $(a,b)$
2. If $D > 0$ and $f_{xx}(a,b) < 0$, then $f$ has a local maximum at $(a,b)$
3. If $D < 0$, then $(a,b)$ is a saddle point
4. If $D = 0$, the test is inconclusive

## Multiple Integrals

### Double Integrals

$$\iint_R f(x,y) ,dA = \int_a^b \int_c^d f(x,y) ,dy ,dx$$

In polar coordinates: $$\iint_R f(r,\theta) ,dA = \int_{\alpha}^{\beta} \int_{r_1(\theta)}^{r_2(\theta)} f(r,\theta) \cdot r ,dr ,d\theta$$

### Triple Integrals

$$\iiint_E f(x,y,z) ,dV = \int_a^b \int_c^d \int_p^q f(x,y,z) ,dz ,dy ,dx$$

In spherical coordinates $(r,\theta,\phi)$: $$\iiint_E f(r,\theta,\phi) ,dV = \int_{\alpha}^{\beta} \int_{\gamma}^{\delta} \int_{r_1}^{r_2} f(r,\theta,\phi) \cdot r^2\sin\phi ,dr ,d\phi ,d\theta$$

In cylindrical coordinates $(r,\theta,z)$: $$\iiint_E f(r,\theta,z) ,dV = \int_{\alpha}^{\beta} \int_{r_1}^{r_2} \int_{z_1}^{z_2} f(r,\theta,z) \cdot r ,dz ,dr ,d\theta$$

## Line and Surface Integrals

### Line Integrals

Line integral of a scalar function $f(x,y,z)$ along curve $C$ parameterized by $\mathbf{r}(t), a \leq t \leq b$: $$\int_C f ,ds = \int_a^b f(\mathbf{r}(t)) |\mathbf{r}'(t)| ,dt$$

Line integral of a vector field $\mathbf{F}$ along curve $C$: $$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) ,dt$$

### Surface Integrals

Surface integral of a scalar function $f(x,y,z)$ over surface $S$: $$\iint_S f ,dS = \iint_D f(x,y,z(x,y)) \sqrt{1 + \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} ,dA$$

Surface integral of a vector field $\mathbf{F}$ (flux) across surface $S$ with unit normal $\mathbf{n}$: $$\iint_S \mathbf{F} \cdot \mathbf{n} ,dS$$

## Fundamental Theorems

### Fundamental Theorem of Line Integrals

If $\mathbf{F} = \nabla f$ is a conservative vector field and $C$ is a curve from point $A$ to point $B$: $$\int_C \mathbf{F} \cdot d\mathbf{r} = f(B) - f(A)$$

### Green's Theorem

For a region $D$ with boundary curve $C$ (positively oriented): $$\oint_C P ,dx + Q ,dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) ,dA$$

### Stokes' Theorem

For a surface $S$ with boundary curve $C$: $$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} ,dS$$

### Divergence Theorem (Gauss's Theorem)

For a solid region $E$ with boundary surface $S$ (positively oriented): $$\iint_S \mathbf{F} \cdot \mathbf{n} ,dS = \iiint_E (\nabla \cdot \mathbf{F}) ,dV$$