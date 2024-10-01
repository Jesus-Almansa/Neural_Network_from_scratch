# Backpropagation

## Index

1. [Introduction](#introduction)
2. [Backpropagation Overview](#backpropagation-overview)
   - [Forward Pass](#forward-pass)
   - [Backward Pass](#backward-pass)
   - [Chain Rule of Calculus](#chain-rule-of-calculus)
3. [The `Value` Class](#the-value-class)
   - [Attributes](#attributes)
   - [Methods](#methods)
     - [`__add__` Method](#add-method)
     - [`__mul__` Method](#mul-method)
     - [`tanh()` Method](#tanh-method)
     - [`backward()` Method](#backward-method)
4. [Example of Backpropagation](#example-of-backpropagation)
5. [Example Notebook](#example-notebook)
6. [Conclusion](#conclusion)



## Introduction

Backpropagation is a key algorithm in machine learning, especially in the context of training deep neural networks. It allows for efficient computation of gradients of the loss function with respect to the parameters of the model using the chain rule. These gradients are used by optimization algorithms like gradient descent to update the model parameters, helping the model minimize the loss function.

In this `README`, we will explore how backpropagation works and explain the custom implementation of backpropagation through the `Value` class.

## Backpropagation Overview

Backpropagation involves two key phases:
1. **Forward Pass**: In the forward pass, the input data is passed through the network, and computations are performed layer by layer to generate the output (prediction).
2. **Backward Pass**: After the forward pass, the backward pass computes the gradients of the loss function with respect to the model's parameters. This is done by applying the chain rule of calculus to propagate errors from the output layer back to each parameter, allowing us to adjust them accordingly.

### Chain Rule of Calculus

The chain rule is essential to backpropagation, as it provides a way to compute the derivative of composite functions. If we have a function `f(g(x))`, the chain rule tells us that the derivative of `f` with respect to `x` is:

\[
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
\]

This principle is applied in backpropagation to compute the gradients as we move from the output layer back to the input.

---

## The `Value` Class

The `Value` class provided below is a simple custom implementation that simulates backpropagation for scalar values. It supports operations like addition, multiplication, and hyperbolic tangent (`tanh`), and it tracks how values are computed to enable backpropagation.

```python
import math

def f(x):
    return 3*x**2 - 4*x + 5

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out 
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out 
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for v in reversed(topo):
            v._backward()
```

### Key Components of the `Value` Class

1. **Attributes**:
   - `data`: The scalar value stored in the `Value` object.
   - `grad`: The gradient associated with this `Value`. This will be used during backpropagation.
   - `_backward`: A function that computes the local gradients and accumulates them during backpropagation.
   - `_prev`: A set of `Value` objects that are parents (i.e., used in the computation of this `Value`).
   - `_op`: The operation (`+`, `*`, `tanh`, etc.) that was used to create this `Value`.
   - `label`: A string to label the `Value` (useful for visualization).

2. **`__add__` Method**:
   - This method overloads the `+` operator, allowing for addition of two `Value` objects. 
   - It also defines the `_backward()` function, which computes how the gradients propagate through an addition operation during backpropagation.

3. **`__mul__` Method**:
   - This method overloads the `*` operator for multiplying two `Value` objects.
   - It defines the `_backward()` function to handle the propagation of gradients through a multiplication operation.

4. **`tanh()` Method**:
   - This method implements the hyperbolic tangent (`tanh`) function. The `tanh` function is commonly used in neural networks as an activation function.
   - It also defines the `_backward()` function for computing gradients through the `tanh` operation.

5. **`backward()` Method**:
   - This is the core of backpropagation. It starts from the output node and propagates the gradients backward through the computational graph.
   - It builds the topological order of the computational graph using a recursive depth-first search (`build()` function) and then applies backpropagation in reverse order.

### Example of Backpropagation

Let's create an example of forward and backward passes using this class:

```python
a = Value(-2.0, label='a')
b = Value(3.0, label='b')

# Forward pass
d = a * b
d.label = 'd'
e = a + b
e.label = 'e'
f = d * e
f.label = 'f'

# Backward pass
f.backward()

# Print the gradients
print(a.grad)  # Gradient of 'a'
print(b.grad)  # Gradient of 'b'
```

In this example:
- We perform some arithmetic operations (`*` and `+`) with `a` and `b` to compute `f`.
- After computing the forward pass, we call `f.backward()` to compute the gradients of `a` and `b` with respect to `f`.
- The gradients are stored in `a.grad` and `b.grad`.

```markdown
## Example Notebook

You can find an example of how backpropagation works in this [notebook](notebooks/Backpropagation.ipynb).
```

## Conclusion

This implementation of backpropagation using a custom `Value` class illustrates the core principles of how deep learning frameworks like PyTorch and TensorFlow compute gradients. By defining the forward operations and manually implementing the chain rule in the `_backward()` functions, you gain a deeper understanding of how these frameworks work under the hood.

This class provides a simple yet powerful mechanism to experiment with backpropagation on scalar values, and it can be extended to handle more complex functions and multi-dimensional data.

