# Neural Network from Scratch: Manual, Graph-based, and PyTorch Implementations

This repository demonstrates how to build a neural network from scratch with multiple approaches for backpropagation, including:
1. **Manual Backpropagation**: Implementing backpropagation by calculating gradients manually.
2. **Graph-based Neural Networks**: Using a computational graph approach to track operations and perform backpropagation.
3. **PyTorch Approach**: Leveraging PyTorch's autograd and `nn.Module` for a more efficient and scalable implementation.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Manual Backpropagation](#manual-backpropagation)
- [Graph-based Neural Networks](#graph-based-neural-networks)
- [PyTorch Approach](#pytorch-approach)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is designed to provide an in-depth understanding of how neural networks work by building them from the ground up. Starting with manual backpropagation, moving to a graph-based approach for better visualization and debugging, and concluding with PyTorch's native tools, you'll gain insights into different methods of implementing backpropagation and model optimization.

The three key implementations covered in this project:
- **Manual Backpropagation**: Hand-derived gradients for each layer.
- **Graph-based Approach**: Using a computational graph to automatically track operations and calculate gradients.
- **PyTorch**: Building models using PyTorch, with its built-in automatic differentiation and neural network modules.

## Installation

1. Clone the repository and navigate to the project folder.
   
   ```bash
   git clone https://github.com/your_username/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Install the required dependencies.

   ```bash
   pip install -r dependencies.txt
   ```

Dependencies include:
- NumPy
- PyTorch (for the PyTorch approach)

## Project Structure

```plaintext
neural-network-from-scratch/
│
├── Notebooks/
│   └── Backpropagation.ipynb    # Jupyter Notebook with explanations and code examples for backpropagation
│
├── src/
│   ├── __init__.py              # Init file for the src package
│   ├── backpropagation.py       # Implementation of manual backpropagation
│   ├── graph_nn.py              # Graph-based neural network with automatic backpropagation
│   ├── nn.py                    # PyTorch-based neural network implementation
│
├── .gitignore                   # Git ignore file
├── dependencies.txt             # List of dependencies
├── README.md                    # Project documentation (this file)
├── setup.py                     # Setup configuration for packaging
└── your_project_name.egg-info/   # Metadata for the package (automatically generated)
```

### Key Files:
- `backpropagation.py`: Implements neural networks with manual backpropagation.
- `graph_nn.py`: Neural network implementation using a computational graph to track operations and backpropagation automatically.
- `nn.py`: Manual implementation of Neuron, Layer and MLP & PyTorch-based implementation for creating and training neural networks using PyTorch’s `nn.Module`.
- `Backpropagation.ipynb`: Jupyter Notebook with examples and code walkthrough for backpropagation methods.

## Manual Backpropagation

The `backpropagation.py` file implements a fully connected neural network with manual backpropagation. This approach involves:
- Explicitly calculating the gradients for each layer.
- Updating weights using gradient descent.

This method is crucial for understanding how gradients are propagated back through the network.

## Graph-based Neural Networks

The `graph_nn.py` file introduces the concept of a computational graph. Each operation is tracked in the graph, and backpropagation is performed automatically by traversing this graph. This makes debugging easier and helps visualize the flow of gradients across layers.

## PyTorch Approach

The `nn.py` file leverages PyTorch’s `nn.Module` and autograd functionalities to build, train, and evaluate a neural network. PyTorch handles the computation graph and gradient calculations internally, making this the most efficient and scalable approach of the three.

### Key Features:
- Automatic differentiation with autograd.
- Efficient forward and backward passes.
- Support for GPU acceleration (if available).