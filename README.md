# NamingNinja

NamingNinja is an educational implementation of character-level language models for generating new names. This project is inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) project and his [YouTube tutorial](https://youtu.be/TCH_1BHY58I).

## Project Overview

This repository contains implementations of character-level language models that learn to generate new names by analyzing patterns in existing ones. The models learn the statistical patterns of characters in names and can generate new, plausible-sounding names that don't exist in the training data.

### Models Implemented

1. **Bi-gram Model** (`bi_gram/`)
   - A simple model that predicts the next character based only on the current character
   - Uses a probability matrix to capture transitions between pairs of characters
   - Implemented with PyTorch for gradient-based optimization

2. **N-gram Model** (`n_gram/`)
   - A more sophisticated model that considers multiple previous characters (context) to predict the next one
   - Uses character embeddings and a neural network architecture
   - Supports variable context length and embedding dimensions

## How It Works

### Bi-gram Model

The bi-gram model works by:
1. Creating a vocabulary of all characters in the training data
2. Building a matrix of transition probabilities between characters
3. Training this matrix using gradient descent to maximize the likelihood of the observed character transitions
4. Generating new names by sampling from the learned probability distribution

### N-gram Model

The n-gram model extends this concept by:
1. Using a specified number of previous characters as context
2. Embedding each character into a lower-dimensional space
3. Passing these embeddings through a neural network to predict the next character
4. Training with mini-batch gradient descent
5. Generating new names by sampling from the predicted probability distribution

## Dataset

The `names.txt` file contains thousands of real names that serve as the training data. The models learn the patterns in these names to generate new ones. This dataset includes common first names from various origins, providing a diverse set of character patterns for the models to learn from.

## Educational Purpose

This project is my third educational implementation for understanding deep learning and language models, particularly for NLP tasks. It follows my previous projects:
- [S-grad](https://github.com/AbdoAlshoki2/S-grad) - An implementation of a differentiable scalar to track gradients

The goal is to understand the fundamental concepts behind language models that form the basis of more complex systems like GPT and other large language models.

## Usage

To train and generate names using the bi-gram model:

```python
from bi_gram import Bi_gram
import torch

# Load names from file
with open('names.txt', 'r') as f:
    names = f.read().splitlines()

# Create and train the model
model = Bi_gram(names)
model.train(num_iterations=1000)

# Generate new names
new_names = model.predict(num_predictions=10)
print(new_names)
```
To use the n-gram model:
```python
from n_gram import N_gram
import torch

# Load names from file
with open('names.txt', 'r') as f:
    names = f.read().splitlines()

# Create and train the model
model = N_gram(names, previous_context=3, embedding_size=2)
model.train(num_iterations=100, batch_size=32)

# Generate new names
new_names = model.predict(num_predictions=10)
print(new_names)
```