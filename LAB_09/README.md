# LAB 9 - Vector Semantics

## Objectives
Understanding:
- different methods of representing words as vectors
- vectors and similarity between vectors
- evaluation of word embeddings

Learning how to:
- train word embeddings with gensim
- use pre-trained word embeddings for similarity computation

## Lab Exercise
### Part 1
Modify the baseline LM_RNN (the idea is to add a set of improvements and see how these affect the performance). Furthremore, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration. Here are the links to the state-of-the-art papers which uses vanilla RNN ([Extension of Recurrent Neural Network language model](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5947611&tag=1) and [Recurrent neural network based language model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)).
- Replace RNN with LSTM (output the PPL)
- Add two dropout layers: (output the PPL)
  - one on embeddings
  - one on the output
- Replace SGD with AdamW (output the PPL)

### Part 2
Add to best model of Part 1 the following regularizations described in [Regularizing and optimizing LSTM language models](https://openreview.net/pdf?id=SyyGPP0TZ) paper:
- Weight Tying (PPL)
- Variational Dropout (PPL)
- Non-monotonically Triggered AvSGD (PPL)

## Prerequisites
Before running the provided scripts, ensure you have the following prerequisites installed:
- **torch**: PyTorch is a powerful deep learning framework for building and training neural networks.
- **numpy**: NumPy is a fundamental library for numerical computations in Python, enabling array operations and linear algebra.
- **tqdm**: tqdm is a library for creating progress bars in command-line interfaces, making it easier to track the progress of tasks.
- **matplotlib**: provides a wide range of tools for creating various types of plots, graphs, charts, and other data visualizations

You can install the necessary dependencies using the following commands:
```bash
pip install torch
pip install numpy
pip install tqdm
pip install matplotlib
```

## Dataset
For completing the exercises we have used the [Penn Treebank](https://paperswithcode.com/dataset/penn-treebank) dataset, a widely used corpus in natural language processing (NLP) and computational linguistics. It consists of text from various sources, primarily newspaper articles, annotated with part-of-speech (POS) tags and syntactic tree structures. The dataset is commonly used for tasks such as language modeling, POS tagging, and syntactic parsing.

## Usage
Ensure you have all the necessary dependencies installed.

### Part 1
To tackle Part 1, you are required to make several modifications to the baseline LM_RNN model. The goal is to observe how these improvements affect the model's performance in terms of perplexity (PPL).
- Replacing RNN with LSTM: replace the vanilla RNN with an LSTM (Long Short-Term Memory) layer. This should be done in the model architecture.
- Adding Dropout Layers: introduce two dropout layers for regularization. Apply dropout to the embeddings and the output of the LSTM.
- Replacing SGD with AdamW: substitute the SGD optimizer with the AdamW optimizer. Adjust hyperparameters such as learning rate, weight decay, and epsilon.

After implementing these changes, you should train the modified model and evaluate it on your dataset. Keep track of the perplexity achieved with this configuration.
1. Enter in the folder `part_1`.
2. Run the `main.py` script.
```bash
cd part_1
python main.py
```
NOTE: if the `trained = True` option has been considered in the main, the output will provide the PPL values for each of the best three models saved in the `/bin` folder, otherwise the training process will start again from the beginning for each model.

### Part 2
For Part 2, you are required to enhance the best model obtained from Part 1, by adding three specific regularizations described in a paper. These regularizations are:
- Weight Tying: implement weight tying, which involves using the same weight matrix for both the input and output embeddings.
- Variational Dropout: introduce variational dropout, a form of dropout that accounts for uncertainty in the dropout rate by using a parameterized distribution.
- Non-monotonically Triggered AvSGD: apply the Non-monotonically Triggered Averaged Stochastic Gradient Descent technique to improve optimization.

Implement these regularizations in the model architecture, adapt your training script accordingly, and train the model again. Monitor the model's performance using perplexity. The above-mentioned regularization techniques are sourced from the official [Github](https://github.com/ahmetumutdurmus/awd-lstm) repository corresponding to the paper titled "[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)".

1. Enter in the folder `part_2`.
2. Run the `main.py` script.
```bash
cd part_2
python main.py
```
NOTE: if the `trained = True` option has been considered in the main, the output will provide the PPL values for each of the best three models saved in the `/bin` folder, otherwise the training process will start again from the beginning for each model.
