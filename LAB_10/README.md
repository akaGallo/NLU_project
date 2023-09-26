# LAB 10 - Sequence Labelling and Text classification tasks

## Objectives
Understanding:
- Sequence Labelling
- Text classification
- Lang class
- Hugging Face

Learning how to:
- Convert words to numbers (word2id)
- Define a neural network in Pytorch

## Lab Exercise
### Part 1
Modify the baseline architecture Model IAS in an addition way:
- Add bidirectionality
- Add dropout layer

### Part 2
Fine-tune BERT model using a multi-task learning setting on intent classification and slot filling. You can refer to this paper to have a better understanding of such model: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Prerequisites
Before you start, make sure you have the following requirements fulfilled:
- **torch**: PyTorch is a powerful deep learning framework for building and training neural networks.
- **numpy**: NumPy is a fundamental library for numerical computations in Python, enabling array operations and linear algebra.
- **tqdm**: tqdm is a library for creating progress bars in command-line interfaces, making it easier to track the progress of tasks.
- **scikit-learn**: sklearn provides tools for data analysis and modeling.
- **Transformers**: transformers. developed by Hugging Face, is a popular open-source library for NLP and deep learning tasks, particularly focused on state-of-the-art pre-trained models for a wide range of NLP applications. It provides an easy-to-use interface to access and utilize pre-trained transformer-based models, including BERT, GPT-2, RoBERTa, and many others.

You can install the necessary dependencies using the following commands:
```bash
pip install torch
pip install numpy
pip install tqdm
pip install scikit-learn
pip install transformers
```

## Dataset
For completing the exercises we have used the [ATIS](https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data) dataset (Airline Travel Information Systems), that is composed of trascriptions of humans asking about flight information. ATIS is a well-known benchmark dataset in the field of natural language processing and specifically in the area of spoken language understanding (SLU).

## Usage
Ensure you have all the necessary dependencies installed.

### Part 1
Inside this folder, you will find the code related to Part 1, where the Model IAS architecture has been modified by adding bidirectionality and a dropout layer.
Run the code and observe the resulting output that demonstrates the changes made to the baseline architecture.
1. Enter in the folder `part_1`.
2. Run the `main.py` script.
```bash
cd part_1
python main.py
```
NOTE: if the `trained = True` option has been considered in the main, the output will provide the `Slot F1` and `Intent Accuracy` values for each of the best three models saved in the `/bin` folder, otherwise the training process will start again from the beginning for each model.

### Part 2
Inside this folder, you will find the code related to Part 2, where the BERT model has been fine-tuned using a multi-task learning setting for intent classification and slot filling.
Run the code and observe the resulting output. You might see the training results of the model on both intent classification and slot filling tasks.

The provided model is based on the reference for customizing BERT within the ModelIAS framework in the [Github](https://github.com/monologg/JointBERT/blob/master/model/modeling_jointbert.py) repository related to the [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909) paper.

1. Enter in the folder `part_2`.
2. Run the `main.py` script.
```bash
cd part_2
python main.py
```
NOTE: if the `trained = True` option has been considered in the main, the output will provide the `Slot F1` and `Intent Accuracy` values for each of the best three models saved in the `/bin` folder, otherwise the training process will start again from the beginning for each model.
