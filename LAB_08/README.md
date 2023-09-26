# LAB 8 - Word Sense Disambiguation

## Objectives
Understanding:
- Lexical Relations
- Word senses in WordNet
- Semantic Similarity (in WordNet)

Learning how to disambiguate word senses:
- Dictionary-based Word Sense Disambiguation with WordNet
    - Lesk Algorithm
    - Graph-based Methods
- Supervised Word Sense Disambiguation
  - Feature Extractions for Word Sense Classification
    - Bag-of-Words
    - Collocational Features
  - Training and Evaluation

## Lab Exercise
- Extend collocational features with
    - POS-tags
    - Ngrams within window
- Concatenate BOW and new collocational feature vectors & evaluate
- Evaluate Lesk Original and Graph-based (Lesk Similarity or Pedersen) metrics on the same test split and compare

Same test set for all the experiments, you can use K-fold validation.

## Prerequisites
Before using the script, make sure you have the following dependencies installed:
- **NLTK**: NLTK (Natural Language Toolkit) is a powerful library for working with human language data.
- **NumPy**: NumPy provides support for large, multi-dimensional arrays and matrices, along with a variety of high-level mathematical functions to operate on these arrays.
- **scikit-learn**: scikit-learn (sklearn) provides simple and efficient tools for data analysis and modeling, including support for classification, regression, clustering, and more.

You can install the necessary dependencies using the following commands:
```bash
pip install nltk
pip install numpy
pip install scikit-learn
```

## Usage
1. Ensure you have all the necessary dependencies installed.
2. Run the `main.py` script using the following command. The script will load the necessary data and preprocess it, train and evaluate a Multinomial Naive Bayes classifier using Bag-of-Words features, train and evaluate the classifier using collocational features, concatenate the results of Bag-of-Words and collocational features, and evaluate.
```bash
python main.py
```
Evaluate Word Sense Disambiguation using different algorithms: Original Lesk, Pedersen and Lesk Similarity.
The script will output the evaluation results, including precision, recall, F1-score, and accuracy for each sense of the ambiguous word.
