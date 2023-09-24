# LAB 4 - Sequence Labeling: Part-of-Speech Tagging

## Objectives
Understanding:
- relation between Classification and Sequence Labeling
- relation between Ngram Modeling and Sequence Labeling
- general setting for Sequence Labeling
- Markov Model Tagging
- Universal Part-of-Speech Tags

Learning how to use scikit-learn to perform a text classification experiment:
- perform POS-tagging using NLTK
- perform POS-tagging using spacy
- train and test (evaluate) POS-tagger with NLTK

## Lab Exercise
- Train and evaluate NgramTagger
  - experiment with different tagger parameters
  - some of them have cut-off
- Evaluate spacy POS-tags on the same test set
  - create mapping from spacy to NLTK POS-tags
    - [SPACY list](https://universaldependencies.org/u/pos/index.html)
    - [NLTK list](https://github.com/slavpetrov/universal-pos-tags)
  - convert output to the required format (see format above)
  - flatten into a list
  - evaluate using [accuracy](https://www.nltk.org/_modules/nltk/metrics/scores.html#accuracy) from nltk.metrics

Dataset: treebank

## Prerequisites
Make sure you have the required libraries installed. You can install them using the following commands:
```bash
pip install nltk
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage
1. Ensure you have all the necessary dependencies installed.
2. Run the `main.py` script using the following command. It will load the NLTK treebank dataset and split it into training and test data. Then it trains n-gram taggers with different n values and cutoff values using the training data and evaluates accuracy on the test data. Finally, it evaluates POS tags using spaCy and NLTK POS tags.
```bash
python main.py
```
