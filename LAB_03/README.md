# LAB 3 - Statistical Language Modeling with NLTK

## Objectives
Understanding:
- Ngrams and Ngram counting
- Vvcabulary and basic usage

Learning how to:
- training language model
- using Ngram language models

## Lab Exercise
Write your own implementation of the Stupid backoff algorithm. Train it and compare the perplexity with the one provided by NLKT. The dataset that you have to use is the Shakespeare Macbeth.
- Stupid Backoff algorithm (use ⍺ = 0.4): [Large Language Models in Machine Translation](https://aclanthology.org/D07-1090.pdf)
- NLTK (StupidBackoff): [NLTK algorithm documentation](https://www.nltk.org/api/nltk.lm.html)

Suggestion: adapt the `compute_ppl` function to compute the perplexity of your model. The PPL has to be computed on the whole corpus and not at sentence level.

## Prerequisites
Ensure you have Python 3.6 or higher installed on your machine. Install the necessary dependency:
```bash
pip install nltk
```

## Algorithms in Focus
### NLTK StupidBackoff
The NLTK StupidBackoff algorithm is harnessed through the `StupidBackoff` class within the `nltk.lm module`. This method employs a straightforward backoff strategy to estimate probabilities for unseen n-grams.

### Custom MyStupidBackoff
The MyStupidBackoff algorithm presents a tailor-made interpretation of NLTK's `MLE` class. Integrating the principles of Stupid Backoff, it introduces the parameter `alpha` for smoother predictions. The critical method `unmasked_score` calculates scores based on the Stupid Backoff approach.


## Usage
1. The functions.py file contains the necessary functions to preprocess a given dataset, build vocabulary, implement the NLTK StupidBackoff and MyStupidBackoff algorithms, and compute perplexity for evaluation.
2. To run the code:
```bash
python main.py
```
