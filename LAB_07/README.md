# LAB 7 - Sequence Labeling: Shallow Parsing

## Objectives
Understanding:
- relation between Sequence Labeling and Shallow Parsing
- IOB Notation
- Joint Segmentation and Classification
- Feature Engineering

Learning how to:
- use Named Entity Recognition in
    - spacy
    - NLTK
- train, test, and evaluate Conditional Random Fields models
- perform feature engineering with CRF

## Lab Exercise
The exsecise is to experiment with a CRF model on NER task. You have to train and test a CRF with using different features on the `conll2002` corpus. The features that you have to experiment with are:
- Baseline using the fetures in sent2spacy_features
- Add the "suffix" feature
- Add all the features used in the tutorial on CoNLL dataset
- Increase the feature window (number of previous and next token) to [-1, +1]
- Increase the feature window (number of previous and next token) to [-2, +2]

Train the model and print results on the test set. The format of the results has to be the following:
```bash
results = evaluate(tst_sents, hyp)
pd_tbl = pd.DataFrame().from_dict(results, orient = 'index')
pd_tbl.round(decimals = 3)
```

## Prerequisites
Before using the script, make sure you have the following dependencies installed:
- **NLTK**: NLTK (Natural Language Toolkit) is a powerful library for working with human language data.
- **SpaCy**: SpaCy is a library for advanced natural language processing in Python.
- **Pandas**: Pandas is a powerful library for data manipulation and analysis.
- **sklearn_crfsuite**: this library provides tools for implementing Conditional Random Fields (CRFs) in scikit-learn.

You can install the necessary dependencies using the following commands:
```bash
pip install nltk
pip install spacy
pip install pandas
pip install sklearn-crfsuite
```

## Usage
1. Ensure you have all the necessary dependencies installed and the code that provides the following feature options for shallow parsing:
    - Baseline Features
    - Suffix Features
    - CONLL Tutorial Features
    - Range of One Features
    - Range of Two Features
2. Run the `main.py` script using the following command. The script will generate an evaluation table for each feature option, showing the performance of the CRF model for part-of-speech tagging and named entity recognition on the provided test data.
```bash
python main.py
```
