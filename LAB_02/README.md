# LAB 2 - Experimental Methodology in Natural Language Processing

## Objectives
Understanding:
- the role and types of evaluation in NLP/ML
- the lower and upper bounds of performance
- correct usage of data for experimentation
- evaluation metrics

Learning how to use scikit-learn to perform a text classification experiment:
- provided baselines
- text vectorization
- evaluation methods

## Lab Exercise
* Using Newsgroup dataset from scikit-learn train and evaluate Linear SVM (LinearSVC) model.
* Experiment with different vectorization methods and parameters, experiment_id in parentheses (e.g. CounVector, CutOff, etc.):
  - binary of Count Vectorization (CountVect)
  - TF-IDF Transformation (TF-IDF)
  - Using TF-IDF: min and max cut-offs (CutOff)
  - Using TF-IDF: wihtout stop-words (WithoutStopWords)
  - Using TF-IDF: without lowercasing (NoLowercase)
  
To print the results: print(experiment_id, the most appropriate score metric to report)). Note: If the SVM doesn't converge play with the  hyperparameter (starting from a low value).


## Prerequisites
Before running the provided scripts, ensure you have the following prerequisites installed:
- scikit-learn: The `fetch_20newsgroups` dataset and SVM classifier are part of scikit-learn, so make sure you have it installed.
```bash
pip install scikit-learn
```

## Usage
1. Ensure you have scikit-learn and other necessary dependencies installed.
2. Run the `main.py` script using the following command. It will load the dataset, define vectorization methods, and display the results of text classification experiments using different vectorization techniques.
```bash
python main.py
```
