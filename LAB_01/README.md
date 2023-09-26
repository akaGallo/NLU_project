# LAB 1 - Corpus and Lexicon

## Objectives
Understanding:
- relation between corpus and lexicon
- effects of pre-processing (tokenization) on lexicon
  
Learning how to:
- load basic corpora for processing
- compute basic descriptive statistic of a corpus
- building lexicon and frequency lists from a corpus
- perform basic lexicon operations
- perform basic text pre-processing (tokenization and sentence segmentation) using python libraries

## Lab Exercise
* Load a corpus from Gutenberg (e.g. milton-paradise.txt).
* Compute descriptive statistics on the reference (.raw, .words, etc.)sentences and tokens.
* Compute descriptive statistics in the automatically processed corpus, both with spacy and nltk.
* Compute lowercased lexicons for all 3 versions (reference, spacy, nltk) of the corpus, compare lexicon sizes.
* Compute frequency distribution for all 3 versions (reference, spacy, nltk) of the corpus, compare top N frequencies.

## Prerequisites
Before using the script, make sure you have the following dependencies installed:
- **NLTK**: NLTK (Natural Language Toolkit) is a powerful library for working with human language data.
- **SpaCy**: SpaCy is a library for advanced natural language processing in Python.
- **en_core_web_sm**: English language model en_core_web_sm is a small-sized model suitable for various natural language processing tasks.
You can install the necessary dependencies using the following commands:
```bash
pip install nltk
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage
1. Place the text corpus you want to analyze in the same directory as `main.py`. For demonstration purposes, the script is currently set to analyze a text file named `milton-paradise.txt`.
2. Run the `main.py` script using the following command:
```bash
python main.py
```
