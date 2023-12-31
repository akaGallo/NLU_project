# LAB 6 - Dependency Grammars with NLTK

## Objectives
Understanding:
- Dependency Relations and Grammars
- Probabilistic Dependency Grammars
- Projective and Non-Projective Parses
- Transition-based Dependency Parsing

Learning how to:
- define dependency grammar in NLTK
- identify a syntactic relation between Head and Dependent
- parse with dependency grammar
- evaluate dependency parser
- use dependency parser of spacy and stanza

## Lab Exercise
- Parse 100 last sentences from dependency treebank using spacy and stanza
    - are the depedency tags of spacy the same of stanza?
- Evaluate the parses using DependencyEvaluator
    - print LAS and UAS for each parser

BUT! To evaluate the parsers, the sentences parsed by spacy and stanza have to be `DependencyGraph` objects. To do this, you have to covert the output of the spacy/stanza to [ConLL](https://universaldependencies.org/format.html) format, from this format extract the columns following the [Malt-Tab](https://cl.lingfil.uu.se/~nivre/research/MaltXML.html) format and finally convert the resulting string into a DependecyGraph. Luckly, there is a library that gets the job done. You have to install the library [spacy_conll](https://github.com/BramVanroy/spacy_conll) and use and adapt to your needs the code that you can find below.

## Prerequisites
Before using the script, make sure you have the following dependencies installed:
- **NLTK**: NLTK (Natural Language Toolkit) is a powerful library for working with human language data.
- **SpaCy**: SpaCy is a library for advanced natural language processing in Python.
- **en_core_web_sm**: English language model en_core_web_sm is a small-sized model suitable for various natural language processing tasks.
- **Stanza**: Stanza is an open-source NLP library designed for a wide range of NLP tasks, including tokenization, part-of-speech tagging, named entity recognition, dependency parsing, and more.
- **Spacy Stanza**: Spacy Stanza wraps the Stanza library, so you can use Stanford's models as a spaCy pipeline
- **Spacy Conll**: Spacy Conll converts the output of the spacy/stanza to ConLL format, then from which extracts the columns following the Malt-Tab format and finally converts the resulting string into a DependecyGraph.

You can install the necessary dependencies using the following commands:
```bash
pip install nltk
pip install spacy
python -m spacy download en_core_web_sm 
pip install stanza
pip install spacy_stanza
pip install spacy_conll
```

## Usage
1. Ensure you have all the necessary dependencies installed.
2. Run the `main.py` script using the following command. The script will output the dependency graphs and parsing performance evaluation for each toolkit. You will see the LAS (Labeled Attachment Score) and UAS (Unlabeled Attachment Score) values, which indicate the accuracy of the dependency parsing.
```bash
python main.py
```
Please note that the NLTK Dependency Treebank dataset contains a limited number of sentences. The script uses the last 100 sentences for comparison. You can adjust this number in the `get_data()` function if needed.
