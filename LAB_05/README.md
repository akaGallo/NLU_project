# LAB 5 - Constituency Grammars with NLTK

## Objectives
Understanding:
- relation between grammar and syntactic parse tree
- relation between grammar and syntactic categories
- relation between grammar and Part-of-Speech tags
- context free grammars (CFG)
- probabilistic context free grammars (PCFG)

Learning how to:
- define CFG in NLTK
- parse with CFG
- learn PCFGs from a treebank
- parse with PCFG
- generate sentences using a grammar in NLTK
- evaluate parser

## Lab Exercise
- Write two or more sentences of your choice.
- Write a PCFG that models your sentences.
- To validate your grammar, parse the sentences with a parser of your choice.
- Then, generate 10 sentences using a PCFG by experimenting with `nltk.parse.generate.generate` using different starting symbols and depths. Optionally, generate 10 sentences with `PCFG.generate()`.

## Prerequisites
Before using the script, make sure you have the following dependencies installed:
- **NLTK**: NLTK (Natural Language Toolkit) is a powerful library for working with human language data.
- **PCFG**: PCFG stands for "Probabilistic Context-Free Grammar," a formalism used in linguistics and NLP for modeling the syntax of natural language.

You can install the necessary dependencies using the following commands:
```bash
pip install nltk
pip install pcfg
```

## Usage
1. Ensure you have all the necessary dependencies installed.
2. Run the `main.py` script using the following command. This script imports functions from the `functions.py` module to generate and validate sentences using NLTK and PCFG.
```bash
python main.py
```
After running the script, you will see outputs demonstrating the generated sentences and parse trees on a personal example using both NLTK and PCFG methods.
