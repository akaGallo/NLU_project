import nltk
from nltk import Nonterminal
from nltk.parse.generate import generate
from pcfg import PCFG

def get_sentences_and_grammar():
    # Example sentences
    my_sentences = [
        "I study the interesting NLU exam of the AIS course",
        "this funny exercise is about constituency grammars with NLTK",
        "the world is yours",
    ]

    # Weighted production rules for the grammar
    my_weighted_rules = [
        'S -> NP VP [1.0]',
        'NP -> PRON [0.1] | NP PP [0.2] | Det NP [0.3] | ADJ NP [0.2] | N [0.1] | N N [0.1]',
        'VP -> V NP [0.6] | V PP [0.4]',
        'PP -> P NP [1.0]',
        'PRON -> "I" [0.6] | "yours" [0.4]',
        'N -> "NLU" [0.1] | "exam" [0.1] | "AIS" [0.1] | "course" [0.1] | "exercise" [0.1] | "constituency" [0.1] | "grammars" [0.2] | "NLTK" [0.1] | "world" [0.1]',
        'Det -> "the" [0.6] | "this" [0.4]',
        'P -> "of" [0.5] | "about" [0.2] | "with" [0.3]',
        'ADJ -> "funny" [0.5] | "interesting" [0.5]',
        'V -> "study" [0.3] | "is" [0.7]',
    ]

    print("\n" + "="*62, "SENTENCES", "="*62 + "\n")
    print(" - ".join(my_sentences) + "\n")
    
    # Create an NLTK PCFG grammar
    nltk_grammar = PCFG.fromstring(my_weighted_rules)
    return my_sentences, nltk_grammar

def validation_grammar(sentences, grammar):
    parser = nltk.InsideChartParser(grammar)
    print("="*61, "PARSE TREES", "="*61 + "\n")
    for sentence in sentences:
        for tree in parser.parse(sentence.split()):
            print(tree.pretty_print(unicodelines = True, nodedist = 6), "\n")

def generate_NLTK_sentences(nltk_grammar):
    print("="*45, "GENERATE SEQUENCES WITHOUT PROBABILITIES", "="*45 + "\n")
    for sentence in generate(nltk_grammar, start = Nonterminal('S'), depth = 5, n = 15):
        print(" ".join(sentence))

def generate_PCFG_sentences(pcfg_grammar):
    print("\n" + "="*51, "GENERATE SEQUENCES WITH PCFG", "="*51 + "\n")
    for sent in pcfg_grammar.generate(10):
        print(sent)
    print()