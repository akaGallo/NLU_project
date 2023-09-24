import nltk, en_core_web_sm, math
from itertools import chain
from nltk.corpus import treebank
from nltk.metrics import accuracy
from spacy.tokenizer import Tokenizer
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

def get_dataset():
    nltk.download('treebank')
    nltk.download("universal_tagset")

    return treebank

def split_data(dataset, train_ratio = 0.8):
    total_size = len(dataset.tagged_sents())
    train_index = math.ceil(total_size * train_ratio)
    train_data = dataset.tagged_sents(tagset = 'universal')[:train_index]
    test_data = dataset.tagged_sents(tagset = 'universal')[train_index:]

    return train_data, test_data

def train_and_evaluate_ngram_tagger(train_data, test_data, n, cutoff):
    if n == 1:
        tagger = UnigramTagger(train_data, cutoff = cutoff)
    elif n == 2:
        tagger = BigramTagger(train_data, cutoff = cutoff)
    elif n == 3:
        tagger = TrigramTagger(train_data, cutoff = cutoff)
    else:
        return None
    accuracy = tagger.accuracy(test_data)
    return accuracy

def get_ngram_tagger(train_data, test_data):
    n_values = [1, 2, 3]
    cutoff_values = [0, 1, 2, 3]
    print("\n" + "="*9, "TRAIN & EVALUATE NGRAM TAGGER", "="*9)
    for n in n_values:
        print("\t")
        for cutoff in cutoff_values:
            accuracy = train_and_evaluate_ngram_tagger(train_data, test_data, n, cutoff)
            print(f"NgramTagger (N = {n}, cutoff = {cutoff}) accuracy = ""{:6.4f}".format(accuracy))

# Mapping from spaCy POS tags to NLTK POS tags
mapping_spacy_to_NLTK = {
    "ADJ": "ADJ",
    "ADP": "ADP",
    "ADV": "ADV",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "DET": "DET",
    "INTJ": "X",
    "NOUN": "NOUN",
    "NUM": "NUM",
    "PART": "PRT",
    "PRON": "PRON",
    "PROPN": "NOUN",
    "PUNCT": ".",
    "SCONJ": "CONJ",
    "SYM": "X",
    "VERB": "VERB",
    "X": "X"
}

def evaluate_POS_tags(test_data, mapping = False):
    nlp = en_core_web_sm.load()
    nlp.tokenizer = Tokenizer(nlp.vocab)

    # Sanity check
    for id_sentence, sentence in enumerate(treebank.sents()):
        doc = nlp(" ".join(sentence))
        if len([token.text for token in doc]) != len(sentence):
            print(id_sentence, sentence)

    if not mapping:
        print("\n" + "="*12, "EVALUATE SPACY POS-TAGS", "="*12 + "\n")
    else:
        print("="*12, "EVALUATE NLTK POS-TAGS", "="*13 + "\n")

    # Process test data and calculate accuracy
    data = []
    flatten_test_data = list(chain.from_iterable(test_data))
    for sentence, _ in flatten_test_data:
        doc = nlp(sentence)
        data.append([(token.text, mapping_spacy_to_NLTK[token.pos_]) if mapping
                     else (token.text, token.pos_) for token in doc])
    data = list(chain.from_iterable(data))
    POS_accuracy = accuracy(data, flatten_test_data)
    print("Accuracy: {:6.4f}\n".format(POS_accuracy))