import nltk, spacy
import pandas as pd
from conll import evaluate
from sklearn_crfsuite import CRF
from nltk.corpus import conll2002
from spacy.tokenizer import Tokenizer

def get_data():
    nltk.download('conll2002')

    nlp = spacy.load("es_core_news_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)

    train_sents = conll2002.iob_sents("esp.train")
    test_sents = conll2002.iob_sents("esp.testa")

    return nlp, train_sents, test_sents

def get_algorithm():
    crf = CRF(algorithm = 'lbfgs', c1 = 0.1, c2 = 0.1, max_iterations = 100, all_possible_transitions = True)
    return crf

# Function to extract labels from a sentence
def sent2labels(sent):
    return [label for _, _, label in sent]

# Function to extract tokens from a sentence
def sent2tokens(sent):
    return [token for token, _, _ in sent]

# Function to extract parts of speech from a sentence
def sent2pos(sent):
    return [pos for _, pos, _ in sent]

# Function to extract features for each token in a sentence using spaCy
def sent2spacy_features(nlp, sent, suffix = False, conll_tutorial = False, range_of_one = False, range_of_two = False):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    pos = sent2pos(sent)
    feats = []
    for index, (token, pos) in enumerate(zip(spacy_sent, pos)):
        # Create different token features based on specified options and context around the current token
        if suffix:
            token_feats = {
                'bias': 1.0,
                'word.lower()': token.lower_,
                'pos': token.pos_,
                'lemma': token.lemma_,   
                'suffix': token.suffix_,
            }
        elif conll_tutorial:
            token_feats = {
                'bias': 1.0,
                'word.lower()': token.lower_,
                'word[-3:]': str(token)[-3:],
                'word[-2:]': str(token)[-2:],
                'word.isupper()': token.is_upper,
                'word.istitle()': token.is_title,
                'word.isdigit()': token.is_digit,
                'postag': pos,
                'postag[:2]': pos[:2],     
            }
        else:
            token_feats = {
                'bias': 1.0,
                'word.lower()': token.lower_,
                'pos': token.pos_,
                'lemma': token.lemma_,   
            }

        if range_of_one or range_of_two:
            if index > 0:
                token_feats.update({
                    '-1:word.lower()': spacy_sent[index - 1].lower_,
                    '-1:word.istitle()': spacy_sent[index - 1].is_title,
                    '-1:word.isupper()': spacy_sent[index - 1].is_upper,
                    '-1:postag': spacy_sent[index - 1].pos_,
                    '-1:postag[:2]': spacy_sent[index - 1].pos_[:2],
                })
            else:
                token_feats['BOS'] = True
            
            if index < len(spacy_sent) - 1:
                token_feats.update({
                    '+1:word.lower()': spacy_sent[index + 1].lower_,
                    '+1:word.istitle()': spacy_sent[index + 1].is_title,
                    '+1:word.isupper()': spacy_sent[index + 1].is_upper,
                    '+1:postag': spacy_sent[index + 1].pos_,
                    '+1:postag[:2]': spacy_sent[index + 1].pos_[:2],
                })
            else:
                token_feats['EOS'] = True

        if range_of_two:
            if index > 1:
                token_feats.update({
                    '-2:word.lower()': spacy_sent[index - 2].lower_,
                    '-2:word.istitle()': spacy_sent[index - 2].is_title,
                    '-2:word.isupper()': spacy_sent[index - 2].is_upper,
                    '-2:postag': spacy_sent[index - 2].pos_,
                    '-2:postag[:2]': spacy_sent[index - 2].pos_[:2],
                })
            else:
                token_feats['BOS'] = True
            
            if index < len(spacy_sent) - 2:
                token_feats.update({
                    '+2:word.lower()': spacy_sent[index + 2].lower_,
                    '+2:word.istitle()': spacy_sent[index + 2].is_title,
                    '+2:word.isupper()': spacy_sent[index + 2].is_upper,
                    '+2:postag': spacy_sent[index + 2].pos_,
                    '+2:postag[:2]': spacy_sent[index + 2].pos_[:2],
                })
            else:
                token_feats['EOS'] = True

        feats.append(token_feats)
        
    return feats

def train_and_evaluate(crf, train_feats, train_label, test_feats, test_sents):
    try:
        crf.fit(train_feats, train_label)
    except AttributeError:
        pass

    predicted = crf.predict(test_feats)
    hypothesis = [[(test_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(predicted)]
    
    results = evaluate(test_sents, hypothesis)
    pd_table = pd.DataFrame().from_dict(results, orient = 'index')
    pd_table.round(decimals = 3)
    print(pd_table)
    print()

def extract_features(nlp, train_sents, test_sents, suffix = False, conll_tutorial = False, range_of_one = False, range_of_two = False):
    train_feats = [sent2spacy_features(nlp, s, suffix, conll_tutorial, range_of_one, range_of_two) for s in train_sents]
    train_label = [sent2labels(s) for s in train_sents]
    test_feats = [sent2spacy_features(nlp, s, suffix, conll_tutorial, range_of_one, range_of_two) for s in test_sents]

    return train_feats, train_label, test_feats

def shallow_parsing_model(features, nlp, crf, train_sents, test_sents):
    if features == "suffix":
        print("\n" + "="*12, "SUFFIX FEATURES", "="*12 + "\n")
        train_feats, train_label, test_feats = extract_features(nlp, train_sents, test_sents, suffix = True)
    
    elif features == "conll_tutorial":
        print("\n" + "="*8, "CONLL TUTORIAL FEATURES", "="*8 + "\n")
        train_feats, train_label, test_feats = extract_features(nlp, train_sents, test_sents, conll_tutorial = True)
    
    elif features == "range_of_one":
        print("\n" + "="*11, "[-1, +1] FEATURES", "="*11 + "\n")
        train_feats, train_label, test_feats = extract_features(nlp, train_sents, test_sents, range_of_one = True)
    
    elif features == "range_of_two":
        print("\n" + "="*11, "[-2, +2] FEATURES", "="*11 + "\n")
        train_feats, train_label, test_feats = extract_features(nlp, train_sents, test_sents, range_of_two = True)
    
    else:
        print("\n" + "="*11, "BASELINE FEATURES", "="*11 + "\n")
        train_feats, train_label, test_feats = extract_features(nlp, train_sents, test_sents)

    train_and_evaluate(crf, train_feats, train_label, test_feats, test_sents)