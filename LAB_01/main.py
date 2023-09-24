from functions import *

if __name__ == "__main__":
    corpus = 'milton-paradise.txt'
    chars, words = get_data(corpus)
    doc = load_spacy_model(chars)

    # List of toolkits to use for statistics
    toolkits = ["REFERENCE", "NLTK", "SPACY"]

    for toolkit in toolkits:
        get_statistics(corpus, chars, toolkit)

    compute_lowercased_lexicon(chars, words, doc)
    compute_frequency_distribution(chars, words, doc, N = 5)