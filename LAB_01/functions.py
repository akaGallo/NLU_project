import nltk, spacy
from collections import Counter

def get_data(corpus):
    # Download necessary resources
    nltk.download('gutenberg')
    nltk.download('punkt')

    # Get characters and words from the Gutenberg corpus
    chars = nltk.corpus.gutenberg.raw(corpus)
    words = nltk.corpus.gutenberg.words(corpus)
    
    return chars, words

def load_spacy_model(chars):
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(chars, disable = ["tagger", "ner", "lemmatizer"])
    return doc

def get_statistics(corpus, chars, toolkit):
    print("\n" + "="*35, toolkit, "DESCRIPTIVE STATISTICS", "="*35)

    if toolkit == "REFERENCE":
        words = nltk.corpus.gutenberg.words(corpus)
        sents = nltk.corpus.gutenberg.sents(corpus)

    elif toolkit == "NLTK":
        words = nltk.word_tokenize(chars)
        sents = nltk.sent_tokenize(chars)

    elif toolkit == "SPACY":
        doc = load_spacy_model(chars)
        words = [token for token in doc]
        sents = [sent for sent in doc.sents]

    else:
        print("Invalid toolkit choice.")

    compute_statistics(chars, words, sents)

def compute_statistics(corpus_chars, corpus_words, corpus_sents):
    # Compute various statistics
    total_chars = len(corpus_chars)
    total_words = len(corpus_words)
    total_sents = len(corpus_sents)

    # Calculate word and sentence lengths
    word_length = [len(word) for word in corpus_words]
    sent_length = [len(sent) for sent in corpus_sents]
    chars_in_sents = [sum(len(word) for word in sent) for sent in corpus_sents]

    # Calculate min, max, and average statistics
    min_token_length = min(word_length)
    max_token_length = max(word_length)
    avg_token_length = round(sum(word_length) / len(corpus_words))

    min_words_per_sentence = min(sent_length)
    max_words_per_sentence = max(sent_length)
    avg_words_per_sentence = round(sum(sent_length) / len(corpus_sents))
    
    min_char_per_sentence = min(chars_in_sents)
    max_char_per_sentence = max(chars_in_sents)
    avg_char_per_sentence = round(sum(chars_in_sents) / len(corpus_sents))

    print(f"\nTotal chars: {total_chars}\nTotal words: {total_words}\nTotal sents: {total_sents}")
    print(f"\nMinimum chars per word: {min_token_length}\nMaximum chars per word: {max_token_length}\nAverage chars per word: {avg_token_length}")
    print(f"\nMinimum words per sentence: {min_words_per_sentence}\nMaximum words per sentence: {max_words_per_sentence}\nAverage words per sentence: {avg_words_per_sentence}")
    print(f"\nMinimum chars per sentence: {min_char_per_sentence}\nMaximum chars per sentence: {max_char_per_sentence}\nAverage chars per sentence: {avg_char_per_sentence}")

def compute_lowercased_lexicon(chars, words, doc):
    # Compute lowercase lexicons for different toolkits
    reference_lexicon = set([w.lower() for w in words])
    nltk_lexicon = set([w.lower() for w in nltk.word_tokenize(chars)])
    spacy_lexicon = set([w.lower_ for w in doc])

    print("\n" + "="*40, "LOWERCASED LEXICONS", "="*40)
    print("\nREFERENCE Lexicon Size:\t", len(reference_lexicon))
    print("NLTK Lexicon Size:\t", len(nltk_lexicon))
    print("SPACY Lexicon Size:\t", len(spacy_lexicon))

def compute_frequency_distribution(chars, words, doc, N):
    # Compute frequency distributions for lowercase lexicons
    reference_freq_list = Counter([w.lower() for w in words])
    nltk_freq_dist = Counter([w.lower() for w in nltk.word_tokenize(chars)])
    spacy_freq_dist = Counter([w.lower_ for w in doc])

    # Define a function to get the top N items from a frequency distribution
    def nbest(d, N = 1):
        return dict(sorted(d.items(), key = lambda item: item[1], reverse = True)[:N])

    print("\n" + "="*28, "FREQUENCY DISTRIBUTION (LOWERCASED LEXICON)", "="*28)
    print("\nTop", N, "REFERENCE Frequency Distribution:\t", nbest(reference_freq_list, N))
    print("Top", N, "NLTK Frequency Distribution:\t", nbest(nltk_freq_dist, N))
    print("Top", N, "SPACY Frequency Distribution:\t", nbest(spacy_freq_dist, N), "\n")
