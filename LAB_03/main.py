from functions import *

if __name__ == "__main__":
    # Define the dataset file and ngram order
    dataset = 'shakespeare-macbeth.txt'
    ngram_order = 3

    dataset_words, dataset_sents, test_data = get_dataset(dataset, ngram_order)

    # Build vocabulary and preprocess data for NLTK StupidBackoff
    vocab, padded_ngrams_oov, flat_text_oov = get_vocabulary(dataset_words, dataset_sents, ngram_order)
    compute_perplexity(ngram_order, vocab, padded_ngrams_oov, flat_text_oov, test_data)

    # Build vocabulary and preprocess data for custom MyStupidBackoff
    my_vocab, my_padded_ngrams_oov, my_flat_text_oov = get_vocabulary(dataset_words, dataset_sents, ngram_order)
    compute_perplexity(ngram_order, my_vocab, my_padded_ngrams_oov, my_flat_text_oov, test_data, my_algorithm = True)