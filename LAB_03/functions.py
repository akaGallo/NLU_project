import nltk
from itertools import chain
from nltk.corpus import gutenberg
from nltk.lm import MLE, Vocabulary, StupidBackoff
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline

def get_dataset(dataset, ngram_order):
    nltk.download('gutenberg')
    nltk.download('punkt')
    nltk.download('stopwords')

    # Preprocess the sentences by converting to lowercase
    dataset_sents = [[w.lower() for w in sent] for sent in gutenberg.sents(dataset)]
    dataset_words = list(flatten(dataset_sents))
    
    # Split the dataset into training and test data
    train_size = int(0.8 * len(dataset_words))
    test_data = list(nltk.ngrams(dataset_words[train_size:], ngram_order))

    return dataset_words, dataset_sents, test_data

def get_vocabulary(words, sents, n_order):
    vocab = Vocabulary(words, unk_cutoff = 2)
    dataset_oov_sents = [list(vocab.lookup(sent)) for sent in sents]
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(n_order, dataset_oov_sents)
    return vocab, padded_ngrams_oov, flat_text_oov

def compute_perplexity(ngram_order, vocab, padded_ngrams_oov, flat_text_oov, test_data, my_algorithm = False):
    if my_algorithm:
        print("="*10, "MY STUPIDBACKOFF ALGORITHM", "="*10 + "\n")
        stupid_backoff = MyStupidBackoff(ngram_order, vocab)
    else:
        print("\n" + "="*10, "NLTK STUPIDBACKOFF ALGORITHM", "="*10 + "\n")
        stupid_backoff = StupidBackoff(order = ngram_order, vocabulary = vocab)

    # Fit the language model
    stupid_backoff.fit(padded_ngrams_oov, flat_text_oov)
    
    # Prepare ngrams for computing perplexity
    ngrams, _ = padded_everygram_pipeline(stupid_backoff.order, [vocab.lookup(sent) for sent in test_data])
    ngrams = chain.from_iterable(ngrams)

    ppl = stupid_backoff.perplexity([x for x in ngrams if len(x) == stupid_backoff.order])
    print("PPL:", ppl, "\n")

class MyStupidBackoff(MLE):
    def __init__(self, order, vocab, alpha = 0.4):
        super().__init__(order = order, vocabulary = vocab)
        self.alpha = alpha
        
    # Override the unmasked_score method to implement the Stupid Backoff algorithm
    def unmasked_score(self, word, context = None):
        if context is None or len(context) == 0:
            score = self.counts[word] / len(self.vocab)
        else:
            score = super().unmasked_score(word, context)
            
            if score == 0:
                score = self.alpha * self.unmasked_score(word, context[1:])
        return score