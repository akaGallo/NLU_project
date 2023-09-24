from functions import *

if __name__ == "__main__":
    sentences, grammar = get_sentences_and_grammar()

    # Validate the provided grammar against the sentences
    validation_grammar(sentences, grammar)

    # Generate sentences using the NLTK framework
    generate_NLTK_sentences(grammar)

    # Generate sentences using the PCFG (Probabilistic Context-Free Grammar) approach
    generate_PCFG_sentences(grammar)
