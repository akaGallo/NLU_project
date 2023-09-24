from functions import *

if __name__ == "__main__":
    # Fetch training and testing data, and create a spaCy pipeline and a CRF model
    nlp, train_sents, test_sents = get_data()
    crf = get_algorithm()
    
    # Iterate over each feature option and run the shallow parsing model
    features = ["baseline", "suffix", "conll_tutorial", "range_of_one", "range_of_two"]
    for feature in features:
        shallow_parsing_model(feature, nlp, crf, train_sents, test_sents)