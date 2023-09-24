from functions import *

if __name__ == "__main__":
    dataset = get_dataset()
    train_data, test_data = split_data(dataset)

    # Train an n-gram tagger using the training data and apply it to the test data
    get_ngram_tagger(train_data, test_data)

    evaluate_POS_tags(test_data)
    evaluate_POS_tags(test_data, mapping = True)