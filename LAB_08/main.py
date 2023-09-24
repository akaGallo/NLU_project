from functions import *

if __name__ == "__main__":
    # Step 1: Obtain the classifier and stratified cross-validation strategy.
    classifier, stratified_split = get_classifier_and_cross_val()

    # Step 2: Train and evaluate using the main data.
    data, labels = get_data()
    vectors = train_and_evaluate(classifier, data, labels, stratified_split)

    # Step 3: Train and evaluate using collocational data.
    data, _ = get_data(collocational = True)
    dvectors = train_and_evaluate(classifier, data, labels, stratified_split, collocational = True)

    # Step 4: Concatenate results and evaluate using concatenated vectors.
    data = [vectors, dvectors]
    uvectors = train_and_evaluate(classifier, data, labels, stratified_split, concatenate = True)

    # Step 5: Evaluate word sense disambiguation using different algorithms.
    algorithms = ['ORIGINAL LESK', 'PEDERSEN', "LESK SIMILARITY"]
    data, labels = get_data(stratified=stratified_split, test_split = True)
    for algorithm in algorithms:
        evaluate_word_sense_disambiguation(data, labels, algorithm)
    print()