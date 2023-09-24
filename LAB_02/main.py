from functions import *

if __name__ == "__main__":
    dataset = get_dataset()

    # Defining a list of score metrics for evaluation
    score_metrics = ['accuracy', 'precision_macro', 'recall_macro', "f1_macro"]
    vectorizers = define_vectorization_methods()

    get_results(dataset, score_metrics, vectorizers)