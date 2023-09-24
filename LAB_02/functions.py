import warnings
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, StratifiedKFold

# Suppressing warnings
warnings.filterwarnings("ignore")

def get_dataset():
    dataset = fetch_20newsgroups(subset = 'all', shuffle = True, random_state = 42)
    return dataset

def define_vectorization_methods():
    vectorization_methods = {
        'CountVect': CountVectorizer(binary = True),
        'TF-IDF': TfidfVectorizer(),
        'CutOff': TfidfVectorizer(min_df = 2, max_df = 0.8),
        'WithoutStopWords': TfidfVectorizer(stop_words = 'english'),
        'NoLowercase': TfidfVectorizer(lowercase = False)
    }
    return vectorization_methods

def get_results(dataset, score_metrics, vectorizers):
    for experiment_id, vectorizer in vectorizers.items():
        x = vectorizer.fit_transform(dataset.data)
        y = dataset.target
        experiment_vectorization_methods(experiment_id, x, y, score_metrics)
    print()

def experiment_vectorization_methods(experiment_id, x, y, score_metrics):
    # Setting up the SVM classifier, using Stratified K-Fold cross-validation
    svm = LinearSVC(C = 0.1, max_iter = 10000)
    stratified_split = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    
    # Cross-validating with the SVM model
    scores = cross_validate(svm, x, y, cv=stratified_split, scoring = score_metrics)
    
    print("\n" + "="*20, "VECTORIZATION METHOD:", experiment_id, "="*20 + "\n")
    for metric in score_metrics:
        print("{}: {:.3}".format(metric, sum(scores['test_' + metric]) / len(scores['test_' + metric])))