from functions import *

if __name__ == "__main__":
    # Task 1: Subjectivity task
    subjectivity_data = load_subjectivity_data(['subj'], ['obj'])
    # train(subjectivity_data, subjectivity_task = True)

    # Task 2.1: Polarity task without objective sentences removal
    polarity_data = load_polarity_data('pos', 'neg')
    # train(polarity_data, polarity_task = True)

    # Task 2.2: Polarity task with objective sentences removal (applying subjectivity model)
    polarity_subjectivity_data = objective_removal(polarity_data)
    # train(polarity_subjectivity_data, subjectivity_task = True, polarity_task = True)

    train(subjectivity_data, subjectivity_task = True, trained = True)
    train(polarity_data, polarity_task = True, trained = True)
    train(polarity_subjectivity_data, subjectivity_task = True, polarity_task = True, trained = True)