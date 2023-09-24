from functions import *

if __name__ == "__main__":
    # Load training and testing data for the laptop14 dataset
    train_text, train_polarity, _, _ = load_laptop14_data('dataset/laptop14_train.txt')
    test_text, test_polarity, test_gold_ot, test_gold_ts = load_laptop14_data('dataset/laptop14_test.txt')

    # Prepare datasets using texts and polarities from data
    train_dataset = prepare_dataset(train_text, train_polarity)
    test_dataset = prepare_dataset(test_text, test_polarity)
    dataloaders = get_dataloaders(train_dataset, test_dataset)

    # Extract aspect terms and polarities predicted by the model
    # pred_ot, pred_ts = train(dataloaders)
    pred_ot, pred_ts = train(dataloaders, trained = True)

    # Compare aspect terms and polarities predicted with the ones extracted from laptop14_test.txt
    ABSA_evaluation(test_gold_ot, test_gold_ts, pred_ot, pred_ts)