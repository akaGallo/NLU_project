from utils import *
from model import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
SUBJECTIVITY_CLASSES = 2
POLARITY_CLASSES = 3

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.00005
KFOLD = 10
N_EPOCHS = 3

def prepare_dataset(sentences, labels, tokenizer, polarity_task):
    encoded_data = tokenizer(sentences, padding = True, truncation = True, return_tensors = 'pt', return_attention_mask = True)  # Encode input sentences using the provided tokenizer
    label_map = {"Neg": 0, "Neu": 1, "Pos": 2} if polarity_task else {"obj": 0, "subj": 1}
    encoded_labels = torch.tensor([label_map[label[0]] for label in labels], dtype = torch.long)  # Encode labels as tensors based on the label mapping
    dataset = torch.utils.data.TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], encoded_labels)  # Create a TensorDataset combining encoded data and labels
    return dataset

def train_loop(train_loader, criterion_loss, optimizer, model):
    model.train()

    for epoch in range(N_EPOCHS):
        for batch in train_loader:
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits

            # Compute the loss and perform backpropagation
            loss = criterion_loss(logits, labels)
            loss.backward()
            optimizer.step()

def test_loop(test_loader, model):
    model.eval()
    test_predictions = []
    test_labels = []

    for batch in test_loader:
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        labels = batch[2].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask)
        logits = outputs.logits

        # Collect model predictions and true labels
        test_predictions.extend(torch.argmax(logits, dim = 1).tolist())
        test_labels.extend(labels.tolist())

    # Compute accuracy based on the collected predictions and labels
    accuracy = accuracy_score(test_labels, test_predictions)
    return accuracy

def train(data, subjectivity_task = False, polarity_task = False, trained = False):
    sentences, labels = data
    best_accuracy = 0
    best_model = None
    accuracies = []

    if subjectivity_task and polarity_task:
        print("\n" + "="*39, "POLARITY WITH OBJECTIVE REMOVAL TASK", "="*39)
    elif polarity_task:
        print("\n" + "="*51, "POLARITY TASK", "="*50)
    elif subjectivity_task:
        print("\n" + "="*49, "SUBJECTIVITY TASK", "="*48)
    else:
        print("Invalid task choice!")
        return

    if not trained:
        # Initialize the model, loss function, optimizer and cross-validation
        model = PolarityModel(POLARITY_CLASSES).to(DEVICE) if polarity_task else SubjectivityModel(SUBJECTIVITY_CLASSES).to(DEVICE)
        criterion_loss = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
        kfold = StratifiedKFold(n_splits = KFOLD, shuffle = True, random_state = 42)

        print("DATASET SIZE:", len(sentences), "(sentences) -", len(labels), "(labels)\n")
        pbar = tqdm(kfold.split(sentences, labels), desc = "Cross-validation of " + str(KFOLD) + " K-fold")
        for train_idx, test_idx in pbar:
            train_sentences, test_sentences = [sentences[i] for i in train_idx], [sentences[i] for i in test_idx]
            train_labels, test_labels = [labels[i] for i in train_idx], [labels[i] for i in test_idx]

            # Prepare training and test datasets, then create DataLoader
            train_dataset = prepare_dataset(train_sentences, train_labels, TOKENIZER, polarity_task)
            test_dataset = prepare_dataset(test_sentences, test_labels, TOKENIZER, polarity_task)

            train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
            test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE)

            # Train the model and evaluate accuracy on the test set
            train_loop(train_loader, criterion_loss, optimizer, model)

            accuracy = test_loop(test_loader, model)
            accuracies.append(accuracy)
            print("\t-->  Accuracy:", accuracy)

            # Update best model if the current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        if subjectivity_task and polarity_task:
            torch.save(best_model, 'bin/subjectivity_polarity.pt')
        elif polarity_task:
            torch.save(best_model, 'bin/polarity.pt')
        else:
            torch.save(best_model, 'bin/subjectivity.pt')

        # Compute the mean accuracy over cross-validation folds
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f"\nMean Accuracy: {mean_accuracy:.4f}\n")

    else:        
        # Prepare the test dataset, then create a DataLoader
        test_dataset = prepare_dataset(sentences, labels, TOKENIZER, polarity_task)
        test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, shuffle = True)

         # For pre-trained models, load the best model
        if subjectivity_task and polarity_task:
            best_model = torch.load('bin/subjectivity_polarity.pt', map_location = DEVICE)
        elif polarity_task:
            best_model = torch.load('bin/polarity.pt', map_location = DEVICE)
        else:
            best_model = torch.load('bin/subjectivity.pt', map_location = DEVICE)

        print("BEST TRAINED MODEL\n")
        print("DATASET SIZE:", len(sentences), "(sentences) -", len(labels), "(labels)")
        # Evaluate the best model on the full dataset
        best_accuracy = test_loop(test_loader, best_model)
        print(f"Accuracy: {best_accuracy:.4f}\n")