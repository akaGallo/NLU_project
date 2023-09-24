from model import *
from utils import *
from evals import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
CRITERION_LOSS = nn.CrossEntropyLoss()
POLARITY_CLASSES = 4  # "O", "T-POS", "T-NEG", "T-NEU"

TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LEARNING_RATE = 0.00005
PATIENCE = 5
N_EPOCHS = 100

def prepare_dataset(text, gold_ts):
    encoded_text = TOKENIZER(text, padding = True, truncation = True, return_tensors = 'pt', return_attention_mask = True)
    gold_ts_map = {"O": 0, "T-POS": 1, "T-NEG": 2, "T-NEU": 3}  # Define a mapping from gold_ts labels to numerical values
    encoded_gold_ts = torch.tensor([gold_ts_map[ts] for ts in gold_ts], dtype = torch.long)
    dataset = torch.utils.data.TensorDataset(encoded_text['input_ids'], encoded_text['attention_mask'], encoded_gold_ts)
    return dataset

def get_dataloaders(train_dataset, test_dataset):
    # Create data loaders for train and test datasets using the specific batch size
    train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE)

    return [train_loader, test_loader]

# Aspect-Based Sentiment Analysis loss function for polarity loss
class ABSALoss(nn.Module):
    def __init__(self):
        super(ABSALoss, self).__init__()
        self.polarity_loss = CRITERION_LOSS

    def forward(self, polarity_logits, polarity_labels):
        polarity_loss = self.polarity_loss(polarity_logits, polarity_labels)
        return polarity_loss

def train_loop(train_loader, criterion_loss, optimizer, model):
    # Set the model to training mode
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        polarity_labels = batch[2].to(DEVICE)

        optimizer.zero_grad()
        polarity_logits = model(input_ids, attention_mask = attention_mask)
        loss = criterion_loss(polarity_logits, polarity_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def test_loop(eval_loader, model):
    # Set the model to eval mode
    model.eval()
    polarity_predictions_list = []
    polarity_labels_list = []

    for batch in eval_loader:
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        polarity_labels = batch[2].to(DEVICE)

        with torch.no_grad():
            polarity_logits = model(input_ids, attention_mask = attention_mask)
        
        polarity_predictions_list.extend(torch.argmax(polarity_logits, dim = -1).tolist())
        polarity_labels_list.extend(polarity_labels.tolist())

    # Compute accuracy based on the collected polarity predictions and polarity labels
    polarity_accuracy = accuracy_score(polarity_labels_list, polarity_predictions_list)
    return polarity_predictions_list, polarity_accuracy

def train(dataloader, trained = False):
    train_loader, test_loader = dataloader

    print("\n" + "="*35, "ASPECT-BASED SENTIMENT ANALYSIS (BERT) MODEL", "="*35)

    pred_ts = []
    if not trained:
        model = ABSAModel(POLARITY_CLASSES).to(DEVICE)
        criterion_loss = ABSALoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

        losses_train = []
        sampled_epochs = []
        best_loss = math.inf
        best_model = None
        pbar = tqdm(range(1, N_EPOCHS))
        patience = PATIENCE

        for epoch in pbar:
            train_loss = train_loop(train_loader, criterion_loss, optimizer, model)
            print()
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)  # Store epoch for plotting
                losses_train.append(np.asarray(train_loss).mean())  # Store average training loss
                pbar.set_description("Train loss: %f" % train_loss)  # Update progress bar description
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model = copy.deepcopy(model).to(DEVICE)
                    patience = PATIENCE
                else:
                    patience -= 1

                # Early stopping
                if patience <= 0:
                    break

        # Make predictions on the test data
        best_model.to(DEVICE)
        pred_ts, test_accuracy = test_loop(test_loader, best_model)
        torch.save(best_model, 'bin/Aspect_based_sentiment_analysis.pt')

    else:
        best_model = torch.load('bin/Aspect_based_sentiment_analysis.pt', map_location = DEVICE)
        print("BEST TRAINED MODEL")
        pred_ts, test_accuracy = test_loop(test_loader, best_model)

    print("\nTest accuracy: ", test_accuracy, "\n")
    
    predicted_ts1, predicted_ts2 = [], []
    gold_ts_map = {0: "O", 1: "POS", 2: "NEG", 3: "NEU"}
    temp_pred_ts = [gold_ts_map[num] for num in pred_ts]
    
    half_pred_ts = len(temp_pred_ts) // 2
    predicted_ts1, _ = convert_polarity_in_ts(predicted_ts1, temp_pred_ts[:half_pred_ts], temp = None, count = 0)
    predicted_ts2, _ = convert_polarity_in_ts(predicted_ts2, temp_pred_ts[half_pred_ts:], temp = None, count = 0)
    predicted_ts = predicted_ts1 + predicted_ts2
    
    predicted_ot = get_ot(predicted_ts) 
    return predicted_ot, predicted_ts
    
def ABSA_evaluation(gold_ot, gold_ts, pred_ot, pred_ts):
    ote_scores, ts_scores = evaluate(gold_ot, gold_ts, pred_ot, pred_ts)

    print("Task 1 - Aspect Term Extraction (F1, Precision, Recall): ({:.5f}, {:.5f}, {:.5f})".format(*ote_scores))
    print("Task 2 - Polarity Detection (F1, Precision, Recall): ({:.5f}, {:.5f}, {:.5f})".format(*ts_scores), "\n")