from utils import *
from model import *

# Check if CUDA (GPU) is available, otherwise use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define batch sizes and hyperparameters
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LEARNING_RATE = 0.00005
DROPOUT = 0.1
CLIP = 5
PATIENCE = 20
N_EPOCHS = 400
OUTPUT_SLOT = lambda x: len(x.slot2id)
OUTPUT_INTENT = lambda x: len(x.intent2id)
PAD_TOKEN = BertTokenizer.from_pretrained("bert-base-uncased").pad_token_id # BERT padding token

def get_dataloaders(train_file, test_file):
    tmp_train_raw = load_data(train_file)
    test_raw = load_data(test_file)

    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10) / (len(tmp_train_raw)), 2)
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])

    X_train, X_valid, _, _ = train_test_split(X, Y, test_size = portion, random_state = 42, stratify = Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    valid_raw = X_valid

    lang = get_lang(train_raw, valid_raw, test_raw)

    train_dataset = IntentsAndSlots(train_raw, lang)
    valid_dataset = IntentsAndSlots(valid_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, collate_fn = collate_fn,  shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = VALID_BATCH_SIZE, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, collate_fn = collate_fn)

    return [train_loader, valid_loader, test_loader], lang

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample["utterances"], sample["utt_mask"])
        # Calculate losses for intent and slot prediction
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        # Backpropagation and optimization step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()

    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            # Perform inference
            slots, intents = model(sample["utterances"], sample["utt_mask"])
            # Calculate losses for intent and slot prediction
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim = 1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim = 1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        pass

    report_intent = classification_report(ref_intents, hyp_intents, zero_division = False, output_dict = True)
    return results, report_intent, loss_array

# Function to initialize weights of a module
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
    
def train(dataloaders, lang, trained = False):
    train_loader, valid_loader, test_loader = dataloaders
    criterion_slots = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if not trained:
        print("\n" + "="*90, "BERT MODEL", "="*90 + "\n")

        model = BERT(OUTPUT_SLOT(lang), OUTPUT_INTENT(lang), dropout = DROPOUT).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_f1 = 0
        pbar = tqdm(range(1, N_EPOCHS))
        patience = PATIENCE

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            if epoch % 5 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                results_valid, _, loss_valid = eval_loop(valid_loader, criterion_slots, criterion_intents, model, lang)
                losses_valid.append(np.asarray(loss_valid).mean())
                f1 = results_valid["total"]["f"]
                if f1 > best_f1:
                    best_f1 = f1
                    patience = PATIENCE
                else:
                    patience -= 1

                if patience <= 0:
                    break

        model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        torch.save(model, 'bin/BERT.pt')

    else:
        print("\n" + "="*52, "BERT MODEL", "="*52)
        print("BEST TRAINED MODEL")
        
        best_model = torch.load('bin/BERT.pt', map_location = DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)

    print("\nSlot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"], "\n")