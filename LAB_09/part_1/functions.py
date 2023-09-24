from model import *

# Define the device for computation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
EMBEDDING_SIZE = 800
HIDDEN_SIZE = 1000
LEARNING_RATE = 0.5
WEIGHT_DECAY = 0.01
EPSILON = 1e-7
N_EPOCHS = 50
PATIENCE = 5
CLIP = 10

def get_dataloaders(train_file, valid_file, test_file):
    train_raw = read_file(train_file)
    valid_raw = read_file(valid_file)
    test_raw = read_file(test_file)
    
    # Create a language instance
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    valid_dataset = PennTreeBank(valid_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]), shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = VALID_BATCH_SIZE, collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]))

    return [train_loader, valid_loader, test_loader], lang

def train_loop(data, optimizer, criterion, model, clip):
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:
        optimizer.zero_grad()  # Clear gradients
        output = model(sample["source"])  # Forward pass
        loss = criterion(output, sample["target"])  # Compute loss
        loss_array.append(loss.item() * sample["number_tokens"])  # Accumulate loss
        number_of_tokens.append(sample["number_tokens"])  # Accumulate token count
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Clip gradients to prevent exploding gradients
        optimizer.step()   # Update model parameters

    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])  # Forward pass
            loss = eval_criterion(output, sample['target'])  # Compute loss
            loss_array.append(loss.item())  # Accumulate loss
            number_of_tokens.append(sample["number_tokens"])  # Accumulate token count

    # Compute perplexity
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

# Function to initialize weights of certain layers in the model
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

def get_model(lang, dropout):
    OUTPUT_SIZE = len(lang.word2id)
    model = LM_LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, pad_index = lang.word2id["<pad>"], dropout = dropout)
    return model    

def train(dataloaders, lang, trained = False, dropout = False, AdamW = False):
    train_loader, valid_loader, test_loader = dataloaders
    criterion_train = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"], reduction = "sum")

    if not trained:
        if AdamW:
            print("\n" + "="*72, "LSTM MODEL WITH DROPOUT USING ADAMW OPTIMIZER", "="*72)
        elif dropout:
            print("\n" + "="*83, "LSTM MODEL WITH DROPOUT", "="*83)
        else:
            print("\n" + "="*89, "LSTM MODEL", "="*89)

        model = get_model(lang, dropout).to(DEVICE)
        model.apply(init_weights)

        if AdamW:
            optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, eps = EPSILON)
        else:
            optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, N_EPOCHS))
        patience = PATIENCE

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
            print()
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)  # Store epoch for plotting
                losses_train.append(np.asarray(loss).mean())  # Store average training loss
                ppl_valid, loss_valid = eval_loop(valid_loader, criterion_eval, model)
                losses_valid.append(np.asarray(loss_valid).mean())  # Store average validation loss
                pbar.set_description("Train PPL: %f" % ppl_valid)  # Update progress bar description
                if ppl_valid < best_ppl:
                    best_ppl = ppl_valid
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = PATIENCE
                else:
                    patience -= 1

                # Early stopping
                if patience <= 0:
                    break

        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        if AdamW:
            torch.save(best_model, 'bin/LSTM+dropout+AdamW.pt')
        elif dropout:
            torch.save(best_model, 'bin/LSTM+dropout.pt')
        else:
            torch.save(best_model, 'bin/LSTM.pt')
            
    # Loading a pre-trained model
    else:
        if AdamW:
            print("\n" + "="*35, "LSTM MODEL WITH DROPOUT USING ADAMW OPTIMIZER", "="*34)
            best_model = torch.load('bin/LSTM+dropout+AdamW.pt', map_location = DEVICE)
        elif dropout:
            print("\n" + "="*46, "LSTM MODEL WITH DROPOUT", "="*45)
            best_model = torch.load('bin/LSTM+dropout.pt', map_location = DEVICE)
        else:
            print("\n" + "="*52, "LSTM MODEL", "="*52)
            best_model = torch.load('bin/LSTM.pt', map_location = DEVICE)

        print("BEST TRAINED MODEL")
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

    print("\nTest PPL: ", final_ppl, "\n")