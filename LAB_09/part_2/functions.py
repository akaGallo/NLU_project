from model import *

# Check if GPU is available, otherwise use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
EMBEDDING_SIZE = 600
HIDDEN_SIZE = 800
LEARNING_RATE = 30
WEIGHT_DECAY = 0.01
EPSILON = 1e-7
N_EPOCHS = 150
PATIENCE = 5
BPTT = 70
CLIP = 5

def get_dataloaders(train_file, valid_file, test_file):
    train_raw = read_file(train_file)
    valid_raw = read_file(valid_file)
    test_raw = read_file(test_file)
    
    # Create a language vocabulary object
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
        optimizer.zero_grad() 
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
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

def get_seq_len(bptt):
        seq_len = bptt if np.random.random() < 0.95 else bptt / 2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = bptt if np.random.random() < 0.95 else bptt / 2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

# Define a function to get the language model
def get_model(lang, weight_tying, VariationalDropout):
    OUTPUT_SIZE = len(lang.word2id)
    model = LM_LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, pad_index = lang.word2id["<pad>"], weight_tying = weight_tying, variationalDropout = VariationalDropout)
    return model    

def train(dataloaders, lang, trained = False, weight_tying = False, VariationalDropout = False, NTASGD = False):
    train_loader, valid_loader, test_loader = dataloaders
    criterion_train = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"], reduction = "sum")

    if not trained:
        print("\n" + "="*50, "LSTM MODEL", end = " ")
        print("WITH VARIATIONAL DROPOUT", end = " ") if VariationalDropout else print("WITH DROPOUT", end = " ")
        if weight_tying: print("AND WEIGHT TYING", end = " ")
        print("USING NTASGD OPTIMIZER", end = " ") if NTASGD else print("USING ADAMW OPTIMIZER", end = " ")
        print("="*50 + "\n")
    
        # If the model is not trained, create a new model and train it
        model = get_model(lang, weight_tying, VariationalDropout).to(DEVICE)
        model.apply(init_weights)

        if NTASGD:
            optimizer = NTASGDoptim(model.parameters(), lr = LEARNING_RATE)
        else:
            optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY, eps = EPSILON)

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, N_EPOCHS))
        patience = PATIENCE

        for epoch in pbar:
            if NTASGD:
                seq_len = get_seq_len(BPTT)
                optimizer.lr(seq_len / BPTT * LEARNING_RATE)

            loss = train_loop(train_loader, optimizer, criterion_train, model, CLIP)
            print()
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                if NTASGD:
                    tmp = {}
                    for (prm,st) in optimizer.state.items():
                        tmp[prm] = prm.clone().detach()
                        prm.data = st['ax'].clone().detach()

                ppl_valid, loss_valid = eval_loop(valid_loader, criterion_eval, model)

                if NTASGD:
                    optimizer.check(ppl_valid)

                losses_valid.append(np.asarray(loss_valid).mean())
                pbar.set_description("Train PPL: %f" % ppl_valid)
                if ppl_valid < best_ppl:
                    best_ppl = ppl_valid
                    best_model = copy.deepcopy(model).to(DEVICE)
                    patience = PATIENCE
                else:
                    patience -= 1
                
                if NTASGD:
                    for (prm,st) in optimizer.state.items():
                        prm.data = tmp[prm].clone().detach()

                if patience <= 0:
                    break

        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        if NTASGD:
            torch.save(best_model, 'bin/LSTM+WeightTying+VariationalDropout+NTASGD.pt')
        elif VariationalDropout:
            torch.save(best_model, 'bin/LSTM+WeightTying+VariationalDropout+AdamW.pt')
        else:
            torch.save(best_model, 'bin/LSTM+WeightTying+dropout+AdamW.pt')

    else:
        if NTASGD:
            print("\n" + "="*20, "LSTM MODEL WITH VARIATIONAL DROPOUT AND WEIGHT TYING USING NTASGD OPTIMIZER", "="*19)
            best_model = torch.load('bin/LSTM+WeightTying+VariationalDropout+NTASGD.pt', map_location = DEVICE)
        elif VariationalDropout:
            print("\n" + "="*20, "LSTM MODEL WITH VARIATIONAL DROPOUT AND WEIGHT TYING USING ADAMW OPTIMIZER", "="*20)
            best_model = torch.load('bin/LSTM+WeightTying+VariationalDropout+AdamW.pt', map_location = DEVICE)
        else:
            print("\n" + "="*26, "LSTM MODEL WITH DROPOUT AND WEIGHT TYING USING ADAMW OPTIMIZER", "="*26)
            best_model = torch.load('bin/LSTM+WeightTying+dropout+AdamW.pt', map_location = DEVICE)

        print("BEST TRAINED MODEL")
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

    print("\nTest PPL: ", final_ppl, "\n")
