from functions import *

if __name__ == "__main__":
    dataloaders, lang = get_dataloaders("dataset/ptb.train.txt", "dataset/ptb.valid.txt", "dataset/ptb.test.txt")

    # train(dataloaders, lang, weight_tying = True)                                             # BATCH = 256/512/512 | EMB_SIZE = 600 | HIDDEN_SIZE = 800 | LR = 0.01 | WD = 0.01 | EPS = 1e-7 | CLIP = 5
    # train(dataloaders, lang, weight_tying = True, VariationalDropout = True)                  # BATCH = 256/512/512 | EMB_SIZE = 600 | HIDDEN_SIZE = 800 | LR = 0.005 | WD = 0.01 | EPS = 1e-7 | CLIP = 5
    # train(dataloaders, lang, weight_tying = True, VariationalDropout = True, NTASGD = True)   # BATCH = 256/512/512 | EMB_SIZE = 600 | HIDDEN_SIZE = 800 | LR = 30 | BPTT = 70 | CLIP = 5

    train(dataloaders, lang, trained = True, weight_tying = True)
    train(dataloaders, lang, trained = True, VariationalDropout = True)
    train(dataloaders, lang, trained = True, NTASGD = True)