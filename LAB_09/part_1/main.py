from functions import *

if __name__ == "__main__":
    dataloaders, lang = get_dataloaders("dataset/ptb.train.txt", "dataset/ptb.valid.txt", "dataset/ptb.test.txt")

    # train(dataloaders, lang)                                # TRAIN/VALID/TEST_BATCH = 128/512/512 | EMB_SIZE = 800 | HIDDEN_SIZE = 1000 | LR = 0.5 | CLIP = 10
    # train(dataloaders, lang, dropout = True)                # TRAIN/VALID/TEST_BATCH = 128/512/512 | EMB_SIZE = 800 | HIDDEN_SIZE = 1000 | LR = 0.5 | CLIP = 10
    # train(dataloaders, lang, dropout = True, AdamW = True)  # TRAIN/VALID/TEST_BATCH = 128/512/512 | EMB_SIZE = 600 | HIDDEN_SIZE = 800 | LR = 0.001 | WEIGHT_DECAY = 0.01 | EPS = 1e-7 | CLIP = 10

    train(dataloaders, lang, trained = True)
    train(dataloaders, lang, trained = True, dropout = True)
    train(dataloaders, lang, trained = True, AdamW = True)
