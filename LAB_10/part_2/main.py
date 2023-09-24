from functions import *

if __name__ == "__main__":
    dataloaders, lang = get_dataloaders('dataset/ATIS/train.json', 'dataset/ATIS/test.json')

    # train(dataloaders, lang)  # BATCH = 128/64/64 | LEARNING_RATE = 0.00005 | DROPOUT = 0.1 | CLIP = 5

    train(dataloaders, lang, trained = True)
