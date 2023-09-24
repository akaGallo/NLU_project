import torch, math, copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

def load_laptop14_data(file_path):
    # Load the entire file
    data = []
    with open(file_path, 'r', encoding = 'utf-8') as file:
        data = file.readlines()

    # Preprocess the loaded data
    text_preprocessed, polarity_list = preprocess_data(data)
    polarity_list = [pol for list in polarity_list for pol in list if list]

    # Obtain aspect terms and polarities from the datadet
    gold_ts = get_ts(data)
    gold_ot = get_ot(gold_ts)

    return text_preprocessed, polarity_list, gold_ot, gold_ts

def preprocess_data(data):
    text_list, polarity_list = [], []

    # Iterate through each line in the input data
    for line in data:
        # Split the line by '####' to split text from aspect_term and polarity
        _, annotation_str = map(str.strip, line.strip().split('####'))
        annotations = annotation_str.split()

        # Extract aspect terms and polarity from the annotations
        aspect_term = [term for term, _ in [ann.split('=') for ann in annotations]]
        polarity = [pol for _, pol in [ann.split('=') for ann in annotations]]
        text_list.append(aspect_term)
        polarity_list.append(polarity)

    # Flatten the list of aspect terms into a single list of tokens
    flattened_list = [token for sublist in text_list for token in sublist]
    return flattened_list, polarity_list

def get_ts(data):
    polarity_list, gold_ts = [], []

    for line in data:
        _, annotation_str = map(str.strip, line.strip().split('####'))
        annotations = annotation_str.split()
        polarity = [pol for _, pol in [ann.split('=') for ann in annotations]]
        polarity_list.append(polarity)

    # Iterate through the polarity values and convert them to gold_ts format
    for polarity in polarity_list:
        polarity = [pol[2:] if pol.startswith('T-') else pol for pol in polarity]
        test_gold_ts, _ = convert_polarity_in_ts(gold_ts, polarity, temp = None, count = 0)

    return test_gold_ts

def get_ot(test_ts):
    test_ot = []
    for ts in test_ts:
        if ts == "O":
            test_ot.append("O")
        elif ts.startswith("B"):
            test_ot.append("B")
        elif ts.startswith("I"):
            test_ot.append("I")
        elif ts.startswith("E"):
            test_ot.append("E")
        else:
            test_ot.append("S")
    
    return test_ot

def convert_polarity_in_ts(list_ts, polarity, temp, count):
    # If it is the last value of the list of polarities
    if polarity[-1] == "x" and count > 0:
        if count == 1:
            list_ts.append("S-" + temp)
        elif count == 2:
            list_ts.append("B-" + temp)
            list_ts.append("E-" + temp)
        else:
            for i in range(count):
                if i == 0:
                    list_ts.append("B-" + temp)
                elif i == count - 1:
                    list_ts.append("E-" + temp)
                else:
                    list_ts.append("I-" + temp)

    # If we don't find previous polarities different to "O" in the list
    elif count == 0:
        for idx, pol in enumerate(polarity):
            if pol == "x":
                pass
            elif pol == "O":
                list_ts.append("O")
                polarity[idx] = "x"
            else:
                temp = pol
                polarity[idx] = "x"
                count += 1
                convert_polarity_in_ts(list_ts, polarity, temp, count)
                break
    
    # If we have found a polarity different to "O" in the list
    else:
        for idx, pol in enumerate(polarity):
            if pol == "x":
                pass
            elif pol == "O" or pol != temp:
                if count == 1:
                    list_ts.append("S-" + temp)
                elif count == 2:
                    list_ts.append("B-" + temp)
                    list_ts.append("E-" + temp)
                else:
                    for i in range(count):
                        if i == 0:
                            list_ts.append("B-" + temp)
                        elif i == count - 1:
                            list_ts.append("E-" + temp)
                        else:
                            list_ts.append("I-" + temp)
                count = 0
                temp = None
                convert_polarity_in_ts(list_ts, polarity, temp, count)
                break
            else:
                polarity[idx] = "x"
                count += 1
                convert_polarity_in_ts(list_ts, polarity, temp, count)
                break
    return list_ts, polarity