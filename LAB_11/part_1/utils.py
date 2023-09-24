import nltk, torch, string, os, zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import StratifiedKFold
from transformers import BertForSequenceClassification, BertTokenizer
from nltk.corpus import subjectivity, movie_reviews

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

def load_subjectivity_data(subjective_label, objective_label):
    # Download the 'subjectivity' dataset from NLTK
    nltk.download('subjectivity')

    # Load sentences categorized as subjective and objective
    subjective_sentences = subjectivity.sents(categories = subjective_label)
    objcetive_sentences = subjectivity.sents(categories = objective_label)

    subjective_preprocesses = []
    objective_preprocesses = []

    # Process subjective sentences
    for sent in subjective_sentences:
        preprocessed_sentence = preprocess_text(' '.join(sent))
        subjective_preprocesses.append([preprocessed_sentence, subjective_label])

    # Process objective sentences
    for sent in objcetive_sentences:
        preprocessed_sentence = preprocess_text(' '.join(sent))
        objective_preprocesses.append([preprocessed_sentence, objective_label])

    # Combine both sets of sentences and labels into a single list
    sentences = [item[0] for item in subjective_preprocesses] + [item[0] for item in objective_preprocesses]
    labels = [item[1] for item in subjective_preprocesses] + [item[1] for item in objective_preprocesses]

    # Join individual words in each sentence to form a single string
    sentences = [' '.join(sentence) for sentence in sentences]
    return [sentences, labels]    

def load_polarity_data(positive_label, negative_label):
    # Download the 'movie_reviews' dataset from NLTK
    nltk.download('movie_reviews')

    # Get a list of file IDs for positive and negative reviews
    positive_reviews = movie_reviews.fileids(positive_label)
    negative_reviews = movie_reviews.fileids(negative_label)

    positive_preprocessed = []
    negative_preprocessed = []

    sid = SentimentIntensityAnalyzer()

    # Iterate through all reviews (both positive and negative)
    for file_id in tqdm(positive_reviews + negative_reviews):
        sentences = movie_reviews.sents(file_id)
        for sent in sentences:
            preprocessed_sentence = preprocess_text(' '.join(sent))

            # Analyze the sentiment of the current sentence
            ss = sid.polarity_scores(' '.join(sent))
            sentiment_label = ['Pos' if ss['compound'] > 0 else 'Neg' if ss['compound'] < 0 else 'Neu']

            # Determine if the current review is positive or negative and append the preprocessed sentence and label
            if file_id.startswith('pos'):
                positive_preprocessed.append([preprocessed_sentence, sentiment_label])
            else:
                negative_preprocessed.append([preprocessed_sentence, sentiment_label])

    # Combine both sets of sentences and labels into a single list
    sentences = [item[0] for item in positive_preprocessed] + [item[0] for item in negative_preprocessed]
    labels = [item[1] for item in positive_preprocessed] + [item[1] for item in negative_preprocessed]

    # Join individual words in each sentence to form a single string
    sentences = [' '.join(sentence) for sentence in sentences]
    return [sentences, labels]

def objective_removal(polarity_data):
    polarity_sentences, polarity_labels = polarity_data

    # Define file paths for storing subjective data (sentences and labels) from movie reviews dataset in zip files (if exist)
    dataset_folder = 'dataset'
    sentences_file = os.path.join(dataset_folder, 'polarity_subjective_sentences.zip')
    labels_file = os.path.join(dataset_folder, 'polarity_subjective_labels.zip')

    subjective_sentences = load_data_from_zip(sentences_file)
    subjective_labels = [[label] for label in load_data_from_zip(labels_file)]

    if not subjective_sentences and not subjective_labels:
        # Check if the 'subjectivity.pt' model exists
        if not os.path.exists('bin/subjectivity.pt'):
            print("ERROR: 'subjectivity.pt' model don't exist!")
            return

        # Load the pre-trained 'subjectivity.pt' model
        subjectivity_model = torch.load('bin/subjectivity.pt', map_location = DEVICE)
        subjectivity_model.eval()

        subjective_sentences = []
        subjective_labels = []

        for sentence, label in zip(polarity_sentences, polarity_labels):
            # Tokenize and prepare the input for the subjectivity model
            inputs = TOKENIZER(sentence, return_tensors = 'pt', padding = True, truncation = True)
            input_ids = inputs['input_ids'].to(DEVICE)
            attention_mask = inputs['attention_mask'].to(DEVICE)

            with torch.no_grad():
                # Pass the input through the subjectivity model to get predictions
                outputs = subjectivity_model(input_ids, attention_mask = attention_mask)

            # Determine if the sentence is subjective (predicted_label == 1)
            predicted_label = torch.argmax(outputs.logits, dim = 1).item()
            if predicted_label == 1:
                subjective_sentences.append(sentence)
                subjective_labels.append(label)

        # Save subjective sentences and labels to zip files for future use
        save_data_to_zip(subjective_sentences, sentences_file)
        save_data_to_zip(subjective_labels, labels_file)

    return [subjective_sentences, subjective_labels]

def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenize the text into words
    tokens = [token.lower() for token in tokens if token not in string.punctuation]  # Convert tokens to lowercase and remove punctuation
    stop_words = set(stopwords.words('english'))  # Get a set of stopwords in English
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords from tokens
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]  # Lemmatize tokens to their base form
    return tokens

def save_data_to_zip(data, file_path):
    # Convert the inner lists to strings
    data_str = [''.join(map(str, item)) for item in data]

    # Save the strings as zip files in the given path
    with zipfile.ZipFile(file_path, 'w') as zip_file:
        zip_file.writestr('data.txt', '\n'.join(data_str))

def load_data_from_zip(file_path):
    data = []
    # Check if the zip files exist in the given path
    if os.path.exists(file_path):
        # Unzip the files and load data
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            data = zip_file.read(zip_file.namelist()[0]).decode().splitlines()
    return data