# Import necessary libraries and modules
import nltk, spacy, stanza, spacy_stanza, en_core_web_sm, random
from spacy.tokenizer import Tokenizer
from nltk.parse import DependencyEvaluator
from nltk.corpus import dependency_treebank
from nltk.parse.dependencygraph import DependencyGraph

def get_data():
    nltk.download('dependency_treebank')

    data = dependency_treebank.sents()[-100:]
    tagged_data = dependency_treebank.parsed_sents()[-100:]
    return data, tagged_data

# Define a function to create a SpaCy pipeline with a custom CONLL formatter
def get_spacy_conversion():
    nlp_spacy = spacy.load("en_core_web_sm")
    config_spacy = {
        "ext_names": {"conll_pd": "pandas"},
        "conversion_maps": {"deprel": {"nsubj": "subj"}}
    }
    nlp_spacy.add_pipe("conll_formatter", config = config_spacy, last = True)
    nlp_spacy.tokenizer = Tokenizer(nlp_spacy.vocab)
    return nlp_spacy

# Define a function to create a Stanza pipeline with a custom CONLL formatter
def get_stanza_conversion():
    stanza.download("en", verbose = False)
    nlp_stanza = spacy_stanza.load_pipeline("en", verbose = False, tokenize_pretokenized = True)
    config_stanza = {
        "ext_names": {"conll_pd": "pandas"},
        "conversion_maps": {"DEPREL": {"nsubj": "subj", "root": "ROOT"}}
    }
    nlp_stanza.add_pipe("conll_formatter", config = config_stanza, last = True)
    return nlp_stanza

def get_dependency_graphs(data, tagged_data, tool, id_sentence):
    print("\n" + "="*68, tool, "DEPENDENCY GRAPHS", "="*68)
    
    # Choose the appropriate toolkit and create a pipeline
    if tool == 'SPACY':
        nlp = get_spacy_conversion()
    elif tool == 'STANZA':
        nlp = get_stanza_conversion()
    else:
        return None
    
    # Initialize a list to store dependency graphs
    graphs = []
    
    # Iterate through the sentences and analyze their dependency graphs
    for idx, sentence in enumerate(data):
        sentence = " ".join(sentence)
        doc = nlp(sentence)
        df = doc._.pandas
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header = False, index = False)

        if idx == id_sentence:
            print("\n" + "SENTENCE", id_sentence, "FROM TREEBANK DATASET:", sentence + "\n\n" + tmp)

        dp = DependencyGraph(tmp)
        graphs.append(dp)

    print("\n" + "="*62, tool, "DEPENDENCY PARSING PERFORMANCE", "="*62 + "\n")
    evaluator = DependencyEvaluator(graphs, tagged_data)
    las, uas = evaluator.eval()
    print("LAS (Labeled Attachment Score):", las)
    print("UAS (Unlabeled Attachment Score):", uas)