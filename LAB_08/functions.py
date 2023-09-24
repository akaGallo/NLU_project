import nltk
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.metrics import precision, recall, f_measure, accuracy
from sklearn.model_selection import cross_validate, StratifiedKFold

def get_classifier_and_cross_val():
    nltk.download("senseval")
    nltk.download("wordnet")
    nltk.download('wordnet_ic')
    nltk.download("omw-1.4")
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    # Create a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    # Define a StratifiedKFold cross-validation strategy
    stratified_split = StratifiedKFold(n_splits = 5, shuffle = True)
    
    return classifier, stratified_split

def get_data(collocational = False, stratified = None, test_split = False):
    # Load instances from the senseval corpus
    instances = nltk.corpus.senseval.instances("interest.pos")
    
    # Generate data and labels
    if collocational:
        data = [collocational_features(inst, pos = True, ngram = True) for inst in instances]
    else:
        data = [" ".join([token[0] for token in inst.context]) for inst in instances]

    labels = [inst.senses[0] for inst in instances]

    # If test split is requested, perform stratified splitting
    if test_split:
        data_test = []
        labels_test = []

        for _, test_idx in stratified.split(data, labels):
            data_test.append([data[i] for i in test_idx])
            labels_test.append([labels[i] for i in test_idx])
        
        return data_test, labels_test

    return data, labels

def train_and_evaluate(classifier, data, labels, stratified_split, collocational = False, concatenate = False):
    # Label encoding for the classification labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    if concatenate:
        print("\n" + "="*26, "BAG-OF-WORDS + COLLOCATIONAL FEATURES", "="*27) 
        vectors = np.concatenate((data[0].toarray(), data[1]), axis = 1)
    else:
        print("\n" + "="*34, "COLLOCATIONAL FEATURES", "="*34) if collocational else print("\n" + "="*34, "BAG-OF-WORDS FEATURES", "="*35)
        # Create vectors using DictVectorizer for collocational or CountVectorizer for Bag-of-Words
        vectorizer = DictVectorizer(sparse = False) if collocational else CountVectorizer()
        vectors = vectorizer.fit_transform(data)
  
    scores = cross_validate(classifier, vectors, labels, cv = stratified_split, scoring = ["f1_micro"])
    print("\n" + "F1_micro =", sum(scores["test_f1_micro"]) / len(scores["test_f1_micro"]))

    return vectors

MAPPING = {
    'interest_1': 'interest.n.01',
    'interest_2': 'interest.n.03',
    'interest_3': 'pastime.n.01',
    'interest_4': 'sake.n.01',
    'interest_5': 'interest.n.05',
    'interest_6': 'interest.n.04',
}

def evaluate_word_sense_disambiguation(data_test, labels_test, algorithm):
    acc_list = []
    refs_list = []
    hyps_list = []
    synsets = []

    print("\n" + "="*35, algorithm, "METRIC", "="*35)
    for idx, (data, label) in enumerate(zip(data_test, labels_test)):
        print("\nTEST SPLIT", idx + 1)
        # Initialize dictionaries to store reference and hypothesis sets
        refs = {k: set() for k in MAPPING.values()}
        hyps = {k: set() for k in MAPPING.values()}

        # Iterate through WordNet synsets for the target word
        for ss in nltk.corpus.wordnet.synsets("interest", pos = "n"):
            if ss.name() in MAPPING.values():
                defn = ss.definition()
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                synsets.append((ss, toks))

        for i, (sent, lbl) in enumerate(zip(data, label)):
            # Determine the target word based on context
            word = "interest" if "interest" in sent else "interests"

            if algorithm == 'ORIGINAL LESK':
                hyp = original_lesk(sent, word, synsets = synsets, majority = True).name()
            elif algorithm == 'LESK SIMILARITY':
                hyp = lesk_similarity(sent, word, pos = "n", synsets = synsets, similarity = "path", majority = True).name()
            elif algorithm == 'PEDERSEN':
                hyp = pedersen(sent, word, pos = "n", synsets = synsets).name()
            else:
                print("Invalid algorithm choice.")
                return

            # Get the mapped reference label
            ref = MAPPING.get(lbl)
            
            refs[ref].add(i)
            hyps[hyp].add(i)

            refs_list.append(ref)
            hyps_list.append(hyp)

        acc = round(accuracy(refs_list, hyps_list), 3)
        acc_list.append(acc)

        for cls in hyps.keys():
            prec = precision(refs[cls], hyps[cls])
            rec = recall(refs[cls], hyps[cls])
            f1 = f_measure(refs[cls], hyps[cls], alpha = 1)
            
            if prec is None: prec = 0
            if rec is None: rec = 0
            if f1 is None: f1 = 0

            print("{:14s}:  precision = {:.3f};  recall = {:.3f};  f1_measure = {:.3f};  size = {}".format(cls, prec, rec, f1, len(refs[cls])))

    final_acc = sum(acc_list) / len(acc_list)
    print("\n" + algorithm, "ACCURACY:", final_acc)

def collocational_features(inst, pos = False, ngram = False):
    # Get the position of the target word in the instance
    p = inst.position
    features = {}

    # Define functions to get word, POS, and n-gram context
    def get_word_context(position):
        if position < 0 or position >= len(inst.context):
            return 'NULL'
        return inst.context[position][0]

    def get_pos_context(position):
        if position < 0 or position >= len(inst.context):
            return 'NULL'
        return inst.context[position][1]

    def get_ngram_context(position1, position2):
        if position1 < 0 or position1 >= len(inst.context) or position2 < 0 or position2 >= len(inst.context):
            return 'NULL'
        return inst.context[position1][0] + " " + inst.context[position2][0]

    # Populate features using context words
    features["w-2_word"] = get_word_context(p - 2)
    features["w-1_word"] = get_word_context(p - 1)
    features["w+1_word"] = get_word_context(p + 1)
    features["w+2_word"] = get_word_context(p + 2)

    # If requested, include POS context
    if pos:
        features["w-2_pos"] = get_pos_context(p - 2)
        features["w-1_pos"] = get_pos_context(p - 1)
        features["w+1_pos"] = get_pos_context(p + 1)
        features["w+2_pos"] = get_pos_context(p + 2)

    # If requested, include n-gram context
    if ngram:
        features["w-2:w-1"] = get_ngram_context(p - 2, p - 1)
        features["w-1:w+1"] = get_ngram_context(p - 1, p + 1)
        features["w+1:w+2"] = get_ngram_context(p + 1, p + 2)

    return features

def original_lesk(context_sentence, ambiguous_word, pos = None, synsets = None, majority = False):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense

def lesk_similarity(context_sentence, ambiguous_word, similarity = "resnik", pos = None, synsets = None, majority = True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    scores = []
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))
            
    if len(scores) == 0:
        return synsets[0][0]
    
    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    
    return best_sense

def pedersen(context_sentence, ambiguous_word, similarity = "resnik", pos = None, synsets = None, threshold = 0.1):
    semcor_ic = nltk.corpus.wordnet_ic.ic("ic-semcor.dat")
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    
    synsets_scores = {}
    for ss_tup in synsets:
        ss = ss_tup[0]
        if ss not in synsets_scores:
            synsets_scores[ss] = 0
        for senses in context_senses:
            scores = []
            for sense in senses[1]:
                if similarity == "path":
                    try:
                        scores.append((sense[0].path_similarity(ss), ss))
                    except:
                        scores.append((0, ss))    
                elif similarity == "lch":
                    try:
                        scores.append((sense[0].lch_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "wup":
                    try:
                        scores.append((sense[0].wup_similarity(ss), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "resnik":
                    try:
                        scores.append((sense[0].res_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "lin":
                    try:
                        scores.append((sense[0].lin_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                elif similarity == "jiang":
                    try:
                        scores.append((sense[0].jcn_similarity(ss, semcor_ic), ss))
                    except:
                        scores.append((0, ss))
                else:
                    print("Similarity metric not found")
                    return None
            value, sense = max(scores)
            if value > threshold:
                synsets_scores[sense] = synsets_scores[sense] + value
    
    values = list(synsets_scores.values())
    if sum(values) == 0:
        print('Warning all the scores are 0')
    senses = list(synsets_scores.keys())
    best_sense_id = values.index(max(values))
    return senses[best_sense_id]

def get_sense_definitions(context):
    lemma_tags = preprocess(context)
    senses = [(w, nltk.corpus.wordnet.synsets(l, p)) for w, l, p in lemma_tags]
    
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            def_list = []
            for s in sense_list:
                defn = s.definition()
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions

def get_top_sense(words, sense_list):
    val, sense = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense

def get_top_sense_sim(context_sense, sense_list, similarity):
    semcor_ic = nltk.corpus.wordnet_ic.ic("ic-semcor.dat")
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense

def preprocess(text):
    mapping = {"NOUN": nltk.corpus.wordnet.NOUN, "VERB": nltk.corpus.wordnet.VERB, "ADJ": nltk.corpus.wordnet.ADJ, "ADV": nltk.corpus.wordnet.ADV,}
    
    sw_list = nltk.corpus.stopwords.words("english")
    lem = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text) if type(text) is str else text
    tagged = nltk.pos_tag(tokens, tagset = "universal")
    tagged = [(w.lower(), p) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if p in mapping]
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    tagged = list(set(tagged))

    return tagged