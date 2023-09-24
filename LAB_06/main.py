from functions import *

if __name__ == "__main__":
    data, tagged_data = get_data()
    
    # Generate a random sentence ID between 1 and 100
    id_sentence = random.randint(1, 100)
    
    # Iterate through each toolkit and analyze the dependency graphs
    toolkits = ['SPACY', 'STANZA']
    for toolkit in toolkits:
        get_dependency_graphs(data, tagged_data, toolkit, id_sentence)
    print()