from utils import *

class SubjectivityModel(torch.nn.Module):
    def __init__(self, num_subjectivity_classes):
        super(SubjectivityModel, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_subjectivity_classes)  # 2 labels: "subj" and "obj"
    
    def forward(self, input_ids, attention_mask):
        return self.bert_model(input_ids, attention_mask = attention_mask)
    
class PolarityModel(torch.nn.Module):
    def __init__(self, num_polarity_classes):
        super(PolarityModel, self).__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_polarity_classes)  # 3 labels: "Neg", "Neu" and "Pos"

    def forward(self, input_ids, attention_mask):
        return self.bert_model(input_ids, attention_mask = attention_mask)