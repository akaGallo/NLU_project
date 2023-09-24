from utils import *

# Define the Aspect-Based Sentiment Analysis model
class ABSAModel(nn.Module):
    def __init__(self, num_polarities):
        super(ABSAModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.polarity_classifier = nn.Linear(self.bert.config.hidden_size, num_polarities)  # 4 labels: "O", "T-POS", "T-NEG", "T-NEU"

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask = attention_mask)
        pooler_output = outputs.pooler_output  # Use the pooled output from BERT
        polarity_logits = self.polarity_classifier(pooler_output)

        return polarity_logits