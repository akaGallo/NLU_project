from utils import *

class BERT(nn.Module):
    def __init__(self, out_slot, out_int, dropout = None):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # Pre-trained BERT model for contextual word representations
        self.slot_out = nn.Linear(self.bert.config.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.bert.config.hidden_size, out_int)
        self.dropout = nn.Dropout(dropout) if dropout else None     # Dropout layer for regularization, if dropout is provided

    def forward(self, utterance, utterance_mask):
        bert_out = self.bert(input_ids = utterance, attention_mask = utterance_mask)         # Pass the utterance through the BERT model
        sequence_output, pooled_output = bert_out.last_hidden_state, bert_out.pooler_output  # Extract sequence and pooled outputs from BERT
        if self.dropout:
            sequence_output = self.dropout(sequence_output)
            pooled_output = self.dropout(pooled_output)
        slots = self.slot_out(sequence_output)   # Generate slot predictions from sequence output
        intent = self.intent_out(pooled_output)  # Generate intent prediction from pooled output
        slots = slots[:, 1:-1, :].permute(0, 2, 1)  # Rearrange slot predictions for compatibility with the loss function

        return slots, intent
