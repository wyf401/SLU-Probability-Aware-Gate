import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF

class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)

        self.dropout = nn.Dropout(config.dropout)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot)
        # self.gate = ProbAwareGate(config.num_intent, config.num_slot)
        if config.use_crf:
            self.crf = CRF(config.num_slot, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        # slot_logits = self.gate(intent_logits,slot_logits)
        return intent_logits, slot_logits

class ProbAwareGate(nn.Module):
    def __init__(self, num_intent, num_slot):
        super(ProbAwareGate,self).__init__()
        self.mapping = nn.Linear(num_intent, num_slot)

    def forward(self, intent_logits, slot_logits):
        intent_slot_logits = self.mapping(intent_logits).unsqueeze(1)
        weight = slot_logits.bmm(intent_slot_logits.transpose(1,2))
        weight = F.normalize(weight)
        weight = weight.expand_as(slot_logits)
        weighted_intent_slot_logits = torch.mul(weight,intent_slot_logits.expand_as(slot_logits))
        slot_logits = slot_logits + weighted_intent_slot_logits
        return slot_logits