import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, use_sims=False):
        super(BertForSequenceClassification, self).__init__(config)
        self.use_sims = use_sims
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sim_features_dropout = torch.nn.Dropout(0.2)
        h_size = config.hidden_size
        if use_sims: h_size += 766
        self.classifier = torch.nn.Linear(h_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sim_features=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        if self.use_sims:
            sim_features = self.sim_features_dropout(sim_features)
            pooled_output = torch.cat([pooled_output, sim_features], dim=-1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class ElmoSequenceClassification(torch.nn.Module):
    def __init__(self, num_labels):
        exit()
        super(ElmoSequenceClassification, self).__init__()
        self.elmo_encoder = Elmo(options_file, weight_file, 2, dropout=0)
        self.gru = torch.nn.GRU(1024, 200, 1, dropout=0, bidirectional=True, batch_first=True)
        self.classifier = torch.nn.Linear(400, num_labels)
        self.a = torch.autograd.Variable(torch.tensor(0.5))
        self.b = torch.autograd.Variable(torch.tensor(0.5))
        self.config = {}

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoded_output = self.elmo_encoder(input_ids)
        a, b = encoded_output['elmo_representations']
        encoded_output = a
        states, encoded_output = self.gru(encoded_output)
        encoded_output = attention_mask.view(-1, 30, 1).type(torch.cuda.FloatTensor)*states
        encoded_output = torch.sum(encoded_output, dim=1)
        logits = self.classifier(encoded_output)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
