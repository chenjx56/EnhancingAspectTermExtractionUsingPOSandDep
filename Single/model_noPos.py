from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from config import args


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)


    def forward(self, text, adj):
        # lens, lens * lens, h_d --> lens, in_hd
        text_weighted = torch.matmul(adj, text) 
        # lens, out_hd
        hidden = self.weight(text_weighted)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = hidden / denom
        return output


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ###
        self.rnn1 = nn.GRU(input_size=config.hidden_size, hidden_size=config.hidden_size//2, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True)
        self.gcn1 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn2 = GraphConvolution(config.hidden_size, config.hidden_size)
        ###
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        dependency_graph=None, #
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        #
        sequence_output, _ = self.rnn1(sequence_output)
        x2 = F.relu(self.gcn1(sequence_output, dependency_graph)) #bz, 102, 768
        x2 = F.relu(self.gcn2(x2, dependency_graph)) # bz, 102, 768
        x = torch.cat((sequence_output, x2), dim=-1) # bz, 102, 768 * 2
        bz, seq, h_d = sequence_output.shape
        assert x.shape == (bz, seq, h_d * 2)
        
        sequence_output = self.dropout(x)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                _, seqlen = labels.shape
                attention_mask = attention_mask[:, :seqlen]
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )