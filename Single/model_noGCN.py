from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from config import args

class RefineLayer(nn.Module):
    def __init__(self, max_lens, input_dim, gru_dim):
        super(RefineLayer, self).__init__()
        self.con_len = max_lens - 1
        self.rnn1 = nn.GRU(input_size=input_dim, hidden_size=gru_dim, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True)
        self.rnn2 = nn.GRU(input_size=input_dim, hidden_size=gru_dim, 
                           num_layers=2, batch_first=True, 
                           bidirectional=True)
        self.linear1 = nn.Linear(2*gru_dim, gru_dim)
        self.linear2 = nn.Linear(2*gru_dim, gru_dim)

    
    def forward(self, inputs):
        context_hidden_org = inputs[:, :self.con_len, :]
        pos_hidden_org = inputs[:, self.con_len:, :]
        # refine by GRU
        context_hidden, _ = self.rnn1(context_hidden_org)
        pos_hidden, _ = self.rnn2(pos_hidden_org)
        # Compresse
        context_hidden = self.linear1(context_hidden)
        pos_hidden = self.linear2(pos_hidden)
        return F.relu(torch.cat((context_hidden, pos_hidden), dim=-1))


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
        self.gru = RefineLayer(max_lens=args.max_seq_length, 
                                        input_dim=config.hidden_size, 
                                        gru_dim=config.hidden_size//2)
        ###
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
        x1 = self.gru(sequence_output) 
        
        sequence_output = self.dropout(x1)

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