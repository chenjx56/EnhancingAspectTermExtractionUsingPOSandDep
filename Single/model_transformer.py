from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertLayer
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
        # lens, lens * lens, in_hd --> lens, in_hd
        text_weighted = torch.matmul(adj, text) 
        # lens, out_hd
        hidden = self.weight(text_weighted)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = hidden / denom
        return output

class RefineLayer(nn.Module):
    def __init__(self, max_lens, config):
        super(RefineLayer, self).__init__()
        self.con_len = max_lens - 1
        
        self.att1 = BertLayer(config)
        self.att2 = BertLayer(config)

        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size//2)
    
    def forward(self, inputs, attention_mask):
        context_hidden_org = inputs[:, :self.con_len, :]
        pos_hidden_org = inputs[:, self.con_len:, :]
        
        # refine by transformer encoder
        context_hidden = self.att1(context_hidden_org, attention_mask)[0]
        pos_hidden = self.att2(pos_hidden_org, attention_mask)[0]
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
        self.gru = RefineLayer(max_lens=args.max_seq_length, config=config)
        
        self.gcn1 = GraphConvolution(config.hidden_size, config.hidden_size)
        self.gcn2 = GraphConvolution(config.hidden_size, config.hidden_size)
        ###
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)

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
        seqlens = input_ids.shape[1] // 2
        attention_mask = attention_mask[:, :seqlens]
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, sequence_output.shape, sequence_output.device)

        x1 = self.gru(sequence_output, extended_attention_mask) 

        x2 = F.relu(self.gcn1(x1, dependency_graph)) #bz, 101, 768
        x2 = F.relu(self.gcn2(x2, dependency_graph)) # bz, 101, 768

        x = torch.cat((x1, x2), dim=-1) # bz, 101, 768 * 2

        bz, seq, h_d = x1.shape
        assert x.shape == (bz, seq, h_d * 2)
        
        sequence_output = self.dropout(x)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # _, seqlen = labels.shape
                # attention_mask = attention_mask[:, :seqlen]
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