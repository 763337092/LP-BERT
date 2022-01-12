import math
import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel, RobertaModel

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
BertLayerNorm = torch.nn.LayerNorm

class BertLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(BertLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x

class MultiTaskBert(BertPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, pretrained_path, config):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(pretrained_path, config=config) # , add_pooling_layer=False
        self.lm_head = BertLMHead(config)
        self.elm_head = BertLMHead(config)

        # self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        mlm_outputs = self.lm_head(sequence_output)
        emlm_outputs = self.elm_head(sequence_output)

        return mlm_outputs, emlm_outputs

class LPBERT(nn.Module):
    def __init__(self, pretrained_path, config, dataset_name):
        super(LPBERT, self).__init__()
        self.dataset_name = dataset_name
        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained(pretrained_path, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, 1)  # 4个 rep_src, rep_tgt, src - tgt, src * tgt

        self.distance_fn = self.euclidean_distance

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def classifier(self, rep_src, rep_tgt):
        if self.dataset_name == 'WN18RR':
            logits = torch.cosine_similarity(rep_src, rep_tgt)
        else:
            cls_feature = torch.cat(
                [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
            )
            cls_feature = self.dropout(cls_feature)
            logits = self.proj(cls_feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None):
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)
        logits = self.classifier(rep_src, rep_tgt)
        distances = self.distance_fn(
            rep_src,#self.linear12(self.gelu(self.linear11(rep_src))),
            rep_tgt#self.linear22(self.gelu(self.linear21(rep_tgt)))
        )
        return torch.sigmoid(logits), distances

    def euclidean_distance(self, rep1, rep2):
        distance = rep1 - rep2
        distance = torch.norm(distance, p=2, dim=-1)
        return distance