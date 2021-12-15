import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from components import masked_avgpool


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config.bert_config)
        self.crossattention = BertAttention(config.bert_config)
        self.intermediate = BertIntermediate(config.bert_config)
        self.output = BertOutput(config.bert_config)

    def forward(
        self,
        hidden_states,  # B num_generated_triples H
        encoder_hidden_states,
        encoder_attention_mask
    ):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0] #hidden_states.shape
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :] # B 1 1 H
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0 #1 1 0 0 -> 0 0 -1000 -1000
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0] #B m H
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output) #B m H
        outputs = (layer_output,) + outputs
        return outputs


class RelModel(nn.Module):
    """Some Information about RelModel"""
    def __init__(self, config):
        super(RelModel, self).__init__()
        self.rounds = 2
        self.rel_num = config.rel_num
        self.config = config
        self.args = config.args

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=config.bert_model_dir)
        self.dropout = nn.Dropout(config.drop_prob)
        self.norm = nn.BatchNorm2d(config.rel_num)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        # self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)

        if config.args.use_feature_enchanced:
            self.feat_enhance_sub = nn.Linear(config.hidden_size, config.hidden_size)
            self.feat_enhance_obj = nn.Linear(config.hidden_size, config.hidden_size)

        self.build_table = nn.Linear(config.hidden_size, config.rel_num)
        self.fcn_sub = nn.Linear(config.rel_num, config.hidden_size)
        self.fcn_obj = nn.Linear(config.rel_num, config.hidden_size)
        self.map_cor = nn.Linear(config.rel_num, 1)

        if 'sent' in self.args.set_ent_level:
            self.map_sub = nn.Linear(config.hidden_size, self.rel_num)
            self.map_obj = nn.Linear(config.hidden_size, self.rel_num)
        if 'map' in self.args.set_rel_level:
            self.map_rel = nn.Linear(config.hidden_size, self.rel_num)

        self.layer = DecoderLayer(config) # Global Feature Mining Module

        self.U = nn.Parameter(torch.FloatTensor(config.rel_num, config.hidden_size+1, config.hidden_size+1)) # +1 meab bias

    def forward(self, ids, masks=None):
        h = self.bert(ids, attention_mask=masks)[0] # bsz, L, d
        h = self.dropout(h)
        bsz, L, d = h.size()

        masks_ = masks.unsqueeze(-1) # bsz, L, 1
        mask_3d = torch.einsum('bij, bkl -> bil', masks_, masks_.permute(0,2,1)) # bsz, L, L
        mask_3d = mask_3d.unsqueeze(-1).repeat(1, 1, 1, self.rel_num) # bsz, L, L, |R|

        sub_h, obj_h = h, h
        if self.args.use_feature_enchanced:
            sub_h = self.feat_enhance_sub(sub_h)
            obj_h = self.feat_enhance_obj(obj_h)

        for i in range(self.rounds):
            
            if self.args.set_table_calc == 'mul':
                t = self.elu(sub_h.unsqueeze(2).repeat(1, 1, L, 1) * obj_h.unsqueeze(1).repeat(1, L, 1, 1)) # bsz, L, L, d
                table_logist = self.build_table(t) # bsz, L, L, |R|

            elif self.args.set_table_calc == 'biaffine':
                sub_extend = torch.cat([sub_h, torch.ones_like(sub_h[..., :1])], dim=-1) # bsz, L, d+1
                obj_extend = torch.cat([obj_h, torch.ones_like(obj_h[..., :1])], dim=-1) # bsz, L, d+1
                table_logist = torch.einsum('bxi, oij, byj -> boxy', sub_extend, self.U, obj_extend) # bsz, |R|, L, L
                table_logist = table_logist.permute(0,2,3,1) # bsz, L, L, |R|
            
            # table_logist = self.relu(table_logist)
            # table_logist = self.norm(table_logist.permute(0,3,1,2)).permute(0,2,3,1)

            sub_h_t = table_logist.max(dim=2).values
            obj_h_t = table_logist.max(dim=1).values
            sub_h_t = self.fcn_sub(sub_h_t)
            obj_h_t = self.fcn_obj(obj_h_t)

            sub_h = sub_h + self.layer(sub_h_t, obj_h_t, masks)[0]
            obj_h = obj_h + self.layer(obj_h_t, sub_h_t, masks)[0]


        table = (table_logist).sigmoid() * mask_3d
        cor = self.map_cor(table_logist).sigmoid().squeeze(-1)

        if self.args.set_rel_level == 'cls_map':
            rels = self.map_rel(h[:,0,:]).sigmoid()
        elif self.args.set_rel_level == 'avg_map':
            h_avg = masked_avgpool(sub_h + obj_h, masks)
            rels = self.map_rel(h_avg).sigmoid()
        elif self.args.set_rel_level == 'maxpooling':
            rels = nn.MaxPool2d(L)((table).permute(0,3,1,2))
            rels = rels.squeeze(-1).squeeze(-1)

        if self.args.set_ent_level == 'sent':
            subs = self.map_sub(h).sigmoid()
            objs = self.map_obj(h).sigmoid()
        elif self.args.set_ent_level == 'sent_enchanced':
            subs = self.map_sub(sub_h).sigmoid()
            objs = self.map_obj(obj_h).sigmoid()
        elif self.args.set_ent_level == 'rel_maxpooling':
            subs = table.max(dim=2).values
            objs = table.max(dim=1).values

        ents = torch.cat([subs.unsqueeze(1), objs.unsqueeze(1)], dim=1)
        return table, cor, rels, ents