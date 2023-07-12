# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : parser/irnet.py
# @Software: PyCharm
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
sys.path.append("..")
from parser import nn_utils
from utils.hypothesis import Hypothesis, ActionInfo
from utils.dataset import Batch
from parser.pointer_net import PointerNet
from rule import define_rule
from parser.plm_utils import plm_encode, encode_hpu


device = torch.device("cuda")


class IRNet(nn.Module):
    def __init__(self, args, grammar):
        super(IRNet, self).__init__()
        self.args = args
        self.grammar = grammar
        self.use_column_pointer = args.column_pointer

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        input_dim = args.action_embed_size  # previous action
        input_dim += args.att_vec_size  # input feeding
        input_dim += args.type_embed_size  # pre type embedding
        lf_input_dim = input_dim

        self.lf_decoder_lstm = nn.LSTMCell(lf_input_dim, args.hidden_size)
        self.sketch_decoder_lstm = nn.LSTMCell(input_dim, args.hidden_size)

        self.plm_tokenizer = RobertaTokenizer.from_pretrained(args.plm_model, do_lower_case=True)
        self.plm_model = RobertaModel.from_pretrained(args.plm_model)
        self.plm_config = RobertaConfig.from_pretrained(args.plm_model)

        self.encoder_dim = self.plm_config.hidden_size

        self.att_sketch_linear = nn.Linear(self.encoder_dim , args.hidden_size, bias=False)
        self.att_lf_linear = nn.Linear(self.encoder_dim, args.hidden_size, bias=False)

        self.sketch_att_vec_linear = nn.Linear(self.encoder_dim + args.hidden_size , args.att_vec_size, bias=False)
        self.lf_att_vec_linear = nn.Linear(self.encoder_dim + args.hidden_size, args.att_vec_size, bias=False)

        self.sketch_begin_vec = Variable(self.new_tensor(self.sketch_decoder_lstm.input_size))
        self.lf_begin_vec = Variable(self.new_tensor(self.lf_decoder_lstm.input_size))

        self.prob_att = nn.Linear(args.att_vec_size, 1)
        self.col_type = nn.Linear(4, self.encoder_dim)

        self.production_embed = nn.Embedding(len(grammar.prod2id), args.action_embed_size)
        self.type_embed = nn.Embedding(len(grammar.type2id), args.type_embed_size)
        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(grammar.prod2id)).zero_())

        # args.readout
        self.read_out_act = F.tanh if args.readout == 'non_linear' else nn_utils.identity
        # # args.embed_size args.att_vec_size
        self.query_vec_to_action_embed = nn.Linear(args.att_vec_size, args.action_embed_size,
                                                   bias=args.readout == 'non_linear')

        self.production_readout = lambda q: F.linear(self.read_out_act(self.query_vec_to_action_embed(q)),
                                                     self.production_embed.weight, self.production_readout_b)

        self.column_rnn_input = nn.Linear(self.encoder_dim, args.action_embed_size, bias=False)
        self.table_rnn_input = nn.Linear(self.encoder_dim, args.action_embed_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)

        self.column_pointer_net = PointerNet(args.hidden_size, self.encoder_dim, attention_type=args.column_att)

        self.table_pointer_net = PointerNet(args.hidden_size, self.encoder_dim, attention_type=args.column_att)

        # initial the embedding layers
        nn.init.xavier_normal_(self.production_embed.weight.data)
        nn.init.xavier_normal_(self.type_embed.weight.data)

    def forward(self, examples):

        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        q_enc, col_enc, q_mask = self.encode(batch, batch.src_sents_word, batch.table_sents_word)
        _, table_enc, _ = self.encode(batch, batch.src_sents_word, batch.schema_sents_word)

        q_enc = self.dropout(q_enc)

        q_enc_sketch = self.att_sketch_linear(q_enc)
        q_enc_lf = self.att_lf_linear(q_enc)
        col_enc = col_enc + F.relu(self.col_type(self.input_type(batch.col_hot_type)))

        loss_sketch, sketch_att = self.sketch_decoding(examples, batch, q_enc, q_enc_sketch, q_mask)
        loss_lf, lf_att = self.lf_decoding(examples, batch, q_enc, q_enc_lf, q_mask, col_enc, table_enc)

        return [loss_sketch, loss_lf]

    def parse(self, examples, beam_size=5):
        bs = len(examples)
        batch = Batch(examples, self.grammar, cuda=self.args.cuda)

        q_enc, col_enc, q_mask = self.encode(batch, batch.src_sents_word, batch.table_sents_word)

        _, table_enc, _ = self.encode(batch, batch.src_sents_word, batch.schema_sents_word)

        q_enc = self.dropout(q_enc)

        q_enc_sketch = self.att_sketch_linear(q_enc)
        q_enc_lf = self.att_lf_linear(q_enc)
        col_enc = col_enc + F.relu(self.col_type(self.input_type(batch.col_hot_type)))

        sketch_completed_beams = self.parse_sketch_with_beam(batch=batch,
                                                             q_enc=q_enc,
                                                             q_enc_sketch=q_enc_sketch,
                                                             q_mask=q_mask,
                                                             beam_size=beam_size)

        sketch_actions = [[] for _ in range(bs)]
        for sid in range(bs):
            if len(sketch_completed_beams[sid]) > 0:
                sketch_actions[sid] = [x for x in sketch_completed_beams[sid][0].actions]

        lf_completed_beams = self.parse_lf_with_beam(batch=batch,
                                                     q_enc=q_enc,
                                                     q_enc_lf=q_enc_lf,
                                                     q_mask=q_mask,
                                                     col_enc=col_enc,
                                                     table_enc=table_enc,
                                                     sketch_actions=sketch_actions,
                                                     beam_size=beam_size)

        return lf_completed_beams, sketch_actions

    def sketch_decoding_step(self, t, h_last, att_last, q_enc, q_enc_sketch, q_mask,
                             batch, tgt_actions, zero_action_embed, zero_type_embed, inference=False):

        if t == 0:
            x = self.new_tensor(len(batch), self.sketch_decoder_lstm.input_size).zero_()
        else:
            action_embeds = []
            prev_types = []
            for bid, tgt_action in enumerate(tgt_actions):
                if not inference:
                    if t < len(tgt_action):
                        # get the last action
                        # This is the action embedding
                        action_last = tgt_action[t - 1]
                        if type(action_last) in [define_rule.Root1,
                                                 define_rule.Root,
                                                 define_rule.Sel,
                                                 define_rule.Filter,
                                                 define_rule.Sup,
                                                 define_rule.N,
                                                 define_rule.Order]:
                            action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                        else:
                            action_embed = zero_action_embed
                    else:
                        action_embed = zero_action_embed
                else:
                    action_last = tgt_action[-1]
                    if type(action_last) in [define_rule.Root1,
                                             define_rule.Root,
                                             define_rule.Sel,
                                             define_rule.Filter,
                                             define_rule.Sup,
                                             define_rule.N,
                                             define_rule.Order]:
                        action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                    else:
                        raise ValueError('unknown action %s' % action_last)

                action_embeds.append(action_embed)

            action_embeds = torch.stack(action_embeds)
            # print("action_embeds:", action_embeds[0].tolist()[:5])
            inputs = [action_embeds]

            for bid, tgt_action in enumerate(tgt_actions):
                if not inference:
                    if t < len(tgt_action):
                        action_last = tgt_action[t - 1]
                        prev_type = self.type_embed.weight[self.grammar.type2id[type(action_last)]]
                    else:
                        prev_type = zero_type_embed
                else:
                    action_last = tgt_action[t - 1]
                    prev_type = self.type_embed.weight[self.grammar.type2id[type(action_last)]]

                prev_types.append(prev_type)

            prev_types = torch.stack(prev_types)

            inputs.append(att_last)
            inputs.append(prev_types)
            x = torch.cat(inputs, dim=-1)

        (h_t, cell_t), att_t, aw = self.decode_step(x=x,
                                                    h_last=h_last,
                                                    src_enc=q_enc,
                                                    src_enc_att_linear=q_enc_sketch,
                                                    decoder=self.sketch_decoder_lstm,
                                                    attention_func=self.sketch_att_vec_linear,
                                                    src_token_mask=q_mask,
                                                    return_att_weight=True)

        return (h_t, cell_t), att_t

    def lf_decoding_step(self, t, h_last, att_last, q_enc, q_enc_lf, q_mask,
                         batch, col_enc, table_enc, zero_action_embed, zero_type_embed,
                         examples=None, beams=None, n_beams=None):
        bs = len(batch)
        if t == 0:
            x = self.new_tensor(bs, self.lf_decoder_lstm.input_size).zero_()
        else:
            prev_types = []
            action_embeds = []
            if beams is None:
                assert examples is not None
                for eid, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_last = example.tgt_actions[t - 1]
                        if type(action_last) in [define_rule.Root1,
                                                 define_rule.Root,
                                                 define_rule.Sel,
                                                 define_rule.Filter,
                                                 define_rule.Sup,
                                                 define_rule.N,
                                                 define_rule.Order]:
                            action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                        else:
                            if isinstance(action_last, define_rule.C):
                                action_embed = self.column_rnn_input(col_enc[eid, action_last.id_c])
                            elif isinstance(action_last, define_rule.T):
                                action_embed = self.table_rnn_input(table_enc[eid, action_last.id_c])
                            elif isinstance(action_last, define_rule.A):
                                action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                            else:
                                action_embed = zero_action_embed
                    else:
                        action_embed = zero_action_embed
                    action_embeds.append(action_embed)
            else:
                for bid, beam in enumerate(beams):
                    action_last = beam.actions[-1]
                    if type(action_last) in [define_rule.Root1,
                                             define_rule.Root,
                                             define_rule.Sel,
                                             define_rule.Filter,
                                             define_rule.Sup,
                                             define_rule.N,
                                             define_rule.Order]:
                        action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                    elif isinstance(action_last, define_rule.C):
                        action_embed = self.column_rnn_input(col_enc[bid, action_last.id_c])
                    elif isinstance(action_last, define_rule.T):
                        action_embed = self.table_rnn_input(table_enc[bid, action_last.id_c])
                    elif isinstance(action_last, define_rule.A):
                        action_embed = self.production_embed.weight[self.grammar.prod2id[action_last.production]]
                    else:
                        raise ValueError('unknown action %s' % action_last)
                    action_embeds.append(action_embed)

            action_embeds = torch.stack(action_embeds)
            inputs = [action_embeds]

            if beams is None:
                assert examples is not None
                for eid, example in enumerate(examples):
                    if t < len(example.tgt_actions):
                        action_last = example.tgt_actions[t - 1]
                        prev_type = self.type_embed.weight[self.grammar.type2id[type(action_last)]]
                    else:
                        prev_type = zero_type_embed
                    prev_types.append(prev_type)
            else:
                for bid, beam in enumerate(beams):
                    action_last = beam.actions[-1]
                    prev_type = self.type_embed.weight[self.grammar.type2id[type(action_last)]]
                    prev_types.append(prev_type)

            prev_types = torch.stack(prev_types)

            inputs.append(att_last)
            inputs.append(prev_types)
            x = torch.cat(inputs, dim=-1)

        (h_t, cell_t), att_t, aw = self.decode_step(x=x,
                                                    h_last=h_last,
                                                    src_enc=q_enc,
                                                    src_enc_att_linear=q_enc_lf,
                                                    src_token_mask=q_mask,
                                                    decoder=self.lf_decoder_lstm,
                                                    attention_func=self.lf_att_vec_linear,
                                                    return_att_weight=True)

        return (h_t, cell_t), att_t

    def sketch_decoding(self, examples, batch, q_enc, q_enc_sketch, q_mask):

        dec_init_vec = self.new_tensor(len(examples), self.args.hidden_size).zero_(), \
                       self.new_tensor(len(examples), self.args.hidden_size).zero_()
        h_tm1 = dec_init_vec

        zero_action_embed = self.new_tensor(self.args.action_embed_size).zero_()
        zero_type_embed = self.new_tensor(self.args.type_embed_size).zero_()
        action_probs = [[] for _ in examples]

        att_tm1 = None
        sketch_att = []

        for t in range(batch.max_sketch_num):

            (h_t, cell_t), att_t = self.sketch_decoding_step(t=t,
                                                             h_last=h_tm1,
                                                             att_last=att_tm1,
                                                             q_enc=q_enc,
                                                             q_enc_sketch=q_enc_sketch,
                                                             q_mask=q_mask,
                                                             batch=batch,
                                                             tgt_actions=[x.sketch for x in examples],
                                                             zero_action_embed=zero_action_embed,
                                                             zero_type_embed=zero_type_embed)

            sketch_att.append(att_t)
            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)

            for e_id, example in enumerate(examples):
                if t < len(example.sketch):
                    action_t = example.sketch[t]
                    act_prob_t_i = apply_rule_prob[e_id, self.grammar.prod2id[action_t.production]]
                    action_probs[e_id].append(act_prob_t_i)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        sketch_probs = []
        for action_probs_i in action_probs:
            eps = self.new_tensor([1e-8 for _ in range(len(action_probs_i))])
            action_probs_i = torch.stack(action_probs_i, dim=0)
            sketch_probs.append(torch.add(action_probs_i, eps).log().sum())

        sketch_prob_var = torch.stack(sketch_probs, dim=0)
        return sketch_prob_var, sketch_att

    def lf_decoding(self, examples, batch, q_enc, q_enc_lf, q_mask, col_enc, table_enc):

        dec_init_vec = self.new_tensor(len(examples), self.args.hidden_size).zero_(), \
                       self.new_tensor(len(examples), self.args.hidden_size).zero_()
        h_last = dec_init_vec

        zero_action_embed = self.new_tensor(self.args.action_embed_size).zero_()
        zero_type_embed = self.new_tensor(self.args.type_embed_size).zero_()

        col_appear_mask = batch.col_appear_mask
        table_appear_mask = batch.table_appear_mask

        action_probs = [[] for _ in examples]
        col_enable = np.zeros(shape=(len(examples)))

        att_last = None
        lf_att = []

        for t in range(batch.max_action_num):
            (h_t, cell_t), att_t = self.lf_decoding_step(t=t,
                                                         h_last=h_last,
                                                         att_last=att_last,
                                                         q_enc=q_enc,
                                                         q_enc_lf=q_enc_lf,
                                                         q_mask=q_mask,
                                                         batch=batch,
                                                         col_enc=col_enc,
                                                         table_enc=table_enc,
                                                         zero_action_embed=zero_action_embed,
                                                         zero_type_embed=zero_type_embed,
                                                         examples=examples)

            lf_att.append(att_t)

            apply_rule_prob = F.softmax(self.production_readout(att_t), dim=-1)
            col_appear_mask_val = torch.from_numpy(col_appear_mask)
            if self.cuda:
                col_appear_mask_val = col_appear_mask_val.cuda()

            if self.use_column_pointer:
                gate = F.sigmoid(self.prob_att(att_t))
                col_weights = self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                      src_token_mask=None) * col_appear_mask_val * gate \
                              + self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                        src_token_mask=None) * (1 - col_appear_mask_val) * (1 - gate)
            else:
                col_weights = self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                      src_token_mask=None)

            table_weights = self.table_pointer_net(src_enc=table_enc, query_vec=att_t.unsqueeze(0),
                                                   src_token_mask=None)

            col_weights.data.masked_fill_(batch.col_token_mask.bool(), -float('inf'))
            table_weights.data.masked_fill_(batch.table_token_mask.expand_as(table_weights).bool(), -float('inf'))

            table_dict = [batch.col_table_dict[x_id][int(x)] for x_id, x in enumerate(col_enable.tolist())]
            table_mask = batch.table_dict_mask(table_dict)
            table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

            col_weights = F.softmax(col_weights, dim=-1)
            table_weights = F.softmax(table_weights, dim=-1)

            # now we should get the loss
            for bid, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_t = example.tgt_actions[t]
                    if isinstance(action_t, define_rule.C):
                        col_appear_mask[bid, action_t.id_c] = 1
                        col_enable[bid] = action_t.id_c
                        act_prob_t_i = col_weights[bid, action_t.id_c]
                        action_probs[bid].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.T):
                        table_appear_mask[bid, action_t.id_c] = 1
                        act_prob_t_i = table_weights[bid, action_t.id_c]
                        action_probs[bid].append(act_prob_t_i)
                    elif isinstance(action_t, define_rule.A):
                        act_prob_t_i = apply_rule_prob[bid, self.grammar.prod2id[action_t.production]]
                        action_probs[bid].append(act_prob_t_i)
                    else:
                        pass

            h_last = (h_t, cell_t)
            att_last = att_t

        lf_probs = []
        for action_probs_i in action_probs:
            eps = self.new_tensor([1e-8 for _ in range(len(action_probs_i))])
            action_probs_i = torch.stack(action_probs_i, dim=0)
            lf_probs.append(torch.add(action_probs_i, eps).log().sum())
            # lf_probs.append(action_probs_i.log().sum())
        # print("lf", lf_probs)
        loss_lf = torch.stack(lf_probs, dim=0)

        return loss_lf, lf_att

    def make_sketch_meta_entries(self, batch_size, beams, att_t):
        meta_entries = [[] for _ in range(batch_size)]
        apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

        for bid, beam in enumerate(beams):
            action_class = beam.get_availableClass()
            beam.sketch_att.append(att_t[bid])
            if action_class in [define_rule.Root1,
                                define_rule.Root,
                                define_rule.Sel,
                                define_rule.Filter,
                                define_rule.Sup,
                                define_rule.N,
                                define_rule.Order]:
                possible_productions = self.grammar.get_production(action_class)
                for possible_production in possible_productions:
                    prod_id = self.grammar.prod2id[possible_production]
                    prod_score = apply_rule_log_prob[bid, prod_id]
                    new_beam_score = beam.score + prod_score.data.cpu()
                    meta_entry = {
                        'action_type': action_class,
                        'prod_id': prod_id,
                        'score': prod_score,
                        'new_beam_score': new_beam_score,
                        'prev_beam_id': bid
                    }
                    meta_entries[beam.sid].append(meta_entry)
            else:
                raise RuntimeError('No right action class')
        return meta_entries

    def make_lf_meta_entries(self, t, batch_size, beams, batch, att_t, col_enc, table_enc,
                             col_token_mask, table_token_mask, col_appear_mask, col_enable, col_table_dict, sketch_actions):

        meta_entries = [[] for _ in range(batch_size)]

        padding_sketch_actions = padding_sketchs(sketch_actions)
        apply_rule_log_prob = F.log_softmax(self.production_readout(att_t), dim=-1)

        col_appear_mask_val = torch.from_numpy(col_appear_mask)
        if self.cuda:
            col_appear_mask_val = col_appear_mask_val.cuda()

        if self.use_column_pointer:
            gate = F.sigmoid(self.prob_att(att_t))
            col_weights = self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * col_appear_mask_val * gate + \
                          self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None) * (1 - col_appear_mask_val) * (1 - gate)
        else:
            col_weights = self.column_pointer_net(src_enc=col_enc, query_vec=att_t.unsqueeze(0),
                                                  src_token_mask=None)

        table_weights = self.table_pointer_net(src_enc=table_enc, query_vec=att_t.unsqueeze(0),
                                               src_token_mask=None)

        col_weights.data.masked_fill_(col_token_mask.bool(), -float('inf'))
        table_weights.data.masked_fill_(table_token_mask.bool(), -float('inf'))

        table_dict = [col_table_dict[x_id][int(x)] for x_id, x in enumerate(col_enable.tolist())]
        table_mask = batch.table_dict_mask(table_dict)
        table_weights.data.masked_fill_(table_mask.bool(), -float('inf'))

        col_weights = F.log_softmax(col_weights, dim=-1)
        table_weights = F.log_softmax(table_weights, dim=-1)

        for bid, beam in enumerate(beams):
            sid = beam.sid
            if t >= len(padding_sketch_actions[sid]):
                continue
            if type(padding_sketch_actions[sid][t]) == define_rule.A:
                possible_productions = self.grammar.get_production(define_rule.A)
                for possible_production in possible_productions:
                    prod_id = self.grammar.prod2id[possible_production]
                    prod_score = apply_rule_log_prob[bid, prod_id]

                    new_beam_score = beam.score + prod_score.data.cpu()
                    meta_entry = {'action_type': define_rule.A,
                                  'prod_id': prod_id,
                                  'score': prod_score,
                                  'new_beam_score': new_beam_score,
                                  'prev_beam_id': bid}
                    meta_entries[sid].append(meta_entry)
            elif type(padding_sketch_actions[sid][t]) == define_rule.C:
                for col_id, _ in enumerate(batch.table_sents[sid]):
                    col_sel_score = col_weights[bid, col_id]
                    new_beam_score = beam.score + col_sel_score.data.cpu()
                    meta_entry = {
                        'action_type': define_rule.C,
                        'col_id': col_id,
                        'score': col_sel_score,
                        'new_beam_score': new_beam_score,
                        'prev_beam_id': bid
                    }
                    meta_entries[sid].append(meta_entry)
            elif type(padding_sketch_actions[sid][t]) == define_rule.T:
                for t_id, _ in enumerate(batch.table_names[sid]):
                    t_sel_score = table_weights[bid, t_id]
                    new_beam_score = beam.score + t_sel_score.data.cpu()
                    meta_entry = {
                        'action_type': define_rule.T,
                        't_id': t_id,
                        'score': t_sel_score,
                        'new_beam_score': new_beam_score,
                        'prev_beam_id': bid
                    }
                    meta_entries[sid].append(meta_entry)
            else:
                prod_id = self.grammar.prod2id[padding_sketch_actions[sid][t].production]
                new_beam_score = beam.score + torch.tensor(0.0)
                meta_entry = {
                    'action_type': type(padding_sketch_actions[sid][t]),
                    'prod_id': prod_id,
                    'score': torch.tensor(0.0),
                    'new_beam_score': new_beam_score,
                    'prev_beam_id': bid
                }
                meta_entries[sid].append(meta_entry)
        return meta_entries

    def organize_sketch_beams(self, t, batch_size, beams, meta_entries, completed_beams, beam_size):
        new_beams = []
        live_beam_ids = []
        new_n_beams = [0 for _ in range(batch_size)]

        for sid in range(batch_size):
            if not meta_entries[sid]: continue
            new_beam_scores = torch.stack([x['new_beam_score'] for x in meta_entries[sid]], dim=0)
            top_new_beam_scores, meta_ids = torch.topk(new_beam_scores,
                                                       k=min(new_beam_scores.size(0),
                                                             beam_size - len(completed_beams[sid])))

            one_n_beams = 0
            one_new_beams = []
            one_live_beam_ids = []
            for new_beam_score, meta_id in zip(top_new_beam_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                meta_entry = meta_entries[sid][meta_id]
                prev_beam_id = meta_entry['prev_beam_id']
                prev_beam = beams[prev_beam_id]

                action_type_str = meta_entry['action_type']
                prod_id = meta_entry['prod_id']
                if prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = meta_entry['score']

                new_beam = prev_beam.clone_and_apply_action_info(action_info)
                new_beam.score = new_beam_score
                new_beam.inputs.extend(prev_beam.inputs)

                if new_beam.is_valid is False:
                    continue

                if new_beam.completed:
                    completed_beams[sid].append(new_beam)
                else:
                    one_n_beams += 1
                    one_new_beams.append(new_beam)
                    one_live_beam_ids.append(prev_beam_id)

            # if len(completed_beams[sid]) >= beam_size:
            #     new_n_beams[sid] = 0
            # else:
            #     new_n_beams[sid] = one_n_beams
            #     new_beams.extend(one_new_beams)
            #     live_beam_ids.extend(one_live_beam_ids)
            new_n_beams[sid] = one_n_beams
            new_beams.extend(one_new_beams)
            live_beam_ids.extend(one_live_beam_ids)

        return new_beams, new_n_beams, live_beam_ids

    def organize_lf_beams(self, t, batch_size, beams, meta_entries, completed_beams, beam_size):
        new_beams = []
        live_beam_ids = []
        new_n_beams = [0 for _ in range(batch_size)]

        for sid in range(batch_size):
            if not meta_entries[sid]:
                continue

            new_beam_scores = torch.stack([x['new_beam_score'] for x in meta_entries[sid]], dim=0)
            top_new_beam_scores, meta_ids = torch.topk(new_beam_scores,
                                                       k=min(new_beam_scores.size(0),
                                                             beam_size - len(completed_beams[sid])))

            one_n_beams = 0
            one_new_beams = []
            one_live_beam_ids = []
            for new_beam_score, meta_id in zip(top_new_beam_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                meta_entry = meta_entries[sid][meta_id]
                prev_beam_id = meta_entry['prev_beam_id']
                prev_beam = beams[prev_beam_id]

                action_type_str = meta_entry['action_type']
                if 'prod_id' in meta_entry:
                    prod_id = meta_entry['prod_id']
                if action_type_str == define_rule.C:
                    col_id = meta_entry['col_id']
                    action = define_rule.C(col_id)
                elif action_type_str == define_rule.T:
                    t_id = meta_entry['t_id']
                    action = define_rule.T(t_id)
                elif prod_id < len(self.grammar.id2prod):
                    production = self.grammar.id2prod[prod_id]
                    action = action_type_str(list(action_type_str._init_grammar()).index(production))
                else:
                    raise NotImplementedError

                action_info.action = action
                action_info.t = t
                action_info.score = meta_entry['score']

                new_beam = prev_beam.clone_and_apply_action_info(action_info)
                new_beam.score = new_beam_score
                new_beam.inputs.extend(prev_beam.inputs)

                if new_beam.completed:
                    completed_beams[sid].append(new_beam)
                else:
                    one_n_beams += 1
                    one_new_beams.append(new_beam)
                    one_live_beam_ids.append(prev_beam_id)

            # if len(completed_beams[sid]) >= beam_size:
            #     new_n_beams[sid] = 0
            # else:
            #     new_n_beams[sid] = one_n_beams
            #     new_beams.extend(one_new_beams)
            #     live_beam_ids.extend(one_live_beam_ids)

            new_n_beams[sid] = one_n_beams
            new_beams.extend(one_new_beams)
            live_beam_ids.extend(one_live_beam_ids)

        return new_beams, new_n_beams, live_beam_ids

    def parse_sketch_with_beam(self, batch, q_enc, q_enc_sketch, q_mask, beam_size):
        dec_init_vec = (self.new_tensor(len(batch), self.args.hidden_size).zero_(),
                        self.new_tensor(len(batch), self.args.hidden_size).zero_())

        h_last = dec_init_vec
        zero_action_embed = self.new_tensor(self.args.action_embed_size).zero_()
        zero_type_embed = self.new_tensor(self.args.type_embed_size).zero_()

        bs = q_enc.size(0)
        
        n_beams = [1 for _ in range(bs)]
        beams = [Hypothesis(sid, is_sketch=True) for sid in range(bs)]
        completed_beams = [[] for _ in range(bs)]

        t = 0
        att_last = None
        while not is_complete(completed_beams, beam_size) and t < self.args.decode_max_time_step:

            exp_q_enc = expand_tensor_by_beam_number(q_enc, n_beams)
            exp_q_enc_sketch = expand_tensor_by_beam_number(q_enc_sketch, n_beams)
            exp_q_mask = expand_tensor_by_beam_number(q_mask, n_beams)


            (h_t, cell_t), att_t = self.sketch_decoding_step(t=t,
                                                             h_last=h_last,
                                                             att_last=att_last,
                                                             q_enc=exp_q_enc,
                                                             q_enc_sketch=exp_q_enc_sketch,
                                                             q_mask=exp_q_mask,
                                                             batch=batch,
                                                             tgt_actions=[x.actions for x in beams],
                                                             zero_action_embed=zero_action_embed,
                                                             zero_type_embed=zero_type_embed,
                                                             inference=True)

            meta_entries = self.make_sketch_meta_entries(batch_size=bs,
                                                         beams=beams,
                                                         att_t=att_t)

            beams, n_beams, live_beam_ids = self.organize_sketch_beams(t=t,
                                                                       batch_size=bs,
                                                                       beams=beams,
                                                                       meta_entries=meta_entries,
                                                                       completed_beams=completed_beams,
                                                                       beam_size=beam_size)

            if not live_beam_ids:
                break

            att_last = att_t[live_beam_ids]
            h_last = (h_t[live_beam_ids], cell_t[live_beam_ids])
            t += 1

        for i in range(bs):
            completed_beams[i].sort(key=lambda beam: -beam.score)

        return completed_beams

    def parse_lf_with_beam(self, batch, q_enc, q_enc_lf, q_mask, col_enc, table_enc, sketch_actions, beam_size):

        bs = len(batch)
        zero_action_embed = self.new_tensor(self.args.action_embed_size).zero_()
        zero_type_embed = self.new_tensor(self.args.type_embed_size).zero_()

        dec_init_vec = (self.new_tensor(bs, self.args.hidden_size).zero_(),
                        self.new_tensor(bs, self.args.hidden_size).zero_())

        h_last = dec_init_vec

        n_beams = [1 for _ in range(bs)]
        beams = [Hypothesis(sid, is_sketch=False) for sid in range(bs)]
        completed_beams = [[] for _ in range(bs)]

        t = 0
        att_last = None
        while not is_complete(completed_beams, beam_size) and t < self.args.decode_max_time_step:

            exp_q_enc = expand_tensor_by_beam_number(q_enc, n_beams)
            exp_q_enc_lf = expand_tensor_by_beam_number(q_enc_lf, n_beams)
            exp_q_mask = expand_tensor_by_beam_number(q_mask, n_beams)

            exp_col_enc = expand_tensor_by_beam_number(col_enc, n_beams)
            exp_table_enc = expand_tensor_by_beam_number(table_enc, n_beams)

            exp_col_token_mask = expand_tensor_by_beam_number(batch.col_token_mask, n_beams)
            exp_table_token_mask = expand_tensor_by_beam_number(batch.table_token_mask, n_beams)

            exp_col_appear_mask, exp_col_enable, exp_col_table_dict = expand_column_mask(n_beams, batch, beams)

            # if t == 1:
            #     print("-----------------------------")
            #     print(t)
            #     print(len(n_beams))
            #     print(n_beams)
            #     print(exp_col_enc.size())
            #     print(exp_col_token_mask.size())
            #     if bs == 64:
            #         st = sum(n_beams[:31])
            #         ed = sum(n_beams[:32])
            #     else:
            #         st = sum(n_beams[:0])
            #         ed = sum(n_beams[:1])
            #
            #     print(st, ed)
            #     for idx in range(st, ed):
            #         print(idx - st, exp_col_enc[idx])
            #         print(exp_col_token_mask[idx])
            #     exit()


            (h_t, cell_t), att_t = self.lf_decoding_step(t=t,
                                                         h_last=h_last,
                                                         att_last=att_last,
                                                         q_enc=exp_q_enc,
                                                         q_enc_lf=exp_q_enc_lf,
                                                         q_mask=exp_q_mask,
                                                         batch=batch,
                                                         col_enc=exp_col_enc,
                                                         table_enc=exp_table_enc,
                                                         zero_action_embed=zero_action_embed,
                                                         zero_type_embed=zero_type_embed,
                                                         beams=beams,
                                                         n_beams=n_beams)

            meta_entries = self.make_lf_meta_entries(t=t,
                                                     batch_size=bs,
                                                     beams=beams,
                                                     batch=batch,
                                                     att_t=att_t,
                                                     col_enc=exp_col_enc,
                                                     table_enc=exp_table_enc,
                                                     col_token_mask=exp_col_token_mask,
                                                     table_token_mask=exp_table_token_mask,
                                                     col_appear_mask=exp_col_appear_mask,
                                                     col_enable=exp_col_enable,
                                                     col_table_dict=exp_col_table_dict,
                                                     sketch_actions=sketch_actions)

            beams, n_beams, live_beam_ids = self.organize_lf_beams(t=t,
                                                                   batch_size=bs,
                                                                   beams=beams,
                                                                   meta_entries=meta_entries,
                                                                   completed_beams=completed_beams,
                                                                   beam_size=beam_size)

            if not live_beam_ids:
                break

            att_last = att_t[live_beam_ids]
            h_last = (h_t[live_beam_ids], cell_t[live_beam_ids])
            t += 1

        for i in range(bs):
            completed_beams[i].sort(key=lambda beam: -beam.score)
            # print([x.score.tolist() for x in completed_beams[i]])
        return completed_beams

    def encode(self, batch, src_sents_word, table_sents_word):
        wemb_n, wemb_h, l_n, n_hs, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx = plm_encode(self.encoder_dim, self.plm_model, self.plm_tokenizer,
                                                      src_sents_word, table_sents_word,
                                                      max_seq_length=self.args.max_seq_length)

        emb = encode_hpu(wemb_h, l_hpu=l_hpu, l_hs=l_hs)
        return wemb_n, emb, batch.len_appear_mask(n_hs)

    def decode_step(self, x, h_last, src_enc, src_enc_att_linear, decoder, attention_func,
                    src_token_mask=None, return_att_weight=False):
        h_t, cell_t = decoder(x, h_last)

        ctx_t, alpha_t = nn_utils.dot_prod_attention(h_t,
                                                     src_enc,
                                                     src_enc_att_linear,
                                                     mask=src_token_mask)

        att_t = F.tanh(attention_func(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        if return_att_weight:
            return (h_t, cell_t), att_t, alpha_t
        else:
            return (h_t, cell_t), att_t

    def input_type(self, values_list):
        B = len(values_list)
        val_len = []
        for value in values_list:
            val_len.append(len(value))
        max_len = max(val_len)
        # for the Begin and End
        val_emb_array = np.zeros((B, max_len, values_list[0].shape[1]), dtype=np.float32)
        for i in range(B):
            val_emb_array[i, :val_len[i], :] = values_list[i][:, :]

        val_inp = torch.from_numpy(val_emb_array)
        if self.args.cuda:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'grammar': self.grammar,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

def padding_sketchs(sketchs):
    padding_results = []
    for bid, sketch in enumerate(sketchs):
        padding_result = []
        for action in sketch:
            padding_result.append(action)
            if type(action) == define_rule.N:
                for _ in range(action.id_c + 1):
                    padding_result.append(define_rule.A(0))
                    padding_result.append(define_rule.C(0))
                    padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Filter and 'A' in action.production:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
            elif type(action) == define_rule.Order or type(action) == define_rule.Sup:
                padding_result.append(define_rule.A(0))
                padding_result.append(define_rule.C(0))
                padding_result.append(define_rule.T(0))
        padding_results.append(padding_result)
    return padding_results

def expand_tensor_by_beam_number(src_enc, n_beams):
    bs = src_enc.size(0)
    n_dim = len(src_enc.size())
    n_exp_sz = [[n_beams[sid]] + [1 for _ in range(n_dim - 1)] for sid in range(bs)]

    exp_src_enc = []
    for sid in range(bs):
        exp_src_enc.append(src_enc[sid].repeat(n_exp_sz[sid]))
    exp_src_enc = torch.cat(exp_src_enc, dim=0)
    return exp_src_enc
    
def expand_column_mask(n_beams, batch, beams):
    col_appear_mask = np.zeros((sum(n_beams), batch.col_appear_mask.shape[1]), dtype=np.float32)
    col_enable = np.zeros(shape=(sum(n_beams)))
    for bid, beam in enumerate(beams):
        for act in beam.actions:
            if type(act) == define_rule.C:
                col_appear_mask[bid][act.id_c] = 1
                col_enable[bid] = act.id_c

    col_table_dict = []
    for i, _num in enumerate(n_beams):
        col_table_dict.extend([copy.deepcopy(batch.col_table_dict[i]) for _ in range(_num)])
    return col_appear_mask, col_enable, col_table_dict

def is_complete(completed_beams, max_beam_sz):
    count = sum([1 if len(x) == max_beam_sz else 0 for x in completed_beams])
    if count == len(completed_beams):
        return True
    else:
        return False