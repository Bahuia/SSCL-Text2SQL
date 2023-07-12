# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : utils/dataset.py
# @Software: PyCharm
"""
import copy
from parser import nn_utils
from rule import define_rule


class Example:
    def __init__(self, src_sent, tgt_actions, vis_seq=None, tab_cols=None, col_num=None, sql=None, sql_toks=None,
                 one_hot_type=None, col_hot_type=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None, cols_id=None, cols_set=None,
                 table_set_type=None, table_col_name=None, table_col_len=None, is_sketch=False, pos_tags=None,
                 dependency_adjacency=None, entities=None, col_pred=None, sketch_adjacency_matrix=None, sketch=None,
                 sql_wordbag_vec=None, schema_wordbag_vec=None):
        self.src_sent = src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.sql = sql
        self.sql_toks = sql_toks
        self.one_hot_type=one_hot_type
        self.col_hot_type = col_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.cols_id = cols_id
        self.cols_set = cols_set
        self.table_set_type = table_set_type
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.pos_tags = pos_tags
        self.entities = entities
        self.col_pred = col_pred
        self.dependency_adjacency = dependency_adjacency
        self.sketch_adjacency_matrix = sketch_adjacency_matrix
        self.sql_wordbag_vec = sql_wordbag_vec
        self.schema_wordbag_vec = schema_wordbag_vec
        self.pseudo_conf = 1.0

        self.truth_actions = copy.deepcopy(tgt_actions)
        if is_sketch:
            self.tgt_actions = list()
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A):
                    continue
                self.tgt_actions.append(ta)
        else:
            self.tgt_actions = list()
            for ta in self.truth_actions:
                self.tgt_actions.append(ta)
        if sketch:
            self.sketch = sketch
        else:
            self.sketch = list()
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.C) or isinstance(ta, define_rule.T) or isinstance(ta, define_rule.A):
                    continue
                self.sketch.append(ta)

        self.parent_actions = list()
        self.parent_actions_idx = list()
        for idx, t_action in enumerate(self.tgt_actions):
            if idx > 0 and self.tgt_actions[idx - 1] == t_action.parent:
                self.parent_actions.append(None)
                self.parent_actions_idx.append(None)
            else:
                self.parent_actions.append(t_action.parent)
                self.parent_actions_idx.append(
                    self.sketch.index(t_action.parent) if t_action.parent is not None else None)


class cached_property(object):

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar, cuda=False):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
        self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]

        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.schema_len = [e.schema_len for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_set_type = [e.table_set_type for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.sketch_adjacency_matrix = [e.sketch_adjacency_matrix for e in examples]
        self.sketches = [e.sketch for e in examples]

        self.pos_tags = [e.pos_tags for e in examples]
        self.entities = [e.entities for e in examples]
        self.dependency_adjacency = [e.dependency_adjacency for e in examples]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    def len_appear_mask(self, length):
        return nn_utils.length_array_to_mask_tensor(length, cuda=self.cuda)

    @property
    def col_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def col_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)
