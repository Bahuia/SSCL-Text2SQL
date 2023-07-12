# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/dist_metric.py
# @Software: PyCharm
"""

class Metric_Processor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.key_word_list = ["select", "from", "where", "distinct", "group", "having", "order", "by", "join", "as", "on",
                         "limit", "not", "in", "intersect", "union", "except", "desc", "asc"]
        self.op_list = ["<", ">", "=", "<=", ">=", "!=", "like", "between"]
        self.agg_list = ["max", "avg", "min", "count", "sum"]
        self.other_list = ["(", ")", "*", ","]

        self.total_key_word = []
        self.total_key_word.extend(self.key_word_list)
        self.total_key_word.extend(self.op_list)
        self.total_key_word.extend(self.agg_list)
        self.total_key_word.extend(self.other_list)
        self.total_key_word_dict = {}
        for i, word in enumerate(self.total_key_word):
            self.total_key_word_dict[word] = i

        self.mask_token = "$"
        self.all_schema_tokens_dict = {}

    def get_metric_info(self, examples):
        for example in examples:
            sql_tokens = example.sql_toks
            sql_vector = self.get_sql_wordbag(sql_tokens)
            schema = [tc[0] for tc in example.tab_cols]
            schema_vector = self.get_schema_wordbag(schema)
            example.sql_wordbag_vec = sql_vector
            example.schema_wordbag_vec = schema_vector
        return examples

    def get_sql_wordbag(self, sql_tokens):
        sql_wordbag_vec = [0 for i in range(len(self.total_key_word))]

        for tok in sql_tokens:
            tok = tok.lower()
            if tok in self.total_key_word_dict:
                sql_wordbag_vec[self.total_key_word_dict[tok]] += 1
        return sql_wordbag_vec

    def get_schema_wordbag(self, schema):
        schema_wordbag_vec = [0 for i in range(len(self.all_schema_tokens_dict))]
        schema_str = " ".join(schema)
        tokens = self.tokenizer.tokenize(schema_str)
        for tok in tokens:
            tok = tok.lower()
            if tok in self.all_schema_tokens_dict:
                schema_wordbag_vec[self.all_schema_tokens_dict[tok]] += 1
        return schema_wordbag_vec

    def get_all_schema_tokens(self, examples):
        all_schema_tokens = []
        for example in examples:
            schema_str = " ".join(tc[0] for tc in example.tab_cols)
            tokens = self.tokenizer.tokenize(schema_str)
            for tok in tokens:
                tok = tok.lower()
                if tok in all_schema_tokens:
                    all_schema_tokens.append(tok)
        self.all_schema_tokens_dict = {k: i for i, k in enumerate(all_schema_tokens)}
