import torch
import os
import sys
sys.path.append("..")
from transformers import BertTokenizer, BertModel

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


    # def get_sql_template(self, sql_tokens):
    #     new_tokens = []
    #     for tok in sql_tokens:
    #         if tok in self.total_key_word_dict:
    #             new_tokens.append(tok)
    #         else:
    #             new_tokens.append(self.mask_token)
    #     return " ".join(new_tokens)
    #
    #
    # def bert_embedding(self, header_list):
    #     input_token = ["[CLS]"]
    #     for header in header_list:
    #         sub_tokens = self.bert_tokenizer.tokenize(header)
    #         input_token.extend(sub_tokens)
    #         input_token.append("[SEP]")
    #
    #     input_ids = self.bert_tokenizer.convert_tokens_to_ids(input_token)
    #     input_mask = [1] * len(input_ids)
    #     # input_segment = [0] * len(input_ids)
    #
    #     input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    #     input_mask_tensor = torch.tensor([input_mask], dtype=torch.long)
    #     if self.device != -1:
    #         input_ids_tensor = input_ids_tensor.to(self.device)
    #         input_mask_tensor = input_mask_tensor.to(self.device)
    #
    #     # input_segment_tensor = torch.tensor(input_segment, dtype=torch.long)
    #
    #     all_layers_bert_enc, pooling_output = self.bert_model(input_ids=input_ids_tensor,
    #                                                      attention_mask=input_mask_tensor,
    #                                                      token_type_ids=None,
    #                                                      return_dict=False)
    #     return pooling_output