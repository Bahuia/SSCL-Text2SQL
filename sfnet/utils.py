# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/utils.py
# @Software: PyCharm
"""
import sys

import tqdm

sys.path.append("..")
import os
import torch
import copy
import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from utils.dataset import Example
from rule.define_rule import *
from parser.plm_utils import generate_plm_inputs
from sfnet.args import init_arg_parser
from sfnet.dist_metric import Metric_Processor


AGG_OPS = ['none', 'max', 'min', 'count', 'sum', 'avg']
wordnet_lemmatizer = WordNetLemmatizer()
args = init_arg_parser()

tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=True)
mp = Metric_Processor(tokenizer)

class GloveHelper(object):
    def __init__(self, glove_file, embedding_size=100):
        self.glove_file = glove_file
        embeds = np.zeros((5000, embedding_size), dtype='float32')
        for i, (word, embed) in enumerate(self.embeddings):
            if i == 5000: break
            embeds[i] = embed

        self.mean = np.mean(embeds)
        self.std = np.std(embeds)

    @property
    def embeddings(self):
        with open(self.glove_file, 'r', encoding='utf8') as f:
            for line in f:
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]])
                yield word, embed

    def emulate_embeddings(self, shape):
        samples = np.random.normal(self.mean, self.std, size=shape)

        return samples

    def get_weights(self, vocab, embedding_size):
        word_ids = set(range(len(vocab.source)))
        numpy_embed = np.zeros(shape=(len(vocab.source), embedding_size))
        for word, embed in self.embeddings:
            if word in vocab.source:
                word_id = vocab.source[word]
                word_ids.remove(word_id)
                numpy_embed[word_id] = embed
        word_ids = list(word_ids)
        numpy_embed[word_ids] = self.emulate_embeddings(shape=(len(word_ids), embedding_size))
        return numpy_embed

    def load_to(self, embed_layer, vocab, trainable=True):

        word_ids = set(range(embed_layer.num_embeddings))
        numpy_embed = np.zeros(shape=(embed_layer.weight.shape[0], embed_layer.weight.shape[1]))
        for word, embed in self.embeddings:
            if word in vocab:
                word_id = vocab[word]
                word_ids.remove(word_id)
                numpy_embed[word_id] = embed

        word_ids = list(word_ids)
        numpy_embed[word_ids] = self.emulate_embeddings(shape=(len(word_ids), embed_layer.embedding_dim))
        embed_layer.weight.data.copy_(torch.from_numpy(numpy_embed))
        embed_layer.weight.requires_grad = trainable

    @property
    def words(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                yield tokens[0]

def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x

def load_dataset_new_for_cl(sql_path, table_data):
    sql_data = []
    # print("Loading data from %s" % sql_path)
    with open(sql_path) as inf:
        data = lower_keys(json.load(inf))
        sql_data += data

    sql_data_new, table_data_new = process(sql_data, table_data)  # comment out if not on full dataset

    schemas = {}
    for tab in table_data:
        schemas[tab['db_id']] = tab

    return to_examples(sql_data_new, table_data_new, schemas)

def load_sscl_task_stream(task_path, task_num=10, calc_metric_vec=False):
    task_list = []
    total_examples = []
    for i in tqdm.tqdm(range(task_num), desc=f'loading stream of {task_num} tasks'):
        table_path = os.path.join(task_path.format(i, "tables.json"))
        train_path = os.path.join(task_path.format(i, "train_irnet.json"))
        semi_path = os.path.join(task_path.format(i, "semi_irnet.json"))
        dev_path = os.path.join(task_path.format(i, "dev_irnet.json"))
        test_path = os.path.join(task_path.format(i, "test_irnet.json"))
        with open(table_path) as f:
            table_data = json.load(f)

        train_examples = load_dataset_new_for_cl(train_path, table_data)
        semi_examples = load_dataset_new_for_cl(semi_path, table_data)
        dev_examples = load_dataset_new_for_cl(dev_path, table_data)
        test_examples = load_dataset_new_for_cl(test_path, table_data)

        total_examples.extend(train_examples)
        total_examples.extend(semi_examples)
        total_examples.extend(dev_examples)
        total_examples.extend(test_examples)
        task_list.append({"train": train_examples, "semi": semi_examples, "dev": dev_examples, "test": test_examples})

    mp.get_all_schema_tokens(total_examples)
    if calc_metric_vec:
        for i in range(task_num):
            task_list[i]["train"] = mp.get_metric_info(task_list[i]["train"])
            task_list[i]["semi"] = mp.get_metric_info(task_list[i]["semi"])
            task_list[i]["dev"] = mp.get_metric_info(task_list[i]["dev"])
            task_list[i]["test"] = mp.get_metric_info(task_list[i]["test"])
    return task_list

def save_args(args, path):
    with open(path, "w") as f:
        f.write(json.dumps(vars(args), indent=4))

def process(sql_data, table_data):
    output_tab = {}
    tables = {}
    tabel_name = set()
    remove_list = list()

    for i in range(len(table_data)):
        table = table_data[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['schema_len'] = []
        length = {}
        for col_tup in temp['col_map']:
            length[col_tup[0]] = length.get(col_tup[0], 0) + 1
        for l_id in range(len(length)):
            temp['schema_len'].append(length[l_id-1])
        temp['foreign_keys'] = table['foreign_keys']
        temp['primary_keys'] = table['primary_keys']
        temp['table_names'] = table['table_names']
        temp['column_types'] = table['column_types']
        db_name = table['db_id']
        tabel_name.add(db_name)
        output_tab[db_name] = temp
        tables[db_name] = table
    output_sql = []
    for i in range(len(sql_data)):

        sql = sql_data[i]
        sql_temp = {}
        sql["question_toks"] = [x for x in sql["question_toks"] if x]

        # add query metadata
        for key, value in sql.items():
            sql_temp[key] = value
        sql_temp['question'] = sql['question']
        sql_temp['question_tok'] = [wordnet_lemmatizer.lemmatize(x).lower() for x in sql['question_toks'] if x not in remove_list]
        sql_temp['rule_label'] = sql['rule_label']
        sql_temp['col_set'] = sql['col_set']
        sql_temp['query'] = sql['query']
        # dre_file.write(sql['query'] + '\n')
        sql_temp['query_tok'] = sql['query_toks']
        sql_temp['table_id'] = sql['db_id']
        table = tables[sql['db_id']]
        sql_temp['col_org'] = table['column_names_original']
        sql_temp['table_org'] = table['table_names_original']
        sql_temp['table_names'] = table['table_names'] if table['table_names'] and table['table_names'][0] else ["NONE"]
        sql_temp['fk_info'] = table['foreign_keys']
        tab_cols = [col[1] for col in table['column_names']]
        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
        sql_temp['col_iter'] = col_iter
        # process agg/sel
        sql_temp['agg'] = []
        sql_temp['sel'] = []
        gt_sel = sql['sql']['select'][1]
        if len(gt_sel) > 3:
            gt_sel = gt_sel[:3]
        for tup in gt_sel:
            sql_temp['agg'].append(tup[0])
            sql_temp['sel'].append(tup[1][1][1]) #GOLD for sel and agg

        # process where conditions and conjuctions
        sql_temp['cond'] = []
        gt_cond = sql['sql']['where']
        if len(gt_cond) > 0:
            conds = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 0]
            for cond in conds:
                curr_cond = []
                curr_cond.append(cond[2][1][1])
                curr_cond.append(cond[1])
                if cond[4] is not None:
                    curr_cond.append([cond[3], cond[4]])
                else:
                    curr_cond.append(cond[3])
                sql_temp['cond'].append(curr_cond) #GOLD for COND [[col, op],[]]

        sql_temp['conj'] = [gt_cond[x] for x in range(len(gt_cond)) if x % 2 == 1]

        # process group by / having
        sql_temp['group'] = [x[1] for x in sql['sql']['groupby']] #assume only one groupby
        having_cond = []
        if len(sql['sql']['having']) > 0:
            gt_having = sql['sql']['having'][0] # currently only do first having condition
            having_cond.append([gt_having[2][1][0]]) # aggregator
            having_cond.append([gt_having[2][1][1]]) # column
            having_cond.append([gt_having[1]]) # operator
            if gt_having[4] is not None:
                having_cond.append([gt_having[3], gt_having[4]])
            else:
                having_cond.append(gt_having[3])
        else:
            having_cond = [[], [], []]
        sql_temp['group'].append(having_cond) #GOLD for GROUP [[col1, col2, [agg, col, op]], [col, []]]

        # process order by / limit
        order_aggs = []
        order_cols = []
        sql_temp['order'] = []
        order_par = 4
        gt_order = sql['sql']['orderby']
        limit = sql['sql']['limit']
        if len(gt_order) > 0:
            order_aggs = [x[1][0] for x in gt_order[1][:1]] # limit to 1 order by
            order_cols = [x[1][1] for x in gt_order[1][:1]]
            if limit != None:
                if gt_order[0] == 'asc':
                    order_par = 0
                else:
                    order_par = 1
            else:
                if gt_order[0] == 'asc':
                    order_par = 2
                else:
                    order_par = 3

        sql_temp['order'] = [order_aggs, order_cols, order_par] #GOLD for ORDER [[[agg], [col], [dat]], []]

        # process intersect/except/union
        sql_temp['special'] = 0
        if sql['sql']['intersect'] is not None:
            sql_temp['special'] = 1
        elif sql['sql']['except'] is not None:
            sql_temp['special'] = 2
        elif sql['sql']['union'] is not None:
            sql_temp['special'] = 3

        if 'stanford_tokenized' in sql:
            sql_temp['stanford_tokenized'] = sql['stanford_tokenized']
        if 'stanford_pos' in sql:
            sql_temp['stanford_pos'] = sql['stanford_pos']
        if 'stanford_dependencies' in sql:
            sql_temp['stanford_dependencies'] = sql['stanford_dependencies']
        if 'hardness' in sql:
            sql_temp['hardness'] = sql['hardness']
        if 'question_labels' in sql:
            sql_temp['question_labels'] = sql['question_labels']

        output_sql.append(sql_temp)
    return output_sql, output_tab

def to_examples(sql_data, table_data, schemas):
    """
    :param sql_data:
    :param table_data:
    :param idxes:
    :param st:
    :param ed:
    :param schemas:
    :return:
    """
    examples = []
    col_org_seq = []
    schema_seq = []

    # file = codecs.open('./type.txt', 'w', encoding='utf-8')
    for i in range(len(sql_data)):
        sql = sql_data[i]
        table = table_data[sql['table_id']]
        origin_sql = sql['question_toks']
        table_names = table['table_names']
        col_org_seq.append(sql['col_org'])
        schema_seq.append(schemas[sql['table_id']])
        tab_cols = [col[1] for col in table['col_map']]
        tab_ids = [col[0] for col in table['col_map']]

        q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]

        col_set = sql['col_set']

        col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]

        question_arg = copy.deepcopy(sql['question_arg'])
        col_set_type = np.zeros((len(col_set_iter), 4))


        for c_id, col_ in enumerate(col_set_iter):

            for q_id, ori in enumerate(q_iter_small):
                if ori in col_:
                    col_set_type[c_id][0] += 1


        question_arg_type = sql['question_arg_type']
        one_hot_type = np.zeros((len(question_arg_type), 6))

        another_result = []
        for count_q, t_q in enumerate(question_arg_type):
            t = t_q[0]
            if t == 'NONE':
                continue
            elif t == 'table':
                one_hot_type[count_q][0] = 1
                question_arg[count_q] = ['table'] + question_arg[count_q]
            elif t == 'col':
                one_hot_type[count_q][1] = 1
                try:
                    col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                    question_arg[count_q] = ['column'] + question_arg[count_q]

                except:
                    # print(col_set_iter, question_arg[count_q])
                    # raise RuntimeError("not in col set")
                    question_arg[count_q] = "NONE"

            elif t == 'agg':
                one_hot_type[count_q][2] = 1
            elif t == 'MORE':
                one_hot_type[count_q][3] = 1

            elif t == 'MOST':
                one_hot_type[count_q][4] = 1

            elif t == 'value':
                one_hot_type[count_q][5] = 1
                question_arg[count_q] = ['value'] + question_arg[count_q]
            else:
                if len(t_q) == 1:
                    for col_probase in t_q:
                        if col_probase == 'asd':
                            continue
                        try:
                            col_set_type[sql['col_set'].index(col_probase)][2] = 5
                            question_arg[count_q] = ['value'] + question_arg[count_q]

                        except:
                            print(sql['col_set'], col_probase)
                            raise RuntimeError('not in col')
                        one_hot_type[count_q][5] = 1
                        another_result.append(sql['col_set'].index(col_probase))
                else:
                    for col_probase in t_q:
                        if col_probase == 'asd':
                            continue
                        col_set_type[sql['col_set'].index(col_probase)][3] += 1

        col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]

        table_dict = {}
        for c_id, c_v in enumerate(col_set):
            for cor_id, cor_val in enumerate(tab_cols):
                if c_v == cor_val:
                    table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

        col_table_dict = {}
        for key_item, value_item in table_dict.items():
            for value in value_item:
                col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
        col_table_dict[0] = [x for x in range(len(table_dict) - 1)]

        for j in range(len(col_set_iter)):
            if (len(col_set_iter[j]) == 1 and col_set_iter[j][0] == "") or len(col_set_iter[j]) == 0:
                col_set_iter[j] = ["NONE"]

        table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in table_names]
        for j in range(len(table_names)):
            if (len(table_names[j]) == 1 and table_names[j][0] == "") or len(table_names[j]) == 0:
                table_names[j] = ["NONE"]

        q_iter_small = [wordnet_lemmatizer.lemmatize(x) for x in origin_sql]

        table_set_type = np.zeros((len(table_names), 1))

        for c_id, col_ in enumerate(table_names):
            if " ".join(col_) in q_iter_small or " ".join(col_) in " ".join(q_iter_small):
                table_set_type[c_id][0] = 5
                continue
            for q_id, ori in enumerate(q_iter_small):
                if ori in col_:
                    table_set_type[c_id][0] += 1
                    # col_hot_type[c_id][6] = q_id + 1

        try:
            rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]
        except:
            continue

        flag = False
        for r_id, rule in enumerate(rule_label):
            if type(rule) == C:
                try:
                    assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
                except:
                    flag = True
                    # print(sql['question'])
        if flag:
            continue

        table_col_name = get_table_colNames(tab_ids, col_iter)

        pos_tags = None
        entities = None
        dependency_graph_adjacency_matrix = None

        col_labels = [i for i in range(len(sql["col_set"]))]

        col_set_iter_remove = [col_set_iter[x] for x in sorted(col_labels)]
        remove_dict = {}
        for k_i, k in enumerate(sorted(col_labels)):
            remove_dict[k] = k_i

        for label in rule_label:
            if type(label) == C:
                if label.id_c not in remove_dict:
                    remove_dict[label.id_c] = len(remove_dict)
                label.id_c = remove_dict[label.id_c]

        col_set_iter = col_set_iter_remove

        remove_dict_reverse = {}
        for k, v in remove_dict.items():
            remove_dict_reverse[v] = k

        enable_table = []
        col_table_dict_remove = {}
        for k, v in col_table_dict.items():
            if k in remove_dict:
                col_table_dict_remove[remove_dict[k]] = v
                enable_table += v

        col_table_dict = col_table_dict_remove

        col_set_type_remove = np.zeros((len(col_set_iter), 4))
        for k, v in remove_dict.items():
            if v < len(col_set_iter_remove):
                col_set_type_remove[v] = col_set_type[k]

        col_set_type = col_set_type_remove

        tokens1, segment_ids1, i_nlu1, i_hds1 = generate_plm_inputs(tokenizer, [" ".join(x) for x in question_arg],
                                                                    [" ".join(x) for x in col_set_iter])

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        if len(input_ids1) > args.max_seq_length:
            continue

        example = Example(
            src_sent=[" ".join(x) for x in question_arg],
            # src_sent=origin_sql,
            col_num=len(col_set_iter),
            vis_seq=(sql['question'], col_set_iter, sql['query']),
            tab_cols=col_set_iter,
            tgt_actions=rule_label,
            sql=sql['query'],
            tab_ids=sql['table_id'],
            sql_toks=sql['query_toks'],
            one_hot_type=one_hot_type,
            col_hot_type=col_set_type,
            schema_len=table['schema_len'],
            table_names=table_names if table_names and table_names[0] else ["NONE"],
            table_len=len(table_names),
            col_table_dict=col_table_dict,
            table_set_type=table_set_type,
            table_col_name=table_col_name,
            table_col_len=len(table_col_name),
            is_sketch=False,
            pos_tags=pos_tags,
            dependency_adjacency=dependency_graph_adjacency_matrix,
            entities=entities,
            sketch_adjacency_matrix=None

        )
        example.remove_dict_reverse = remove_dict_reverse
        example.sql_json=copy.deepcopy(sql)
        examples.append(example)

    return examples

def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result

def calc_beam_acc(beam_search_result, example):
    results = list()
    beam_result = False
    truth = " ".join([str(x) for x in example.tgt_actions]).strip()
    for bs in beam_search_result:
        pred = " ".join([str(x) for x in bs.actions]).strip()
        if truth == pred:
            results.append(True)
            beam_result = True
        else:
            results.append(False)
    return results[0], beam_result
