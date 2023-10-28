# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/args.py
# @Software: PyCharm
"""
import argparse
import torch
import random
import numpy as np


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')

    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--accumulation_step', default=4, type=int, help='Gradient Accumulation')

    arg_parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')

    arg_parser.add_argument('--plm_model', default='bert-base-uncased', type=str, help='plm_model')

    arg_parser.add_argument('--action_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=32, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')

    # readout layer
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')

    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    arg_parser.add_argument('--plm_lr', default=2e-5, type=float, help='learning rate of PLM')

    arg_parser.add_argument('--epoch_num', default=50, type=int, help='Maximum Epoch')

    arg_parser.add_argument('--task_num', type=int, default=10)
    arg_parser.add_argument('--task_path', type=str, default="../data/spider_task_stream/task_{}/{}")
    arg_parser.add_argument('--memory_size', type=int, default=20)
    arg_parser.add_argument('--candidate_size', type=int, default=50)
    arg_parser.add_argument('--max_patience', type=int, default=10)
    arg_parser.add_argument('--device', type=str, default="0")

    arg_parser.add_argument('--max_seq_length', type=int, default=300)
    arg_parser.add_argument('--warm_boot_epoch', type=int, default=15)
    arg_parser.add_argument('--student_cl', action="store_true")
    arg_parser.add_argument('--teacher_cl', action="store_true")
    arg_parser.add_argument('--agent', type=str, default="naive")
    arg_parser.add_argument('--eval_epoch', type=int, default=20)
    arg_parser.add_argument('--k_mediod_max_epoch', type=int, default=20)
    arg_parser.add_argument('--st_epoch_num', type=int, default=10)
    arg_parser.add_argument('--st_rate', type=float, default=0.3)
    arg_parser.add_argument('--student_sampling_name', type=str, default="none")
    arg_parser.add_argument('--teacher_sampling_name', type=str, default="none")
    arg_parser.add_argument('--without_student_sql', action="store_true")
    arg_parser.add_argument('--without_teacher_sql', action="store_true")

    args = arg_parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    return args