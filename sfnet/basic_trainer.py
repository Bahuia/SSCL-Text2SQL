# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/basic_trainer.py
# @Software: PyCharm
"""

import os
import sys
import tqdm
import torch
import random
sys.path.append("..")
from rule.define_rule import C
from rule import define_rule
from sfnet.task_controller import TaskController
from parser.irnet import IRNet
from sfnet.utils import save_args, calc_beam_acc


class BasicTrainer(object):
    def __init__(self, args, model_save_path):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.grammar = define_rule.Grammar(is_sketch=None)

        self.model = IRNet(args, self.grammar).to(self.device)

        optimizer_grouped_parameters = self.make_optimizer_groups(self.model)
        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters)

        self.task_controller = TaskController(args)
        self.args.task_num = len(self.task_controller.task_list)

        self.first_acc_list = [-1 for i in range(self.args.task_num)]
        self.bwt_list = [float("-inf") for i in range(self.args.task_num)]
        self.avg_acc_list = [-1 for i in range(self.args.task_num)]
        self.whole_acc_list = [-1 for i in range(self.args.task_num)]
        self.temp_fwt = [float("-inf") for i in range(self.args.task_num)]
        self.fwt_list = [float("-inf") for i in range(self.args.task_num)]
        self.acc_rand_list = [0.0 for i in range(self.args.task_num)]

        self.model_save_path = model_save_path
        self.log_path = os.path.join(self.model_save_path, 'log')
        save_args(args, os.path.join(self.model_save_path, "config.json"))
        os.makedirs(self.log_path, exist_ok=True)

    def make_optimizer_groups(self, model):
        param_optimizer = list(model.named_parameters())

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in ['plm_model'])], 'lr': self.args.lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['plm_model'])], 'lr': self.args.lr * 0.1}
        ]
        return optimizer_grouped_parameters

    def save(self, model, name="model.bin"):
        torch.save({"model": model.state_dict()}, open(os.path.join(self.model_save_path, name), 'wb'))

    def load(self, model, name="model.bin"):
        model.load_state_dict(torch.load(open(os.path.join(self.model_save_path, name), "rb"), map_location=self.device)["model"])

    def train_one_batch(self, examples, model, report_loss, example_num):
        score = model.forward(examples)
        loss_sketch = -score[0]
        loss_lf = -score[1]

        _loss = torch.sum(loss_sketch).data.item() + torch.sum(loss_lf).data.item()

        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)

        loss = loss_lf + loss_sketch

        report_loss += _loss
        example_num += len(examples)
        return report_loss, example_num, loss

    def train_one_epoch(self, examples, model, optimizer, optimize_step=True):
        model.train()

        random.shuffle(examples)
        report_loss = 0.0
        st = 0
        cnt = 0
        example_num = 0

        optimizer.zero_grad()
        while st < len(examples):
            ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

            report_loss, example_num, loss = self.train_one_batch(examples[st:ed],
                                                                  model,
                                                                  report_loss,
                                                                  example_num)
            loss.backward()

            if (cnt + 1) % self.args.accumulation_step == 0 or ed == len(examples):
                if self.args.clip_grad > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad)
                if optimize_step:
                    optimizer.step()
                    optimizer.zero_grad()

            st = ed
            cnt += 1
        return report_loss / len(examples)

    def epoch_acc(self, examples, model):
        model.eval()
        batch_size = 64
        one_acc_num = 0.0
        one_sketch_num = 0.0
        total_sql = 0
        right_result = []
        wrong_result = []
        beam_wrong_result = list()

        best_correct = 0
        beam_correct = 0

        sel_num = []
        sel_col = []
        agg_col = []
        table_col = []
        json_datas = []

        examples=examples

        for st in tqdm.tqdm(range(0, len(examples), batch_size), desc="Evaluating", leave=False):
            ed = st + batch_size if st + batch_size < len(examples) else len(examples)

            with torch.no_grad():
                results_all = model.parse(examples[st:ed],
                                          beam_size=self.args.beam_size)

            for i, example in enumerate(examples[st:ed]):
                try:
                    results = results_all[0][i]
                    sketch_actions = " ".join(str(x) for x in results_all[1][i])
                    list_preds = []
                    for x in results[0].actions:
                        if type(x) == C:
                            x.id_c = example.remove_dict_reverse[x.id_c]
                    pred = " ".join([str(x) for x in results[0].actions])
                    for x in results:
                        list_preds.append(" ".join(str(x.actions)))
                except Exception as e:
                    pred = ""
                    sketch_actions = ""

                simple_json = example.sql_json
                simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1][i])
                simple_json['model_result'] = pred
                json_datas.append(simple_json)

                example.sql_json['model_result'] = pred

                if len(results) > 0:
                    pred = []
                    x_id = 0
                    while x_id < len(results[0].actions):
                        pred.append(results[0].actions[x_id])
                        if type(results[0].actions[x_id]) == C and results[0].actions[x_id].id_c == 0:
                            x_id += 1
                        x_id += 1
                    pred = " ".join([str(x) for x in pred])
                else:
                    pred = " "

                glod = []
                x_id = 0
                while x_id < len(example.tgt_actions):
                    if type(example.tgt_actions[x_id]) == C:
                        example.tgt_actions[x_id].id_c = example.remove_dict_reverse[example.tgt_actions[x_id].id_c]
                    glod.append(example.tgt_actions[x_id])
                    if type(example.tgt_actions[x_id]) == C and example.tgt_actions[x_id].id_c == 0:
                        x_id += 1
                    x_id += 1
                glod = " ".join([str(x) for x in glod])

                sketch_glod = " ".join([str(x) for x in example.sketch])

                src_str = " ".join([str(x) for x in example.src_sent])
                if sketch_glod == sketch_actions:
                    one_sketch_num += 1
                if pred == glod:
                    one_acc_num += 1
                else:
                    pass

                glod = " ".join([str(x) for x in example.tgt_actions]).strip()
                if len(results) > 0:
                    pred = " ".join([str(x) for x in results[0].actions]).strip()
                    _best_correct, _beam_correct = calc_beam_acc(results, example)
                else:
                    pred = ""
                    _best_correct, _beam_correct = False, False
                if _beam_correct:
                    beam_correct += 1
                else:
                    preds = [" ".join([str(x) for x in r.actions]) for r in results]
                    preds.append(glod)
                    preds.append(src_str)
                    preds.append(example.sql)
                    beam_wrong_result.append(preds)
                if _best_correct:
                    best_correct += 1
                    right_result.append((i + st, pred, glod, src_str, example.sql))
                else:
                    wrong_result.append((i + st, pred, glod, src_str, example.sql))

                total_sql += 1

        return best_correct / total_sql, beam_correct / total_sql, \
               (right_result, wrong_result, beam_wrong_result), \
               (sel_num, sel_col, agg_col, table_col)

    def eval_task_stream(self, task_id, cur_test_acc):
        avg_test_acc = 0
        whole_test_acc = 0
        n_examples = 0
        temp_bwt = [float("-inf") for k in range(task_id)]
        for k in tqdm.tqdm(range(task_id), desc='Evaluating past tasks', leave=False):
            test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.task_controller.task_list[k]["test"],
                                                                               self.model)

            # print("---", k, test_acc, self.first_acc_list[k])
            whole_test_acc += test_acc * len(self.task_controller.task_list[k]["test"])
            n_examples += len(self.task_controller.task_list[k]["test"])
            temp_bwt[k] = max(test_acc - self.first_acc_list[k], temp_bwt[k])
            avg_test_acc += test_acc
        avg_test_acc += cur_test_acc
        avg_test_acc /= (task_id + 1)

        whole_test_acc += cur_test_acc * len(self.task_controller.task_list[task_id]["test"])
        n_examples += len(self.task_controller.task_list[task_id]["test"])
        whole_test_acc /= n_examples

        self.avg_acc_list[task_id] = avg_test_acc
        self.whole_acc_list[task_id] = whole_test_acc

        if task_id < self.args.task_num - 1:
            test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.task_controller.task_list[task_id + 1]["test"],
                                                                               self.model)
            self.temp_fwt[task_id + 1] = test_acc - self.acc_rand_list[task_id + 1]

        if task_id > 0:
            self.bwt_list[task_id] = sum(temp_bwt) / len(temp_bwt) if task_id > 0 else 0
            self.fwt_list[task_id] = sum(self.temp_fwt[1:task_id + 1]) / task_id

        # print('Evaluation: \tAvg Acc: %.4f\tWhole Acc: %.4f\tBWT: %.4f\tFWT: %.4f\n' % (self.avg_acc_list[task_id],
        #                                                                                 self.whole_acc_list[task_id],
        #                                                                                 self.bwt_list[task_id],
        #                                                                                 self.fwt_list[task_id]))
