# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/trainer.py
# @Software: PyCharm
"""
import json
import math
import sys
import copy
import time
import torch
import os
import random

import tqdm
from torch.utils.tensorboard import SummaryWriter
sys.path.append(".")
from rule import define_rule
from rule.define_rule import C, Root1
from parser.irnet import IRNet
from sfnet.basic_trainer import BasicTrainer


class SFNet(BasicTrainer):

    def __init__(self, args, model_save_path):
        super(SFNet, self).__init__(args, model_save_path)

        self.teacher = IRNet(self.args, self.grammar).to(self.device)
        optimizer_grouped_parameters_teacher = self.make_optimizer_groups(self.teacher)
        self.optimizer_teacher = torch.optim.Adam(optimizer_grouped_parameters_teacher)

    def save(self, model, name="model.bin"):
        torch.save({"model": model.state_dict()}, open(os.path.join(self.model_save_path, name), 'wb'))

    def load(self, model, name="model.bin"):
        model.load_state_dict(torch.load(open(os.path.join(self.model_save_path, name), "rb"), map_location=self.device)["model"])

    def train_with_teacher(self):

        writer = SummaryWriter(self.log_path)

        for i in tqdm.tqdm(range(self.args.task_num), desc="Experience task stream"):
            best_result = {
                "acc": 0.0,
                "epoch": 0,
                "teacher_acc": 0.0,
                "teacher_epoch": 0
            }

            sql_result_path = os.path.join(self.model_save_path, 'sql_result', f'task_{i}')
            os.makedirs(sql_result_path, exist_ok=True)

            patience = 0
            if i > 0:
                self.load(self.teacher)

            memory_student_train, memory_student_semi, memory_teacher_train, memory_teacher_semi = self.task_controller.get_memory(i)

            teacher_train_examples = list(self.task_controller.task_list[i]["train"])
            if self.args.teacher_cl:
                teacher_train_examples.extend(memory_teacher_train)

            for epoch in tqdm.tqdm(range(self.args.warm_boot_epoch), desc=f'Task {i} ### teacher warm boot', leave=False):
                cur_teacher_train_examples = list(teacher_train_examples)
                if self.args.teacher_cl:
                    cur_memory_teacher_semi_examples = self.task_controller.get_semi_memory(memory_teacher_semi,
                                                                                            int(self.args.st_rate * len(teacher_train_examples)))
                    cur_teacher_train_examples.extend(cur_memory_teacher_semi_examples)

                loss = self.train_one_epoch(cur_teacher_train_examples,
                                            self.teacher,
                                            self.optimizer_teacher)
                writer.add_scalar(f'task_{i}/teacher_warm_boot/loss', loss, epoch)

                if epoch < self.args.eval_epoch:
                    continue

                dev_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.task_controller.task_list[i]["dev"],
                                                                                  self.teacher)
                writer.add_scalar(f'task_{i}/teacher_warm_boot/dev_acc', dev_acc, epoch)

                if dev_acc >= best_result['teacher_acc']:
                    best_result['teacher_acc'], best_result['teacher_epoch'] = dev_acc, epoch
                    self.save(self.teacher, name="teacher.bin")
                    patience = 0
                else:
                    patience += 1

                if patience > self.args.max_patience:
                    break

            for epoch in tqdm.tqdm(range(self.args.st_epoch_num), desc=f'Task {i} ### teacher self-training', leave=False):
                cur_teacher_train_examples = list(self.task_controller.task_list[i]["train"])
                with torch.no_grad():
                    cur_teacher_semi_examples = self.task_controller.get_semi_memory(self.task_controller.task_list[i]["semi"],
                                                                                     int(self.args.st_rate * len(teacher_train_examples)))
                    cur_teacher_semi_examples = [x for x in self.predict_pseudo_labels(cur_teacher_semi_examples, self.teacher) if x.pseudo_conf > 0]
                    cur_teacher_train_examples.extend(cur_teacher_semi_examples)

                if self.args.teacher_cl:
                    cur_memory_teacher_semi_examples = self.task_controller.get_semi_memory(memory_teacher_semi,
                                                                                            int(self.args.st_rate * len(teacher_train_examples)))
                    cur_teacher_train_examples.extend(cur_memory_teacher_semi_examples)

                loss = self.train_one_epoch(cur_teacher_train_examples,
                                            self.teacher,
                                            self.optimizer_teacher,
                                            use_conf=True)
                writer.add_scalar(f'task_{i}/teacher_self_training/loss', loss, epoch)

                dev_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.task_controller.task_list[i]["dev"],
                                                                                  self.teacher)
                writer.add_scalar(f'task_{i}/teacher_self_training/dev_acc', dev_acc, epoch)

                if dev_acc >= best_result['teacher_acc']:
                    best_result['teacher_acc'], best_result['teacher_epoch'] = dev_acc, epoch
                    self.save(self.teacher, name="teacher.bin")
                    patience = 0
                else:
                    patience += 1

                if patience > self.args.max_patience:
                    break

            with torch.no_grad():
                self.task_controller.task_list[i]["semi"] = [x for x in self.predict_pseudo_labels(self.task_controller.task_list[i]["semi"],
                                                                                                   self.teacher) if x.pseudo_conf > 0]
                self.task_controller.build_memory(i, self.teacher)

            patience = 0
            for epoch in tqdm.tqdm(range(self.args.epoch_num), desc=f'Task {i} ### student episodic memory replay', leave=False):
                cur_student_train_examples = list(self.task_controller.task_list[i]["train"])
                if self.args.student_cl:
                    cur_student_train_examples.extend(memory_student_train)
                    cur_memory_student_semi_examples = self.task_controller.get_semi_memory(memory_student_semi,
                                                                                            int(self.args.st_rate * len(cur_student_train_examples)))
                    cur_student_train_examples.extend(cur_memory_student_semi_examples)

                cur_student_semi_examples = self.task_controller.get_semi_memory(self.task_controller.task_list[i]["semi"],
                                                                                 int(self.args.st_rate * len(cur_student_train_examples)))
                cur_student_train_examples.extend(cur_student_semi_examples)

                loss = self.train_one_epoch(cur_student_train_examples,
                                            self.model,
                                            self.optimizer)
                writer.add_scalar(f'task_{i}/student_emr/loss', loss, epoch)

                if epoch < self.args.eval_epoch:
                    continue

                dev_acc, beam_acc, (right_dev, wrong_dev, _), write_data = self.epoch_acc(self.task_controller.task_list[i]["dev"],
                                                                                          self.model)
                writer.add_scalar(f'task_{i}/student_emr/dev_acc', dev_acc, epoch)

                if dev_acc >= best_result['acc']:
                    best_result['acc'], best_result['epoch'] = dev_acc, epoch
                    self.save(self.model, name="model.bin")
                    json.dump(right_dev, open(os.path.join(sql_result_path, 'right_dev.json'), 'w'), indent=2)
                    json.dump(wrong_dev, open(os.path.join(sql_result_path, 'wrong_dev.json'), 'w'), indent=2)
                    patience = 0
                else:
                    patience += 1

                if patience > self.args.max_patience:
                    break

            self.load(self.model)
            test_acc, beam_acc, (right_test, wrong_test, _), write_data = self.epoch_acc(self.task_controller.task_list[i]["test"],
                                                                                         self.model)
            writer.add_scalar('student_cl/first_test_acc', test_acc, i)
            json.dump(right_test, open(os.path.join(sql_result_path, 'right_test.json'), 'w'), indent=2)
            json.dump(wrong_test, open(os.path.join(sql_result_path, 'wrong_test.json'), 'w'), indent=2)

            self.first_acc_list[i] = test_acc
            self.eval_task_stream(i, test_acc)

            writer.add_scalar('student_cl/avg_acc', self.avg_acc_list[i], i)
            writer.add_scalar('student_cl/whole_acc', self.whole_acc_list[i], i)
            writer.add_scalar('student_cl/bwt_acc', self.bwt_list[i], i)
            writer.add_scalar('student_cl/fwt_acc', self.fwt_list[i], i)


        writer.close()

        return self.avg_acc_list, self.whole_acc_list, self.bwt_list, self.fwt_list

    def train_one_epoch(self, examples, model, optimizer, optimize_step=True, use_conf=False):
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
                                                                  example_num,
                                                                  use_conf=use_conf)

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

    def predict_pseudo_labels(self, examples, model):
        batch_size = 64
        model.eval()
        for st in tqdm.tqdm(range(0, len(examples), batch_size), desc="Predicting pseudo labels", leave=False):
            ed = st + batch_size if st + batch_size < len(examples) else len(examples)

            with torch.no_grad():
                results_all = model.parse(examples[st:ed], beam_size=self.args.beam_size)

            for i, example in enumerate(examples[st:ed]):
                try:
                    results = results_all[0][i]
                    all_actions = []
                    conf = math.exp(results[0].score) if results[0].score < 0 else 1.0
                    x_id = 0
                    while x_id < len(results[0].actions):
                        all_actions.append(results[0].actions[x_id])
                        if type(results[0].actions[x_id]) == C and results[0].actions[x_id].id_c == 0:
                            x_id += 1
                        x_id += 1

                except Exception as e:
                    all_actions = []
                    conf = 0.0

                if not all_actions:
                    all_actions = [Root1(3)]
                    conf = 0.0

                sketch_actions = list()
                flag = False
                for action in all_actions:
                    if isinstance(action, define_rule.C) or isinstance(action, define_rule.T) or isinstance(action, define_rule.A):
                        flag = True
                        continue
                    sketch_actions.append(action)

                if not flag:
                    conf = 0.0

                parent_actions = list()
                parent_actions_idx = list()
                for idx, t_action in enumerate(all_actions):
                    if idx > 0 and all_actions[idx - 1] == t_action.parent:
                        parent_actions.append(None)
                        parent_actions_idx.append(None)
                    else:
                        parent_actions.append(t_action.parent)
                        parent_actions_idx.append(sketch_actions.index(t_action.parent) if t_action.parent is not None else None)

                examples[st + i].tgt_actions = all_actions
                examples[st + i].sketch = sketch_actions
                examples[st + i].pseudo_conf = conf
                examples[st + i].parent_actions = parent_actions
                examples[st + i].parent_actions_idx = parent_actions_idx
        return examples
