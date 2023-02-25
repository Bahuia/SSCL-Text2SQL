import math
import random
import sys
import os
sys.path.append("..")
import random
from src.rule.define_rule import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
from src.cssl.cssl_task import TaskController
from transformers import AdamW
from src.rule import define_rule
from src.model import Seq2Tree
from src.cssl.cssl_utils import save_args, calc_beam_acc
import nn_utils
import time
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
import threading


class CSSLTrainer(object):
    def __init__(self, args, model_save_path):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.grammar = define_rule.Grammar(is_sketch=None)
        self.vocab = pickle.load(open(args.vocab_path, "rb"))

        print("Loading Task Data ...")
        self.tc = TaskController(args)

        print("Init the Model ...")
        self.model = Seq2Tree(args, self.vocab, self.grammar, self.vocab).to(self.device)
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in ['bertModel'])],
             'lr': args.lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['bertModel'])],
             'lr': args.bert_lr}
        ]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)

        self.temp_acc_list = [-1.0 for i in range(args.task_num)]
        self.bwt_list = [float("-inf") for i in range(args.task_num)]
        self.acc_list = [-1.0 for i in range(args.task_num)]
        self.fwt_temp_list = [float("-inf") for i in range(args.task_num)]
        self.fwt_list = [float("-inf") for i in range(args.task_num)]
        self.acc_rand_list = [0.0 for i in range(args.task_num)]
        self.whole_acc_list = [float("-inf") for i in range(args.task_num)]

        self.model_save_path = model_save_path
        save_args(self.args, os.path.join(self.model_save_path, "config.json"))
        self.train_time = 0.0
        self.pred_time = 0.0
        self.eval_time = 0.0


    def train_normal(self):
        for i in range(self.args.task_num):
            best_result = {"acc": 0.0, "epoch": 0}
            for epoch in range(self.args.epoch):
                start_time = time.time()
                cur_train_examples = []
                for j in range(i + 1):
                    cur_train_examples.extend(self.tc.task_list[j]["train"])

                loss = self.train_one_epoch(cur_train_examples, self.model, self.optimizer)
                print("Task {}, Epoch Train {}, Loss {}, Time: {}\n".format(i, epoch, loss, time.time()- start_time))
                self.train_time += time.time() - start_time

                continue
                if epoch < self.args.eval_epoch:

                total_dev_acc = self.dev_cl(i, epoch)
                if total_dev_acc >= best_result['acc']:
                    best_result['acc'], best_result['epoch'] = total_dev_acc, epoch
                    self.save(self.model, name="model.bin")
                    print('NEW BEST MODEL: \tEpoch: {}\tTest Acc: {}'.format(epoch, total_dev_acc))
                self.load(self.model)
            start_eval_time = time.time()
            self.test_cl(i)
            self.eval_time += time.time() - start_eval_time

        print("pred_time: ", self.pred_time)
        print("eval_time: ", self.eval_time)
        print("total_time: ", self.train_time + self.eval_time)
        return self.acc_list, self.bwt_list, self.fwt_list, self.whole_acc_list


    def train_naive(self):
        for i in range(self.args.task_num):
            memory_student_train, memory_student_semi, memory_teacher_train, memory_teacher_semi = self.tc.get_memory(i, self.args.stu_semi)
            train_examples = list(self.tc.task_list[i]["train"])

            if self.args.stu_cl:
                train_examples.extend(memory_student_train)
            if self.args.tec_cl:
                train_examples.extend(memory_teacher_train)

            if self.args.stu_semi:
                print("Warm Boot ...")
                for epoch in range(self.args.warm_boot_epoch):
                    start_train_time = time.time()
                    cur_train_examples = list(train_examples)
                    if self.args.stu_cl:
                        cur_memory_student_semi = self.tc.get_semi_memory(memory_student_semi, int(self.args.st_rate * len(train_examples)))
                        cur_train_examples.extend(cur_memory_student_semi)
                    if self.args.tec_cl:
                        cur_memory_teacher_semi = self.tc.get_semi_memory(memory_teacher_semi, int(self.args.st_rate * len(train_examples)))
                        cur_train_examples.extend(cur_memory_teacher_semi)
                    loss = self.train_one_epoch(cur_train_examples, self.model, self.optimizer)
                    print("Warm Boot, Task: {}, Epoch: {}, Loss: {}, Time: {}\n".format(i, epoch, loss, time.time() - start_train_time))
                    self.train_time += time.time() - start_train_time

            best_result = {"acc": 0.0, "epoch": 0}
            for epoch in range(self.args.st_epoch if self.args.stu_semi else self.args.epoch):
                start_time = time.time()
                cur_train_examples = list(train_examples)
                if self.args.stu_semi:
                    with torch.no_grad():
                        cur_semi_examples = self.tc.get_semi_memory(self.tc.task_list[i]["semi"], int(self.args.st_rate * len(train_examples)))
                        start_pred_time = time.time()
                        cur_semi_examples = [x for x in self.predict_pseudo_labels(cur_semi_examples, self.model) if x.pseudo_conf]
                        self.pred_time += time.time() - start_pred_time
                        cur_train_examples.extend(cur_semi_examples)
                    if self.args.stu_cl:
                        cur_memory_student_semi = self.tc.get_semi_memory(memory_student_semi, int(self.args.st_rate * len(train_examples)))
                        cur_train_examples.extend(cur_memory_student_semi)
                    if self.args.tec_cl:
                        cur_memory_teacher_semi = self.tc.get_semi_memory(memory_teacher_semi, int(self.args.st_rate * len(train_examples)))
                        cur_train_examples.extend(cur_memory_teacher_semi)
                start_train_time = time.time()
                loss = self.train_one_epoch(cur_train_examples, self.model, self.optimizer)
                self.train_time += time.time() - start_train_time
                print("Task: {}, Epoch: {}, Loss: {}, Time: {}\n".format(i, epoch, loss, time.time() - start_train_time))
                if epoch < self.args.eval_epoch:
                    continue

                start_eval_time = time.time()
                with torch.no_grad():
                    total_dev_acc = self.dev_cl(i, epoch)
                self.eval_time += time.time() - start_eval_time

                if total_dev_acc >= best_result["acc"]:
                    best_result["acc"], best_result["epoch"] = total_dev_acc, epoch
                    self.save(self.model, name="model.bin")
                    print("New Best Model: \tEpoch: {}, Dev Acc: {}".format(epoch, total_dev_acc))
                self.load(self.model)
            start_pred_time = time.time()
            with torch.no_grad():
                self.tc.task_list[i]["semi"] = self.predict_pseudo_labels(self.tc.task_list[i]["semi"], self.model)
                self.tc.build_memory(i, self.model, self.args.stu_semi)
            self.pred_time += time.time() - start_pred_time
            start_eval_time = time.time()
            self.test_cl(i)
            self.eval_time += time.time() - start_eval_time
        print("total_stu_sampling_time: ", self.tc.total_stu_sampling_time)
        print("total_tec_sampling_time: ", self.tc.total_tec_sampling_time)
        print("train_time: ", self.train_time)
        print("pred_time: ", self.pred_time)
        print("eval_time: ", self.eval_time)
        print("total_time: ", self.train_time + self.eval_time + self.pred_time)
        return self.acc_list, self.bwt_list, self.fwt_list, self.whole_acc_list


    def train_student_teacher(self):
        for i in range(self.args.task_num):
            teacher = Seq2Tree(self.args, self.vocab, self.grammar, self.vocab).to(self.device)
            param_optimizer_teacher = list(teacher.named_parameters())
            optimizer_grouped_parameters_teacher = [
                {'params': [p for n, p in param_optimizer_teacher if not any(nd in n for nd in ['bertModel'])],
                 'lr': self.args.lr},
                {'params': [p for n, p in param_optimizer_teacher if any(nd in n for nd in ['bertModel'])],
                 'lr': self.args.bert_lr}
            ]
            optimizer_teacher = torch.optim.Adam(optimizer_grouped_parameters_teacher, lr=self.args.lr)
            if i > 0:
                self.load(teacher)

            memory_student_train, memory_student_semi, memory_teacher_train, memory_teacher_semi = self.tc.get_memory(i)
            teacher_train_examples = list(self.tc.task_list[i]["train"])
            if self.args.tec_cl:
                teacher_train_examples.extend(memory_teacher_train)

            for epoch in range(self.args.warm_boot_epoch):
                start_time = time.time()
                cur_teacher_train_examples = list(teacher_train_examples)
                if self.args.tec_cl:
                    cur_memory_teacher_semi_examples = self.tc.get_semi_memory(memory_teacher_semi, int(self.args.st_rate * len(teacher_train_examples)))
                    cur_teacher_train_examples.extend(cur_memory_teacher_semi_examples)
                start_train_time = time.time()
                loss = self.train_one_epoch(cur_teacher_train_examples, teacher, optimizer_teacher)
                print("Teacher Warm Boot Training: Task {}, Epoch {}, Loss {}, Time {}\n".format(i, epoch, loss, time.time() - start_time))
                self.train_time += time.time() - start_train_time

            for epoch in range(self.args.st_epoch):
                start_time = time.time()
                cur_teacher_train_examples = list(self.tc.task_list[i]["train"])
                with torch.no_grad():
                    cur_teacher_semi_examples = self.tc.get_semi_memory(self.tc.task_list[i]["semi"], int(self.args.st_rate * len(teacher_train_examples)))
                    start_pred_time = time.time()
                    cur_teacher_semi_examples = [x for x in self.predict_pseudo_labels(cur_teacher_semi_examples, self.model) if x.pseudo_conf]
                    self.pred_time += time.time() - start_pred_time
                    cur_teacher_train_examples.extend(cur_teacher_semi_examples)
                if self.args.tec_cl:
                    cur_memory_teacher_semi_examples = self.tc.get_semi_memory(memory_teacher_semi, int(self.args.st_rate * len(teacher_train_examples)))
                    cur_teacher_train_examples.extend(cur_memory_teacher_semi_examples)
                start_train_time = time.time()
                loss = self.train_one_epoch(cur_teacher_train_examples, teacher, optimizer_teacher)
                print("Teacher ST Training: Task {}, Epoch {}, Loss {}, Time {}\n".format(i, epoch, loss, time.time() - start_time))
                self.train_time += time.time() - start_train_time

            with torch.no_grad():
                self.tc.task_list[i]["semi"] = self.predict_pseudo_labels(self.tc.task_list[i]["semi"], teacher)
                self.tc.build_memory(i, teacher)

            best_result = {"acc": 0.0, "epoch": 0}
            for epoch in range(self.args.epoch):
                start_time = time.time()
                cur_student_train_examples = list(self.tc.task_list[i]["train"])
                if self.args.stu_cl:
                    cur_student_train_examples.extend(memory_student_train)

                if self.args.stu_cl:
                    cur_memory_student_semi_examples = self.tc.get_semi_memory(memory_student_semi, int(self.args.st_rate * len(cur_student_train_examples)))
                    cur_student_train_examples.extend(cur_memory_student_semi_examples)
                cur_student_semi_examples = self.tc.get_semi_memory(self.tc.task_list[i]["semi"], int(self.args.st_rate * len(cur_student_train_examples)))
                cur_student_train_examples.extend(cur_student_semi_examples)

                start_train_time = time.time()
                loss = self.train_one_epoch(cur_student_train_examples, self.model, self.optimizer)
                print("Student Training: Task {}, Epoch {}, Loss {}, Time {}\n".format(i, epoch, loss, time.time() - start_time))
                self.train_time += time.time() - start_train_time
                if epoch < self.args.eval_epoch:
                    continue

                start_eval_time = time.time()
                total_dev_acc = self.dev_cl(i, epoch)
                self.eval_time += time.time() - start_eval_time
                if total_dev_acc >= best_result["acc"]:
                    best_result["acc"], best_result["epoch"] = total_dev_acc, epoch
                    self.save(self.model, name="model.bin")
                    print("New Best Model: \tEpoch: {}, Dev Acc: {}".format(epoch, total_dev_acc))
                self.load(self.model)
            start_eval_time = time.time()
            self.test_cl(i)
            self.eval_time += time.time() - start_eval_time
        print("total_stu_sampling_time: ", self.tc.total_stu_sampling_time)
        print("total_tec_sampling_time: ", self.tc.total_tec_sampling_time)
        print("train_time: ", self.train_time)
        print("pred_time: ", self.pred_time)
        print("eval_time: ", self.eval_time)
        print("total_time: ", self.train_time + self.eval_time + self.pred_time)
        return self.acc_list, self.bwt_list, self.fwt_list, self.whole_acc_list


    def test_cl(self, task_ids):
        start_time = time.time()
        i = task_ids
        total_test_acc = 0.0
        temp_bwt = [float("-inf") for k in range(i)]
        for k in range(i + 1):
            test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.tc.task_list[k]["test"])
            if k == i:
                print("test acc: {}, beam acc: {}".format(test_acc, beam_acc))
                self.temp_acc_list[k] = max(self.temp_acc_list[k], test_acc)
            else:
                temp_bwt[k] = max(test_acc - self.temp_acc_list[k], temp_bwt[k])
            total_test_acc += test_acc
        total_test_acc = total_test_acc / (i + 1)
        self.acc_list[i] = max(self.acc_list[i], total_test_acc)

        if i < self.args.task_num - 1:
            test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.tc.task_list[i + 1]["test"])
            self.fwt_temp_list[i + 1] = max(test_acc - self.acc_rand_list[i + 1], self.fwt_temp_list[i + 1])
        if i > 0:
            bwt = sum(temp_bwt) / len(temp_bwt) if i > 0 else 0
            self.bwt_list[i] = max(bwt, self.bwt_list[i])
            self.fwt_list[i] = max(sum(self.fwt_temp_list[1:i + 1]) / i, self.fwt_list[i])

        cur_whole_test_examples = []
        for k in range(i + 1):
            cur_whole_test_examples.extend(self.tc.task_list[k]["test"])
        whole_test_acc, whole_beam_acc, (right, wrong, _), write_data = self.epoch_acc(cur_whole_test_examples)
        self.whole_acc_list[i] = max(self.whole_acc_list[i], whole_test_acc)

        print("Evaluation Test: \tTime: %.4f\tTest Acc: %.4f\tTotal Test Acc: %.4f\tWhole Time Acc: %.4f\n" % (time.time() - start_time, self.temp_acc_list[i], total_test_acc, whole_test_acc))


    def dev_cl(self, task_ids, epoch):
        start_time = time.time()
        i = task_ids
        total_test_acc = 0.0
        cur_acc = 0.0
        for k in range(i + 1):
            test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.tc.task_list[k]["dev"])
            total_test_acc += test_acc
            if k == i:
                cur_acc = test_acc
        total_test_acc /= (i + 1)
        print("Evaluation Dev: \tEpoch: %d\tTime: %.4f\tDev Acc: %.4f\tTotal CL Test Acc: %.4f\n" % (epoch, time.time() - start_time, cur_acc, total_test_acc))
        return total_test_acc


    def save(self, model, name="model.bin"):
        torch.save({"model": model.state_dict()},
                   open(os.path.join(self.model_save_path, name), 'wb'))


    def load(self, model, name="model.bin"):
        model.load_state_dict(torch.load(open(os.path.join(self.model_save_path, name), "rb"), map_location=self.device)["model"])


    def train_one_epoch(self, examples, cur_model, cur_optimizer):
        cur_model.train()
        # shuffe
        random.shuffle(examples)
        cum_loss = 0.0
        st = 0
        while st < len(examples):
            ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

            cur_optimizer.zero_grad()

            score = cur_model.forward(examples[st:ed])
            loss_sketch = -score[0]
            loss_lf = -score[1]
            if self.args.semi_name == "st":
                conf_tensor = torch.tensor([example.pseudo_conf for example in examples[st:ed]], dtype=torch.long).to(self.device)
                loss_sketch *= conf_tensor
                loss_lf *= conf_tensor
            #
            loss_sketch = torch.mean(loss_sketch)
            loss_lf = torch.mean(loss_lf)

            # print(loss_sketch, loss_lf)
            #
            loss = loss_lf + loss_sketch

            # TODO: what is the sup_attention?
            loss.backward()
            if self.args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(cur_model.parameters(), self.args.clip_grad)
            cur_optimizer.step()
            # some records
            cum_loss += loss.data.cpu().numpy() * (ed - st)
            st = ed
        return cum_loss / len(examples)


    def epoch_acc(self, examples):
        start_time = time.time()
        self.model.eval()
        total_sql = 0
        best_correct, beam_correct = 0, 0

        json_datas = []
        st = 0

        while st < len(examples):
            ed = st + self.args.thread_num if st + self.args.thread_num < len(examples) else len(examples)
            global_result_all = self.multi_threads_parsing(examples[st:ed], self.args.beam_size, ed - st)
            for i, results_all in enumerate(global_result_all):
                example = examples[st + i]
                try:
                    results = results_all[0]
                    for x in results[0].actions:
                        if type(x) == C:
                            x.id_c = example.remove_dict_reverse[x.id_c]
                    pred = " ".join([str(x) for x in results[0].actions])
                except Exception as e:
                    pred = ""

                simple_json = examples[st + i].sql_json
                simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
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

                x_id = 0
                while x_id < len(example.tgt_actions):
                    if type(example.tgt_actions[x_id]) == C:
                        example.tgt_actions[x_id].id_c = example.remove_dict_reverse[example.tgt_actions[x_id].id_c]
                    if type(example.tgt_actions[x_id]) == C and example.tgt_actions[x_id].id_c == 0:
                        x_id += 1
                    x_id += 1

                if len(results) > 0:
                    _best_correct, _beam_correct = calc_beam_acc(results, example)
                else:
                    _best_correct, _beam_correct = False, False
                if _beam_correct:
                    beam_correct += 1

                if _best_correct:
                    best_correct += 1

                total_sql += 1
            st = ed
        with open("lf_predict.json", "w") as f:
            json.dump(json_datas, f)
        return best_correct / total_sql, beam_correct / total_sql, (None, None, None), None


    def predict_pseudo_labels(self, examples, cur_model):
        start_time = time.time()
        cur_model.eval()
        st = 0
        while st < len(examples):
            ed = st + self.args.thread_num if st + self.args.thread_num < len(examples) else len(examples)
            global_result_all = self.multi_threads_parsing(examples[st:ed], self.args.beam_size, ed - st)
            for i, results_all in enumerate(global_result_all):

                try:
                    results = results_all[0]
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
                    conf = 0

                sketch_actions = list()
                for action in all_actions:
                    if isinstance(action, define_rule.C) or isinstance(action, define_rule.T) or isinstance(action, define_rule.A):
                        continue
                    sketch_actions.append(action)

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
            st = ed
        print("Pseudo Label Prediction Time Cost is ", time.time() - start_time)
        return examples


    def multi_threads_parsing(self, examples, beam_size, thread_num):
        global_result = [[] for i in range(thread_num)]
        def thread_function(example, beam_size, cur_model, idx):
            with torch.no_grad():
                cur_result = cur_model.parse(example, beam_size=beam_size)
                global_result[idx] = cur_result

        t_list = []
        for i in range(thread_num):
            t = threading.Thread(target=thread_function, args=(examples[i], beam_size, self.model, i))
            t.start()
            t_list.append(t)
        for t in t_list:
            t.join()
        return global_result



