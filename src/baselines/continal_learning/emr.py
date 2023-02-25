import torch
import os
import time
import json
import random
import pickle
import quadprog
import datetime
import torch.utils.data as torch_data
import numpy as np
from rule.define_rule import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1
from cssl.cssl_task import TaskController
from transformers import AdamW
from rule import define_rule
from cssl.cssl_utils import save_args, calc_beam_acc
from utils import to_batch_seq, epoch_acc
from model import Seq2Tree
from baselines.continal_learning.cl_utils import RANDOM, FSS, BALANCE, LFS, DLFS, GSS


class EMRTrainer():
    def __init__(self, args, model_save_path):
        self.args = args
        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.grammar = define_rule.Grammar(is_sketch=None)
        self.vocab = pickle.load(open(args.vocab_path, "rb"))

        print("Init the Model ...")
        self.model = Seq2Tree(args, self.vocab, self.grammar, self.vocab).to(self.device)
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in ['bertModel'])], 'lr': args.lr},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in ['bertModel'])], 'lr': args.lr * 0.05}
        ]

        self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)

        print("Load Task Data ...")
        self.task_controller = TaskController(args)

        self.temp_acc_list = [-1 for i in range(args.task_num)]
        self.bwt_list = [float("-inf") for i in range(args.task_num)]
        self.acc_list = [-1 for i in range(args.task_num)]
        self.fwt_temp_list = [float("-inf") for i in range(args.task_num)]
        self.fwt_list = [float("-inf") for i in range(args.task_num)]
        self.acc_rand_list = [0.0 for i in range(args.task_num)]

        self.model_save_path = model_save_path
        save_args(args, os.path.join(self.model_save_path, "config.json"))

        # CL component
        self.past_task_id = -1
        self.observed_task_ids = []
        self.memory_data = {}  # stores exemplars class by class

    def train(self):
        for i in range(self.args.task_num):
            best_result = {"acc": 0.0, "epoch": 0}
            examples = self.task_controller.task_list[i]["train"]

            if i != self.past_task_id:
                self.observed_task_ids.append(i)
                self.past_task_id = i

            self.memory_data[i] = []
            sampled_examples = RANDOM(examples=examples,
                                      memory_size=self.args.memory_size)
            # sampled_examples = FSS(model=self.model,
            #                        examples=examples,
            #                        memory_size=self.args.memory_size, args=self.args)
            # sampled_examples = BALANCE(examples=examples,
            #                            memory_size=self.args.memory_size)
            # sampled_examples = LFS(examples=examples,
            #                            memory_size=self.args.memory_size)
            # sampled_examples = DLFS(examples=examples,
            #                        memory_size=self.args.memory_size)

            # print("DLFS", sampled_examples)

            self.memory_data[i].extend(sampled_examples)

            for epoch in range(self.args.epoch):
                self.model.train()
                epoch_begin = time.time()
                random.shuffle(examples)
                st = 0
                report_loss, example_num = 0.0, 0

                while st < len(examples):
                    ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

                    self.optimizer.zero_grad()

                    report_loss, example_num, loss = self.train_one_batch(examples[st:ed], report_loss, example_num)

                    if len(self.observed_task_ids) > 1:
                        for _task_id in range(len(self.observed_task_ids) - 1):
                            start_time = time.time()
                            past_task_id = self.observed_task_ids[_task_id]
                            replay_examples = self.memory_data[past_task_id]

                            assert past_task_id != i

                            random.shuffle(replay_examples)
                            replay_report_loss = 0.0
                            _st = 0
                            replay_example_num = 0

                            while _st < len(replay_examples):
                                _ed = _st + self.args.batch_size if _st + self.args.batch_size < len(
                                    replay_examples) else len(replay_examples)

                                # print("++++", len(replay_examples[_st:_ed]))

                                replay_report_loss, replay_example_num, replay_loss = self.train_one_batch(
                                    replay_examples[_st:_ed],
                                    replay_report_loss,
                                    replay_example_num)
                                loss += replay_loss
                                report_loss += replay_report_loss
                                _st = _ed

                    loss.backward()
                    if self.args.clip_grad > 0.:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.args.clip_grad)
                    self.optimizer.step()
                    st = ed

                print("\nTask {}, Epoch Train {}, Loss {}, Time {}".format(i, epoch, report_loss, time.time() - epoch_begin))

                total_test_acc = 0.
                temp_bwt = [float("-inf") for k in range(i)]
                for k in range(i + 1):
                    test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(
                        self.task_controller.task_list[k]["test"])
                    if k == i:
                        print("Test acc: {}, beam acc: {}".format(test_acc, beam_acc))
                        self.temp_acc_list[k] = max(self.temp_acc_list[k], test_acc)
                    else:
                        temp_bwt[k] = max(test_acc - self.temp_acc_list[k], temp_bwt[k])
                    total_test_acc += test_acc
                total_test_acc = total_test_acc / (i + 1)
                self.acc_list[i] = max(self.acc_list[i], total_test_acc)

                if i < self.args.task_num - 1:
                    test_acc, beam_acc, (right, wrong, _), write_data = self.epoch_acc(
                        self.task_controller.task_list[i + 1]["test"])
                    self.fwt_temp_list[i + 1] = max(test_acc - self.acc_rand_list[i + 1], self.fwt_temp_list[i + 1])
                if i > 0:
                    bwt = sum(temp_bwt) / len(temp_bwt) if i > 0 else 0
                    self.bwt_list[i] = max(bwt, self.bwt_list[i])
                    self.fwt_list[i] = max(sum(self.fwt_temp_list[1:i + 1]) / i, self.fwt_list[i])

                print('Evaluation: \tEpoch: %d\tTime: %.4f\tTest acc: %.4f\tTotal CL Test acc: %.4f' % (
                    epoch, time.time() - epoch_begin, self.acc_list[i], total_test_acc))

                if total_test_acc >= best_result['acc']:
                    best_result['acc'], best_result['epoch'] = total_test_acc, epoch
                    self.save(self.model, name="model.bin")
                    print('NEW BEST MODEL: \tEpoch: %d\tTest acc: %.4f' % (epoch, total_test_acc))
                self.load(self.model)

        whole_test_acc, whole_beam_acc, (right, wrong, _), write_data = self.epoch_acc(self.task_controller.whole_test)
        return self.acc_list, self.bwt_list, self.fwt_list, whole_test_acc

    def train_one_batch(self, examples, report_loss, example_num):
        score = self.model.forward(examples)
        loss_sketch = -score[0]
        loss_lf = -score[1]

        _loss = torch.sum(loss_sketch).data.item() + torch.sum(loss_lf).data.item()
        #
        loss_sketch = torch.mean(loss_sketch)
        loss_lf = torch.mean(loss_lf)

        loss = loss_lf + loss_sketch

        report_loss += _loss
        example_num += len(examples)
        return report_loss, example_num, loss

    def train_one_epoch(self, examples, optimize_step=True):
        self.model.train()
        # shuffe
        random.shuffle(examples)
        report_loss = 0.0
        st = 0
        example_num = 0

        while st < len(examples):
            ed = st + self.args.batch_size if st + self.args.batch_size < len(examples) else len(examples)

            if optimize_step:
                self.optimizer.zero_grad()

            report_loss, example_num, loss = self.train_one_batch(examples[st:ed], report_loss, example_num)

            # TODO: what is the sup_attention?
            loss.backward()
            if self.args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)

            if optimize_step:
                self.optimizer.step()
            # some records
            st = ed
        return report_loss / len(examples)

    def save(self, model, name="model.bin"):
        torch.save({"model": model.state_dict()},
                   open(os.path.join(self.model_save_path, name), 'wb'))

    def load(self, model, name="model.bin"):
        model.load_state_dict(
            torch.load(open(os.path.join(self.model_save_path, name), "rb"), map_location=self.device)["model"])

    def epoch_acc(self, examples):
        self.model.eval()
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

        for example in examples:

            results_all = self.model.parse(example, beam_size=self.args.beam_size)

            try:
                results = results_all[0]
                sketch_actions = " ".join(str(x) for x in results_all[1])
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
            simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred
            json_datas.append(simple_json)

            example.sql_json['model_result'] = pred
            # print(example.sql_json)

            # example.sql_json['fusion_results'] = list_preds

            # json_datas.append(example.sql_json)

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
                right_result.append((pred, glod, src_str, example.sql))
            else:
                wrong_result.append((pred, glod, src_str, example.sql))

            total_sql += 1

        with open('lf_predict.json', 'w') as f:
            json.dump(json_datas, f)
        # print('sketch acc is ', one_sketch_num / total_sql)
        # print(best_correct / total_sql, beam_correct / total_sql)
        # quit()
        return best_correct / total_sql, beam_correct / total_sql, (right_result, wrong_result, beam_wrong_result), \
               (sel_num, sel_col, agg_col, table_col)
