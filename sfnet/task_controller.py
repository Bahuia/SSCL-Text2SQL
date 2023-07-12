# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : sfnet/task_controller.py
# @Software: PyCharm
"""
import sys
import random
sys.path.append("..")
from sfnet.utils import load_sscl_task_stream
from sfnet.sampling import RANDOM, FSS, PRIOR, BALANCE, LFS, DLFS, prompt_sampling, review_sampling


class TaskController(object):

    def __init__(self, args):
        self.task_list = load_sscl_task_stream(args.task_path,
                                               args.task_num,
                                               calc_metric_vec=(args.student_sampling_name == "dual"
                                                                or args.teacher_sampling_name == "dual"))

        self.args = args
        self.task_num = args.task_num
        self.memory_size = self.args.memory_size
        self.memory_list = [{"student": {"train": [], "semi": []},
                             "teacher": {"train": [], "semi": []}}
                            for i in range(self.args.task_num)]

        if self.args.student_sampling_name == "random":
            self.function_student = RANDOM
        elif self.args.student_sampling_name == "fss":
            self.function_student = FSS
        elif self.args.student_sampling_name == "prior":
            self.function_student = PRIOR
        elif self.args.student_sampling_name == "balance":
            self.function_student = BALANCE
        elif self.args.student_sampling_name == "lfs":
            self.function_student = LFS
        elif self.args.student_sampling_name == "dlfs":
            self.function_student = DLFS
        elif self.args.student_sampling_name == "dual":
            self.function_student = review_sampling
        else:
            self.function_student = None

        if self.args.teacher_sampling_name == "random":
            self.function_teacher = RANDOM
        elif self.args.teacher_sampling_name == "dual":
            self.function_teacher = prompt_sampling
        else:
            self.function_teacher = None

    def build_memory(self, task_id, model, use_semi=True):
        if self.args.teacher_sampling_name == "random":
            self.memory_list[task_id]["teacher"]["train"] = self.function_teacher(self.task_list[task_id]["train"],
                                                                                  self.memory_size,
                                                                                  self.args,
                                                                                  model) if self.function_teacher else []
            self.memory_list[task_id]["teacher"]["semi"] = self.function_teacher(self.task_list[task_id]["semi"],
                                                                                 self.memory_size,
                                                                                 self.args,
                                                                                 model) if self.function_teacher and use_semi else []

        if self.args.student_sampling_name != "none":
            self.memory_list[task_id]["student"]["train"] = self.function_student(self.task_list[task_id]["train"],
                                                                                  self.memory_size,
                                                                                  self.args,
                                                                                  model) if self.function_student else []
            self.memory_list[task_id]["student"]["semi"] = self.function_student(self.task_list[task_id]["semi"],
                                                                                 self.memory_size,
                                                                                 self.args,
                                                                                 model) if self.function_student and use_semi else []

    def get_memory(self, task_id, use_semi=True):
        memory_student_train, memory_student_semi = [], []
        memory_teacher_train, memory_teacher_semi = [], []
        for i in range(task_id):
            memory_student_train.extend(self.memory_list[i]["student"]["train"])
            memory_student_train.extend(self.memory_list[i]["student"]["semi"] if use_semi else [])

            if self.args.teacher_sampling_name != "dual":
                memory_teacher_train.extend(self.memory_list[i]["teacher"]["train"])
                memory_teacher_semi.extend(self.memory_list[i]["teacher"]["semi"] if use_semi else [])
            else:
                memory_teacher_train.extend(prompt_sampling(self.task_list[i]["train"],
                                                            self.task_list[task_id]["semi"],
                                                            self.memory_size,
                                                            self.args))
                memory_teacher_semi.extend(prompt_sampling(self.task_list[i]["semi"],
                                                           self.task_list[task_id]["semi"],
                                                           self.memory_size,
                                                           self.args))

        memory_student_semi = [x for x in memory_student_semi if x.pseudo_conf]
        memory_teacher_semi = [x for x in memory_teacher_semi if x.pseudo_conf]

        return memory_student_train, memory_student_semi, memory_teacher_train, memory_teacher_semi

    def get_semi_memory(self, semi_memory, max_len):
        random.shuffle(semi_memory)
        return [x for x in semi_memory[:min(max_len, len(semi_memory))] if x.pseudo_conf]
