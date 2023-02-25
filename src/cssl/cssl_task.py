import sys
sys.path.append("..")
import random
import os
import json
import time
import numpy as np

from src.cssl.cssl_utils import load_dataset_for_cl
from src.cssl.cssl_sampling import RANDOM, FSS, PRIOR, BALANCE, LFS, DLFS, TEACHER, STUDENT


class TaskController(object):

    def __init__(self, args):
        self.task_list = load_dataset_for_cl(args.task_path, args.task_num, calc_metric_vec=(args.stu_sampling_name == "dual" or args.tec_sampling_name == "dual"))
        self.args = args
        self.task_num = args.task_num
        self.memory_size = self.args.memory_size
        self.memory_list = [{"student": {"train": [], "semi": []}, "teacher": {"train": [], "semi": []}} for i in range(self.args.task_num)]

        if self.args.stu_sampling_name == "random":
            self.function_student = RANDOM
        elif self.args.stu_sampling_name == "fss":
            self.function_student = FSS
        elif self.args.stu_sampling_name == "prior":
            self.function_student = PRIOR
        elif self.args.stu_sampling_name == "balance":
            self.function_student = BALANCE
        elif self.args.stu_sampling_name == "lfs":
            self.function_student = LFS
        elif self.args.stu_sampling_name == "DLFS":
            self.function_student = DLFS
        else:
            self.function_student = None

        if self.args.tec_sampling_name == "random":
            self.function_teacher = RANDOM
        elif self.args.tec_sampling_name == "dual":
            self.function_teacher = TEACHER
        else:
            self.function_teacher = None

        self.total_stu_sampling_time = 0.0
        self.total_tec_sampling_time = 0.0


    def build_memory(self, task_id, model, use_semi=True):
        print("Build Memory with Student: {}, Teacher: {} ...".format(self.args.stu_sampling_name, self.args.tec_sampling_name))
        start_time = time.time()
        if self.args.tec_sampling_name != "dual" and self.args.tec_sampling_name != "none":
            self.memory_list[task_id]["teacher"]["train"] = self.function_teacher(self.task_list[task_id]["train"], self.memory_size, self.args, model) if self.function_teacher else []
            self.memory_list[task_id]["teacher"]["semi"] = self.function_teacher(self.task_list[task_id]["semi"], self.memory_size, self.args, model) if self.function_teacher and use_semi else []
        end_teacher_time = time.time()
        if self.args.stu_sampling_name != "none":
            self.memory_list[task_id]["student"]["train"] = self.function_student(self.task_list[task_id]["train"], self.memory_size, self.args, model) if self.function_student else []
            self.memory_list[task_id]["student"]["semi"] = self.function_student(self.task_list[task_id]["semi"], self.memory_size, self.args, model) if self.function_student and use_semi else []
        end_student_time = time.time()
        print("Time Cost of Teacher Sampling {} is {}".format(self.args.tec_sampling_name, end_teacher_time - start_time))
        print("Time Cost of Student Sampling {} is {}".format(self.args.stu_sampling_name, end_student_time - end_teacher_time))
        self.total_tec_sampling_time += end_teacher_time - start_time
        self.total_stu_sampling_time += end_student_time - end_teacher_time


    def get_memory(self, task_id, use_semi=True):
        print("Get Memory with Student: {}, Teacher: {} ...".format(self.args.stu_sampling_name, self.args.tec_sampling_name))
        memory_student_train, memory_student_semi = [], []
        memory_teacher_train, memory_teacher_semi = [], []
        for i in range(task_id):
            memory_student_train.extend(self.memory_list[i]["student"]["train"])
            memory_student_train.extend(self.memory_list[i]["student"]["semi"] if use_semi else [])

            if self.args.tec_sampling_name != "dual":
                memory_teacher_train.extend(self.memory_list[i]["teacher"]["train"])
                memory_teacher_semi.extend(self.memory_list[i]["teacher"]["semi"] if use_semi else [])
            else:
                start_time = time.time()
                memory_teacher_train.extend(TEACHER(self.task_list[i]["train"], self.task_list[task_id]["semi"], self.memory_size, self.args))
                memory_teacher_semi.extend(TEACHER(self.task_list[i]["semi"], self.task_list[task_id]["semi"], self.memory_size, self.args))
                print("Time Cost of Teacher Sampling Dual is {}".format(time.time() - start_time))
                self.total_tec_sampling_time += time.time() - start_time

        memory_student_semi = [x for x in memory_student_semi if x.pseudo_conf]
        memory_teacher_semi = [x for x in memory_teacher_semi if x.pseudo_conf]

        return memory_student_train, memory_student_semi, memory_teacher_train, memory_teacher_semi


    def get_semi_memory(self, semi_memory, max_len):
        random.shuffle(semi_memory)
        return [x for x in semi_memory[:min(max_len, len(semi_memory))] if x.pseudo_conf]


