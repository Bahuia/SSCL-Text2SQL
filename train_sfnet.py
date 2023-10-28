# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
# @Time    : 2022/8/1
# @Author  : Xinnan Guo & Yongrui Chen
# @File    : train_sfnet.py
# @Software: PyCharm
"""
import os
import datetime
import csv
import sys

import torch.cuda

sys.path.append("..")
from sfnet.trainer import SFNet
from sfnet.args import init_arg_parser


def init_log_checkpoint_path():
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(os.path.curdir, "saved_model", dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path, dir_name

def make_result_file(out_path, avg_acc_list, whole_acc_list, bwt_list, fwt_list):
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + [i for i in range(len(avg_acc_list))])
        writer.writerow(["avg_acc"] + avg_acc_list)
        writer.writerow(["whole_acc"] + whole_acc_list)
        writer.writerow(["bwt"] + bwt_list)
        writer.writerow(["fwt"] + fwt_list)

def run_training():
    args = init_arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_save_path, dir_name = init_log_checkpoint_path()
    print("Current training data will be saved in: {}".format(model_save_path))

    print("#########################################################")
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print("#########################################################\n")

    print("Init trainer ...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    trainer = SFNet(args, model_save_path)

    avg_acc_list, whole_acc_list, bwt_list, fwt_list = trainer.train_with_teacher()
    print("Finish continual training.")

    result_path = "./results/"
    os.makedirs(result_path, exist_ok=True)

    out_path = os.path.join(result_path, "result_{}.csv".format(dir_name))
    make_result_file(out_path, avg_acc_list, whole_acc_list, bwt_list, fwt_list)

    print("Average Accuracy:", [round(x, 3) for x in avg_acc_list])
    print("Whole Accuracy:", [round(x, 3) for x in whole_acc_list])
    print("Backward Transfer:", [round(x, 3) for x in bwt_list])
    print("Forward Transfer:", [round(x, 3) for x in fwt_list])
    print()




if __name__ == "__main__":
    run_training()