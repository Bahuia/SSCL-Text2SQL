import os
from cssl.cssl_trainer import CSSLTrainer
from cssl.cssl_args import init_arg_parser
import datetime
import json
import time

def init_log_checkpoint_path():
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(os.path.curdir, "saved_model", dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def run_training():
    args = init_arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_save_path = init_log_checkpoint_path()
    print("Current Training Data Will Be Saved in Path: {}".format(model_save_path))

    print("Init Train Controller ...")
    train_controller = CSSLTrainer(args, model_save_path)
    start_time = time.time()
    if args.model == "naive":
        acc_list, bwt_list, fwt_list, whole_acc_list = train_controller.train_normal()
    elif args.model == "normal":
        acc_list, bwt_list, fwt_list, whole_acc_list = train_controller.train_naive()
    elif args.model == "student_teacher":
        acc_list, bwt_list, fwt_list, whole_acc_list = train_controller.train_student_teacher()
    else:
        print("Wrong Model Name!!!")
        return

    result = {"acc_list": acc_list, "avg_acc": sum(acc_list) / len(acc_list),
              "bwt_list": bwt_list, "avg_bwt": sum(bwt_list[1:]) / len(bwt_list[1:]),
              "fwt_list": fwt_list, "avg_fwt": sum(fwt_list[1:]) / len(fwt_list[1:]),
              "whole_test_acc": whole_acc_list, "avg_whole": sum(whole_acc_list) / len(whole_acc_list)}
    json.dump(result, open(os.path.join(model_save_path, "result.json"), "w", encoding="utf8"), ensure_ascii=False, indent=4)
    print("Finish CSSL !")


if __name__ == "__main__":
    run_training()