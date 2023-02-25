#!/bin/bash

nohup python -u ./src/train_cl.py --task_path ./data/spider_task_stream/task{}/{} \
--cuda \
--epoch 20 \
--beam_size 2 \
--model_name table \
--sentence_features \
--column_pointer \
--warm_boot_epoch 10 \
--st_epoch 15 \
--stu_cl \
--stu_semi \
--tec_cl \
--tec_semi \
--model student_teacher \
--device 0 \
--eval_epoch 20 \
--stu_sampling_name dual \
--tec_sampling_name dual >> nohup.out &