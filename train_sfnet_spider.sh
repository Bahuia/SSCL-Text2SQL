#!/bin/bash

python -u train_sfnet.py \
--task_path ./data/spider_task_stream/task_{}/{} \
--plm_model Salesforce/grappa_large_jnt \
--task_num 10 \
--seed 2023 \
--cuda \
--device 1 \
--batch_size 16 \
--memory_size 30 \
--accumulation_step 2 \
--epoch_num 20 \
--st_epoch_num 15 \
--warm_boot_epoch 20 \
--beam_size 5 \
--max_seq_length 300 \
--column_pointer \
--student_cl \
--student_ssl \
--teacher_cl \
--teacher_ssl \
--agent sfnet \
--eval_epoch 10 \
--student_sampling_name dual \
--teacher_sampling_name dual
