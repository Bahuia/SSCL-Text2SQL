#!/bin/bash

python -u train_sfnet.py \
--task_path ./data/wikisql_task_stream/task_{}/{} \
--plm_model Salesforce/grappa_large_jnt \
--task_num 10 \
--seed 17 \
--cuda \
--device 0 \
--batch_size 16 \
--memory_size 200 \
--accumulation_step 2 \
--epoch_num 30 \
--st_epoch_num 15 \
--warm_boot_epoch 60 \
--beam_size 7 \
--max_seq_length 80 \
--column_pointer \
--student_cl \
--teacher_cl \
--agent sfnet \
--eval_epoch 10 \
--student_sampling_name dual \
--teacher_sampling_name dual
