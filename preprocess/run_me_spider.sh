#!/bin/bash

data=../data/spider_task_stream/
table_data=../data/spider_task_stream/
output=../data/spider_task_stream/

#echo "Start download NLTK data"
#python download_nltk.py
#echo "Finish."

echo "Start process the Spider task streams ..."
python data_process.py --data_path ${data} --table_path ${table_data} --output ${output} --mode stream
echo "Finish.\n"

echo "Start generate SemQL from SQL."
python sql2SemQL.py --data_path ${output} --table_path ${table_data} --output ${data} --mode stream
echo "Finish.\n"
