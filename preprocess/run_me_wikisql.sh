#!/bin/bash

data=../data/wikisql_task_stream_0213/
table_data=../data/wikisql_task_stream_0213/
output=../data/wikisql_task_stream_0213/

echo "Start download NLTK data"
python download_nltk.py
echo "Finish."

echo "Start process the origin WikiSQL dataset"
python data_process.py --data_path ${data} --table_path ${table_data} --output ${output} --mode stream

echo "Start generate SemQL from SQL"
python sql2SemQL.py --data_path ${output} --table_path ${table_data} --output ${data} --mode stream
