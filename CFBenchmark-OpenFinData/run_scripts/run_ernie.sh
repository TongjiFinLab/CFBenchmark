#!/bin/bash

model_name="eb3" # eb3 for ernie-bot-3.5; eb4 for ernie-bot-4
result_path="../results"

cd ../src

python exec_fineva_main.py \
    --model_name ${model_name} \
    --save_path ${result_path}

python get_score.py \
    --model_name ${model_name} \
    --result_path ${result_path}


