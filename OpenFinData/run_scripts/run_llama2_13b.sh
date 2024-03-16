#!/bin/bash

model_name="llama2_13b"
model_path="path/to/Llama-2-13b-chat-hf"
result_path="../results"

cd ../src

python exec_fineva_main.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --save_path ${result_path}

python get_score.py \
    --model_name ${model_name} \
    --result_path ${result_path}
