import argparse
import os
import time

import pandas as pd
from tqdm import tqdm

#from evaluator.chatgpt_evaluator import ChatGPTEvaluator
from evaluator.baichuan2_evaluator import Baichuan2Evaluator
from evaluator.baichuan_evaluator import BaichuanEvaluator
from evaluator.chatglm2_evaluator import ChatGLM2Evaluator
from evaluator.chatglm_evaluator import ChatGLMEvaluator
from evaluator.llama2_evaluator import LLaMA2Evaluator
from evaluator.llama_evaluator import LLaMAEvaluator
from evaluator.qwen_evaluator import QWenEvaluator
from evaluator.ernie_evaluator import ERNIEEvaluator
from utils.dataloader import load_dataset
from utils.file_utils import save_json


def load_evaluator(model_name_, model_path):
    model_name = model_name_.lower()
    if "chatglm2" in model_name:
        evaluator = ChatGLM2Evaluator(model_path)
    elif "chatglm" in model_name:
        evaluator = ChatGLMEvaluator(model_path)
    elif "llama2" in model_name:
        generation_config = dict(
            temperature=0.01,
            top_k=40,
            top_p=0.7,
            max_new_tokens=1024
        )
        evaluator = LLaMA2Evaluator(generation_config, model_path)
    elif "llama" in model_name:
        max_new_tokens = 1024
        generation_config = dict(
            temperature=0.2,
            top_k=40,
            top_p=0.7,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens
        )
        evaluator = LLaMAEvaluator(max_new_tokens, generation_config, model_path)
    elif model_name == "gpt35":
        evaluator = ChatGPTEvaluator()
    elif "baichuan2" in model_name:
        evaluator = Baichuan2Evaluator(model_path)
    elif "baichuan" in model_name:
        evaluator = BaichuanEvaluator(model_path)
    elif "qwen" in model_name:
        evaluator = QWenEvaluator(model_path)
    elif "eb" in model_name:
        evaluator = ERNIEEvaluator(model_name)
    else:
        print(f'{model_name} 模型暂不支持，请实现对应的 evaluator 类')
        return
    return evaluator


def fineva_main(args):
    model_name = args.model_name
    model_path = args.model_path
    save_path = args.save_path

    # 导入模型
    evaluator = load_evaluator(model_name, model_path)
    print(f'模型 {model_name} 加载成功')

    # 导入数据
    dataset = load_dataset("../data",)

    # 大模型推理
    for data in tqdm(dataset, ncols=100):
        prompt_query = data['prompt']
        # 大模型输出
        model_answer = evaluator.answer(prompt_query)
        if model_name == 'gpt35':
            time.sleep(0.5)
        data[f'{model_name}_answer'] = model_answer


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存为 csv
    try:
        save_json(dataset, os.path.join(save_path, f'{model_name}_ga.json'))
    except Exception as e:
        print(dataset)
        print(f"Error occurred while saving as json: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--model_path', required=False, type=str)
    parser.add_argument('--save_path', required=True, type=str)
    args = parser.parse_args()
    fineva_main(args)
