import requests
import json
from tqdm import tqdm, trange
import random
import numpy as np
import pandas as pd
import time
import re
import argparse
import os

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}".format(os.environ.get("BAIDU_API_KEY"), os.environ.get("BAIDU_SECRET_KEY"))
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def construct_input(input_list):
    api_input = list()
    for i, item in enumerate(input_list):
        if i % 2 == 0:
            api_input.append({
                "role": "user",
                "content": item
            })
        else:
            api_input.append({
                "role": "assistant",
                "content": item
            })
    return api_input

def baidu_generate_eb4(prompt):
    
    assert prompt[-1]['role'] == 'user'
    global prompt_price
    global comple_price
    try:
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    except:
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
        except:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    # print(prompt)
    payload = json.dumps({
        "messages": prompt
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    for _ in range(10):
        try:
            gap = random.uniform(0, 0.5)
            time.sleep(gap)
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.text
        except:
            continue
    return json.dumps({'result': ''})

def baidu_generate_eb3_8k(prompt):
    
    assert prompt[-1]['role'] == 'user'
    global prompt_price
    global comple_price
    try:
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_bot_8k?access_token=" + get_access_token()
    except:
        try:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_bot_8k?access_token=" + get_access_token()
        except:
            url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_bot_8k?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": prompt
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    for _ in range(10):
        try:
            gap = random.uniform(0, 0.5)
            time.sleep(gap)
            response = requests.request("POST", url, headers=headers, data=payload)
            return response.text
        except:
            continue
    return json.dumps({'result': ''})

def gpt_api(query, model_type="eb4"):
    if model_type == 'eb4':
        fn = baidu_generate_eb4
    elif model_type == 'eb3':
        fn = baidu_generate_eb3_8k
    else:
        raise ValueError(f"model_type {model_type} is not supported")
    return json.loads(fn(construct_input([query])))['result']

if __name__ == '__main__':
    print(gpt_api("你是谁", 'eb3'))