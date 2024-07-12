import pandas as pd
import os
import json

def load_dataset(path):
    dataset = []

    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if f.endswith(".json")]:
            subject = dirpath.split("/")[-1]
            path = os.path.join(dirpath, filename)
            domain = filename.split(".")[0]
            frame = json.load(open(path, 'r'))
            if domain in ['股票分析', '基金分析', '行情分析', '行业板块分析', '公告解读', '宏观解读', '事件解读', '行业解读']:
                each_data = load_generation(frame, subject, domain)
            elif domain in ['金融实体识别']:
                each_data = load_recognization(frame, subject, domain)
            elif domain in ['金融业务合规', '信息安全合规']:
                each_data = load_compliance(frame, subject, domain)
            elif domain in ['金融数据检查', '金融数值提取', '金融指标计算', '金融实体消歧', '金融意图理解', '情绪识别', '金融事实', '金融术语']:
                each_data = load_choice(frame, subject, domain)
            dataset += each_data
    return dataset

def load_choice(data, subject, domain):
    dataset = list()
    for data_dict in data:
        if 'E' in data_dict:
            input_data = '{question}\nA、{A}\nB、{B}\nC、{C}\nD、{D}\nE、{E}\n'.format(**data_dict)
            sentence = '你从A、B、C、D、E这几个选项中选出一个作为回答问题的最恰当的答案，你只能输出一个字符，并且这个字符是A、B、C、D、E中一个。'
        elif 'D' in data_dict:
            input_data = '{question}\nA、{A}\nB、{B}\nC、{C}\nD、{D}\n'.format(**data_dict)
            sentence = '你从A、B、C、D这几个选项中选出一个作为回答问题的最恰当的答案，你只能输出一个字符，并且这个字符是A、B、C、D中一个。'
        elif 'C' in data_dict:
            input_data = '{question}\nA、{A}\nB、{B}\nC、{C}\n'.format(**data_dict)
            sentence = '你从A、B、C这几个选项中选出一个作为回答问题的最恰当的答案，你只能输出一个字符，并且这个字符是A、B、C中一个。'
        prompt_constructor = input_data.replace('给出正确选项。\n', '给出正确选项。\n问题：').replace('给出正确选项。', sentence)
        answer = '{answer}'.format(**data_dict)
        id = data_dict['id']
        details = {}
        details["id"] = int(id)
        details["subject"] = subject
        details["domain"] = domain
        details["prompt"] = prompt_constructor
        details["answer"] = answer
        dataset.append(details)
    return dataset

def load_compliance(data, subject, domain):
    dataset = list()
    for data_dict in data:
        prompt_constructor = '{question}'.format(**data_dict)
        answer = 'nan'
        id = data_dict['id']
        details = {}
        details["id"] = int(id)
        details["subject"] = subject
        details["domain"] = domain
        details["prompt"] = prompt_constructor
        details["answer"] = answer
        dataset.append(details)
    return dataset

def load_recognization(data, subject, domain):
    dataset = list()
    for data_dict in data:
        prompt_constructor = '{question}'.format(**data_dict)
        answer = '{answer}'.format(**data_dict)
        id = data_dict['id']
        details = {}
        details["id"] = int(id)
        details["subject"] = subject
        details["domain"] = domain
        details["prompt"] = prompt_constructor
        details["answer"] = answer
        dataset.append(details)
    return dataset

def load_generation(data, subject, domain):
    dataset = list()
    for data_dict in data:
        prompt_constructor = '{question}'.format(**data_dict)
        answer = dict()
        idx = 1
        while True:
            if 'criterium{}'.format(str(idx)) in data_dict:
                answer['criterium{}'.format(str(idx))] = data_dict['criterium{}'.format(str(idx))]
                idx += 1
            else:
                break
        id = data_dict['id']
        details = {}
        details["id"] = int(id)
        details["subject"] = subject
        details["domain"] = domain
        details["prompt"] = prompt_constructor
        details["answer"] = answer
        dataset.append(details)
    return dataset

def split_standard(domain_name):
    targets = ["_dev", "_val", "_test"]
    for item in targets:
        if domain_name.endswith(item):
            domain_name = domain_name.split(item)[0]
    return domain_name

if __name__ == '__main__':
    dataset = load_dataset(path='/mnt/workspace/CFGPT/evaluation/OpenFinData/data')
    print(dataset)