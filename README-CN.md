<div style="text-align:center">
<!-- <img src="https://big-cheng.com/k2/k2.png" alt="k2-logo" width="200"/> -->
<h2>📈 CFBenchmark: Chinese Financial Assistant with Large Language Model</h2>
</div>

<div align="center">

<a href='https://arxiv.org/abs/2311.05812'><img src='https://img.shields.io/badge/Paper-ArXiv-C71585'></a> 
<a href='https://huggingface.co/datasets/TongjiFinLab/CFBenchmark'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-CFBenchmark-red'></a> 
![license](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

[English](README.md) | 简体中文

</div>

# 简介

欢迎来到**CFBenchmark**

近年来，随着大语言模型（LLM）的快速发展，现有的大语言模型在各项任务中都取得了优异的表现。 然而，我们注意到，目前专注于大语言模型在特定领域表现的基准测试数量有限。

“书生•济世”中文金融评测基准（CFBenchmark）基础版本由[CFBenchmark-Basic](https://huggingface.co/datasets/TongjiFinLab/CFBenchmark)和[CFBenchmark-OpenFinData](https://github.com/open-compass/OpenFinData)两部分数据组成，主要包含以下几方面，来评测相关大模型在金融实际应用中的各项能力和安全性：
* 金融自然语言处理，主要关注模型对金融文本的理解和生成能力，如金融实体识别，行业分类，研报总结和风险评估；
* 金融场景计算，侧重于评估模型在特定金融场景下的计算和推理能力，如风险评估和投资组合优化；
* 金融分析与解读任务，检验模型在理解复杂金融报告、预测市场趋势和辅助决策制定方面的能力；
* 金融合规与安全检查，评估模型潜在的合规风险，如生成内容的隐私性、内容安全性、金融合规性等方面的能力。

未来，“书生•济世”中文金融评测基准将继续深化金融大模型评测体系建设，包括大模型在金融行业应用过程中的模型生成内容的准确性、及时性、安全性、隐私性、合规性等能力评估。

<div align="center">
  <img src="imgs/Framework.png" width="100%"/>
  <br />
  <br /></div>

# 更新

\[2024.03.17\] 增加了在金融数据集[CFBenchmark-OpenFinData](https://github.com/open-compass/OpenFinData)上的评测内容，提供了该数据集中对应主观题的一种评测代码实现方式，并测试了9个大模型在[OpenFinData](https://github.com/open-compass/OpenFinData) 数据集上的评测结果。

>  [OpenFinData](https://github.com/open-compass/OpenFinData)数据来源于东方财富与上海人工智能实验室联合发布的开源项目，更多详情：[Github地址](https://github.com/open-compass/opencompass/blob/main/configs/datasets/OpenFinData/OpenFinData.md)。

\[2023.11.10\] 我们发布了[CFBenchmark-Basic](https://huggingface.co/datasets/TongjiFinLab/CFBenchmark)和对应的[技术报告](https://arxiv.org/abs/2311.05812)，主要针对大模型在金融自然语言任务和金融文本生成任务上的能力进行全面评测。

# 目录

- [CFBenchmark-Basic](#cfbenchmark-basic)
- [快速开始](#快速开始)
- [测试结果](#测试结果)
- [致谢](#致谢)
- [未来的工作](#未来的工作)
- [许可证](#许可证)
- [引用](#引用)

# CFBenchmark-Basic

CFBenchmark的基础版本包括3917个金融文本涵盖三个方面和八个任务，从金融识别、金融分类、金融生成三个方面进行组织。
* 识别-公司：识别与财务文件相关的公司名称，共273个。
* 识别-产品：识别与财务文件相关的产品名称，共297个。
* 分类-情感分析：对于财务文件相关的情感类别进行分类，共591个。
* 分类-事件检测：对于财务文件相关的事件类别进行分类，共577个。
* 分类-行业确认：对于财务文件相关的二级行业进行分类，共402个。
* 生成-投资建议：基于提供的财务文件生成投资建议，共593个。
* 生成-风险提示：基于提供的财务文件生成投资建议，共591个。
* 生成-内容总结：基于提供的财务文件生成投资建议，共593个。

我们提供了两个模型，展示了零样本（Zero-shot）和少样本（Few-shot）是如何进行测试的。

样例1 少样本（Few-shot）的输入：
<div align="center">
  <img src="imgs/fewshot.png" width="100%"/>
  <br />
  <br /></div>

样例2 零样本（Zero-shot）的输入：
<div align="center">
  <img src="imgs/zeroshot.png" width="100%"/>
  <br />
  <br /></div>

# 快速开始

## 安装

以下展示了一个安装的简单步骤。
 ```python
    conda create --name CFBenchmark python=3.10
    conda activate CFBenchmark
 ```

```python
    git clone https://github.com/TongjiFinLab/CFBenchmark
    cd CFBenchmark
    pip install -r requirements.txt
```

## 测评

### CFBenchmark-Basic

我们在 ```CFBenchmark-Basic/src``` 中为您准备了测试和评估代码。

为了运行测评，您可以在命令行中运行以下代码：

```cmd
cd CFBenchmark-Basic/src
python -m run.py
```

您可以进入```CFBenchmark-Basic/src/run.py```来修改其中的参数，让代码运行的路径符合您的要求。

```py
from CFBenchmark import CFBenchmark
if __name__=='__main__':

    # EXPERIMENT SETUP
    modelname = 'YOUR-MODEL-NAME'
    model_type= 'NORMAL' #NORMAL or LoRA
    model_path= 'YOUR-MODEL-PATH'
    peft_model_path= ''#PASS YOUR OWN PATH OF PEFT MODEL IF NEEDED
    fewshot_text_path= '../fewshot'#DEFAULT PATH
    test_type='few-shot'#LET'S TAKE THE FEW-SHOT TEST AS AN EXAMPLE
    response_path='../cfbenchmark-response'#PATH TO RESERVE THE RESPONSE OF YOUR MODEL
    scores_path='../cfbenchmark-scores'	#PATH TO RESERVE THE SCORE OF YOUR MODEL
    embedding_model_path='../bge-zh-v1.5' #PASS YOUR OWN PATH OF BGE-ZH-V1.5
    benchmark_path='../data' #DEFAULT PATH

    #generate Class CFBenchmark
    cfb=CFBenchmark(
        model_name=modelname,
        model_type=model_type,
        model_path=model_path,
        peft_model_path=peft_model_path,
        fewshot_text_path=fewshot_text_path,
        test_type=test_type,
        response_path=response_path,
        scores_path=scores_path,
        embedding_model_path=embedding_model_path,
        benchmark_path=benchmark_path,
    )
    
    cfb.generate_model()# TO GET RESPONSE FROM YOUR MODEL
    cfb.get_test_scores()# TO GET YOUR MODEL SCORES FROM RESPONSE
```

我们在```codes/CFBenchmark.py```中定义了一个类“CFBenchmark”来进行评估。

```Py
class CFBenchmark:
    def __init__(self,
                 model_name,
                 model_type,
                 model_path,
                 peft_model_path,
                 fewshot_text_path,
                 test_type,
                 response_path,
                 scores_path,
                 embedding_model_path,
                 benchmark_path
                 ) -> None:
```

* 您可以使用参数来设置模型的路径。 如果你想使用进行LoRA微调后的模型，请将``model_type``设置为````LoRA````并通过````peft_model_path```传递你的peft模型路径。
* 您可以将``test-type``设置为'zero-shot'或'few-shot'来进行不同的评估。
* 为“bzh-zh-v1.5”设置“embedding_model_path”，用于计算余弦相似度。
* 您可以修改“CFBenchmark.generate_model()”中的超参数来生成文本。
* 我们在Hugging Face和Github中都提供了保存为Dataset数据类型的CFBenchmark。

### CFBenchmark-OpenFinData

我们在```CFBenchmark-OpenFinData``` 中为您准备了测试和评估的代码与数据。
评测代码的设计与Fineva1.0相似，通过```CFBenchmark-OpenFinData/src/evaluator```对于评测模型的调用方式进行定义，并通过```CFBenchmark-OpenFinData/run_scripts```中的bash文件对于关键参数进行配置和实验。

为了运行测评，您可以在命令行中运行以下代码：

```cmd
cd CFBenchmark-OpenFinData/run_scripts
sh run_baichuan2_7b.sh
```

值得注意的是，因为OpenFinData的评测过程涉及主观题的判断，因此我们的评测框架借助了文心一言来对金融解读与分析类问题和金融合规类问题进行评测。为了顺利试用文心一言的API参与评测，请您在环境变量中设置```BAIDU_API_KEY```和```BAIDU_SECRET_KEY```，以便于```CFBenchmark-OpenFinData/src/get_score.py```的```get_access_token```函数可以顺利运行。

```Py
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
```


# 测试结果

我们使用两种类型的指标来评估金融领域大语言模型在 CFBenchmark 上的表现。
对于CFBenchmark-Basic中的识别和分类任务，我们采用 **F1_score** 作为评估指标，平衡了精度和召回率。 

对于CFBenchmark-Basic中的生成任务，我们利用地面实况的向量（通过**bge-zh-v1.5**生成）表示和生成的答案之间的**余弦相似度**来衡量生成能力。 

对于CFBenchmark-OpenFinData中的knowledge, calculation, 和identification任务，我们直接计算多项选择题的准确率进行模型效果评估。

对于CFBenchmark-OpenFinData中的explanation, analysis, 和compliance任务，我们利用文心一言4作为打分器，来判断模型生成结果和真实答案之间的正确性。

大模型的表现如下表所示：

## CFBenchmark-Basic
| Model               | Size | Company   | Product   | R.Avg     | Sector    | Event     | Sentiment | C.Avg     | Summary   | Risk      | Suggestion | G.Avg     | Avg       |
| ------------------  | ---- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- | --------- | --------- |
| GPT-3.5             | -    | 79.7      | 19.8      | 49.8      | 45.3      | 45.8      |  42.5     | 45.5      | 59.3      | 54.1      | 77.1       | 63.5      | 52.9     |
| GPT-4               | -    | 83.3      | 38.2      | 60.8      | 48.2      | 50.1      |  49.9     | 49.4      | 65.3      | 60.2      | 79.2       | 68.2      | 59.4     |
| ERNIE-Bot-3.5       | -    | 80.7      | 30.0      | 53.3      | 40.8      | 35.0      |  18.6     | 31.5      | 71.5      | 59.0      | 71.6       | 67.3      | 50.7     |
| ERNIE-Bot-4         | -    | 81.9      | 41.7      | 61.8      | 41.8      | 35.8      |  37.5     | 38.4      | 72.1      | 62.9      | 71.8       | 68.9      | 56.4     |
| ChatGLM2-6B         | 6B   | 74.7      | 31.3      | 53.0      | 28.5      | 30.0      |  35.7     | 31.4      | 65.7      | 45.4      | 67.1       | 59.4      | 47.9     |
| ChatGLM3-6B         | 6B   | 75.1      | 25.2      | 50.2      | 33.5      | 32.7      |  39.7     | 35.3      | 68.4      | 53.6      | 70.5       | 64.2      | 49.9     |
| GLM4-9B-Chat        | 9B   | 81.3      | 26.1      | 53.7      | 49.6      | 51.5      |  47.6     | 49.6      | 73.5      | 62.4      | 72.6       | 69.5      | 57.6     |
| Qwen-Chat-7B        | 7B   | 76.3      | 36.0      | 56.2      | 40.0      | 36.7      |  26.5     | 34.4      | 54.8      | 30.7      | 37.9       | 41.1      | 43.9     |
| Qwen1.5-Chat-7B     | 7B   | 83.5      | 35.3      | 59.4      | 34.3      | 37.5      |  51.6     | 41.1      | 73.7      | 58.7      | 73.1       | 68.5      | 56.3     |
| Qwen2-Chat-7B       | 7B   | 82.4      | 34.8      | 58.6      | 54.4      | 49.9      |  41.1     | 48.5      | 75.0      | 55.9      | 76.9       | 69.2      | 58.8     |
| Baichuan2-7B-Chat   | 7B   | 75.7      | 40.2      | 57.9      | 42.5      | 47.5      |  32.3     | 40.8      | 72.5      | 64.8      | 73.2       | 70.2      | 56.3     |
| Baichuan2-13B-Chat  | 13B  | 79.7      | 31.4      | 55.6      | 47.2      | 50.7      |  38.7     | 45.5      | 73.9      | 63.4      | 74.6       | 70.6      | 57.2     |
| InternLM2-7B-Chat   | 7B   | 75.7      | 19.5      | 47.6      | 46.4      | 28.4      |  42.2     | 39.0      | 73.7      | 54.3      | 74.9       | 67.6      | 51.4     |
| InternLM2-20B-Chat  | 20B  | 74.2      | 27.6      | 50.9      | 48.4      | 32.4      |  37.4     | 39.4      | 73.2      | 58.0      | 74.1       | 68.4      | 52.9     |
| InternLM2.5-7B-Chat | 7B   | 75.2      | 24.3      | 49.8      | 53.1      | 34.3      |  45.7     | 44.4      | 74.5      | 57.0      | 73.2       | 68.2      | 54.1     |

## CFBenchmark-OpenFinData

| Model               | Size | Knowledge | Caluation | Explanation | Identification | Analysis | Compliance | Average | 
| ------------------  | ---- | -------   | ------    | -----       | ---------      | -----    | -------    | -----   |
| GPT-3.5             | -    | 77.2      | 68.8      | 81.9        | 76.3           | 75.1     | 35.8       | 63.9    | 
| GPT-4               | -    | 89.2      | 77.2      | 84.4        | 76.9           | 82.5     | 39.2       | 74.9    | 
| ERNIE-Bot-3.5       | -    | 78.0      | 70.4      | 82.1        | 75.3           | 77.7     | 36.7       | 70.0    | 
| ERNIE-Bot-4         | -    | 87.3      | 73.6      | 84.3        | 77.0           | 79.1     | 37.3       | 73.1    | 
| ChatGLM2-6B         | 6B   | 62.4      | 37.2      | 70.8        | 59.2           | 58.3     | 38.7       | 54.4    | 
| ChatGLM3-6B         | 6B   | 66.5      | 38.0      | 76.5        | 61.5           | 60.1     | 32.0       | 55.8    | 
| GLM4-9B-Chat        | 9B   | 81.8      | 56.9      | 79.3        | 63.5           | 78.2     | 29.5       | 64.9    | 
| Qwen-Chat-7B        | 7B   | 71.3      | 40.5      | 71.4        | 58.6           | 51.3     | 40.0       | 55.5    | 
| Qwen1.5-Chat-7B     | 7B   | 67.3      | 53.9      | 84.6        | 67.7           | 76.8     | 30.0       | 63.3    | 
| Qwen2-Chat-7B       | 7B   | 82.5      | 61.3      | 84.2        | 69.8           | 80.1     | 19.3       | 66.2    | 
| Baichuan2-7B-Chat   | 7B   | 46.2      | 37.0      | 76.5        | 60.2           | 55.0     | 28.7       | 50.6    | 
| Baichuan2-13B-Chat  | 13B  | 69.3      | 39.5      | 75.3        | 65.7           | 62.0     | 31.3       | 57.2    | 
| InternLM2-7B-Chat   | 7B   | 70.2      | 39.9      | 73.4        | 62.8           | 61.4     | 39.5       | 57.8    |
| InternLM2-20B-Chat  | 20B  | 76.4      | 52.6      | 76.3        | 66.2           | 63.9     | 42.1       | 62.9    |
| InternLM2.5-7B-Chat | 7B   | 80.7      | 66.6      | 85.0        | 71.7           | 83.1     | 35.4       | 70.4    |

## CFBenchmark 

| Model               | Size | 金融自然语言 | 金融场景计算 | 金融分析与解读| 金融合规与安全 | 平均 |
| ------------------  | ---- | ---------- | ---------- | ----------- | ----------- | ---- |
| GPT-3.5             | -    | 52.9       | 74.1       | 78.5        | 35.8        | 60.3 |
| GPT-4               | -    | 59.4       | 83.5       | 83.5        | 39.2        | 66.4 |
| ERNIE-Bot-3.5       | -    | 50.7       | 74.5       | 79.9        | 36.7        | 60.4 |
| ERNIE-Bot-4         | -    | 56.4       | 82.8       | 81.7        | 37.3        | 64.6 |
| ChatGLM2-6B         | 6B   | 47.9       | 64.1       | 64.6        | 38.7        | 53.8 |
| ChatGLM3-6B         | 6B   | 49.9       | 68.2       | 68.3        | 32.0        | 54.6 |
| GLM4-9B-Chat        | 9B   | 57.6       | 67.4       | 78.8        | 29.5        | 58.3 |
| Qwen-Chat-7B        | 7B   | 43.9       | 67.1       | 61.4        | 40.0        | 53.1 |
| Qwen1.5-Chat-7B     | 7B   | 56.3       | 73.2       | 80.7        | 30.0        | 60.0 |
| Qwen2-Chat-7B       | 7B   | 58.8       | 78.8       | 82.2        | 19.3        | 59.8 |
| Baichuan2-7B-Chat   | 7B   | 56.3       | 61.0       | 65.8        | 28.7        | 53.0 |
| Baichuan2-13B-Chat  | 13B  | 57.2       | 70.1       | 68.6        | 31.3        | 56.8 |
| InternLM2-7B-Chat   | 7B   | 51.4       | 68.8       | 67.4        | 39.5        | 56.8 |
| InternLM2-20B-Chat  | 20B  | 52.9       | 73.0       | 70.1        | 42.1        | 59.5 |
| InternLM2.5-7B-Chat | 7B   | 54.1       | 79.1       | 84.0        | 35.4        | 63.2 |



# 致谢
CFBenchmark 研发过程中参考了以下开源项目。 我们向项目的研究人员表示尊重和感谢。

- tiiuae/falcon LLM series(https://huggingface.co/tiiuae/falcon-7b)
- bigscience/bloomz LLM series(https://huggingface.co/bigscience/bloomz-7b1)
- QwenLM/Qwen LLM series(https://github.com/QwenLM/Qwen)
- THUDM/ChatGLM2-6b(https://github.com/THUDM/ChatGLM2-6B)
- baichuan-inc/Baichuan2 LLM series(https://github.com/baichuan-inc/Baichuan2)
- InternLM/InternLM LLM series(https://github.com/InternLM/InternLM)
- ssymmetry/BBT-FinCUGE-Applications(https://github.com/ssymmetry/BBT-FinCUGE-Applications)
- chancefocus/PIXIU(https://github.com/chancefocus/PIXIU)
- SUFE-AIFLM-Lab/FinEval(https://github.com/SUFE-AIFLM-Lab/FinEval)
- alipay/financial_evaluation_dataset(https://github.com/alipay/financial_evaluation_dataset)
- open-compass/OpenFinData(https://github.com/open-compass/OpenFinData)
- QwenLM/Qwen(https://github.com/QwenLM/Qwen)

# 未来的工作
- [ ] 针对中文金融使用中各种场景，提出更多的评测任务，丰富CFBenchmark系列基准。

# 许可证
CFBenchmark是一项仅用于非商业使用的研究预览，受OpenAI生成数据的使用条款约束。如果您发现任何潜在的风险行为，请与我们联系。该代码发布在Apache License 2.0下。

# 感谢我们的贡献者 ：
<a href="https://github.com/TongjiFinLab/CFBenchmark/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TongjiFinLab/CFBenchmark" />
</a>

# 引用

```bibtex
@misc{lei2023cfbenchmark,
      title={{CFBenchmark}: Chinese Financial Assistant Benchmark for Large Language Model}, 
      author={Lei, Yang and Li, Jiangtong and Cheng, Dawei and Ding, Zhijun and Jiang, Changjun},
      year={2023},
      eprint={2311.05812},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

