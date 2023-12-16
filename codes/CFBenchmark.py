import os
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset,load_from_disk
import torch
import argparse
import pickle

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
                 benchmark_path,
                 data_source_type
                 ) -> None:
        self.model_path=model_path

        self.classifications=['company','product',
                            'sector','event','sentiment',
                            'summary','risk','suggestion']

        
        self.modelname=model_name
        self.model_type=model_type
        self.peft_model_path=peft_model_path
        self.fewshot_text_path=fewshot_text_path
        self.test_type=test_type
        self.response_path=response_path
        self.scores_path=scores_path
        self.embedding_model_path=embedding_model_path
        self.data_source_type=data_source_type       
        self.benchmark_path=benchmark_path

        self.fewshot_text={}
        if test_type=='few-shot':
            for item in self.classifications:
                filename='fewshot-'+item+'.txt'
                with open(os.path.join(fewshot_text_path,filename), 'r',encoding='utf-8') as file:
                    content = file.read()
                    self.fewshot_text[item]=content

        self.t2v_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_path)
        self.t2v_model = AutoModel.from_pretrained(
            self.embedding_model_path,
            load_in_8bit = False,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
        )
        self.t2v_model.eval()

        labels={}
        with open("../labels_info.pkl",'rb')as file:
            labels=pickle.load(file)

        self.labels=labels
    
    def generate_model(self):
        if self.model_type !='LoRA':    
            model_dir=self.model_path
            if self.modelname =='chatglm2-6b':
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_dir,
                    load_in_8bit = False,
                    trust_remote_code=True,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16
                )
                self.model = self.model.eval()
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    load_in_8bit=False,
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype=torch.float16
                ).to('cuda:0')
                self.model = self.model.eval()
            
        else:
            base_model = self.model_path
            peft_model_path = self.peft_model_path
            self.model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit = False,
            trust_remote_code=True,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16
            )
            self.model = PeftModel.from_pretrained(base_model,peft_model_path)
            self.model = self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print('getting {} response'.format(os.path.join(self.model_path,self.modelname)))   
        self.get_model_results()
        
    def get_row_response(self,model,tokenizer,row,classes,types):
        context=row['input']    
        instruction=''
        if types=='zero-shot':
            instruction=row['instruction']+context
        else:
            instruction=self.fewshot_text[classes]
            case='\ncase4：\n新闻内容：'+context
            if classes=='sector' or  classes=='event' or  classes=='sentiment':
                labels=row['instruction'].split('（',1)[1]
                labels=labels.split('）',1)[0]
                case=case+'\n类别：（'+labels+'）\n'
            instruction=instruction+case

        instruction=instruction+'\n回答：'
        inputs=None
        inputs = tokenizer(instruction, return_tensors='pt',max_length=8191).to('cuda:0')
        out=''    

        if classes=='summmary' or classes=='suggestion' or classes=='risk':
            repe_pena=1.02
            if types=='few-shot':
                repe_pena=1.05
            out=model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=repe_pena,
            )
        else:
            repe_pena=1.00
            if types=='few-shot':
                repe_pena=1.03
            out=model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=repe_pena,
            )
        generated_text = tokenizer.decode(out.cpu()[0], skip_special_tokens=True)
        if types=='zero-shot':
            generated_text=generated_text.split('回答：',1)[-1]
        else:
            generated_text=generated_text.split('回答：',4)[-1]
        generated_text=generated_text.split('\n',1)[0].strip()
        return generated_text

    def get_model_results(self):
        save_dir= os.path.join(self.response_path,self.test_type)
        save_dir=os.path.join(save_dir,self.modelname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for item in self.classifications:
            print('dealing {}'.format(item))
            if self.data_source_type=='offline':
                dataset=load_from_disk(self.benchmark_path)
            else:
                dataset=load_dataset(self.benchmark_path)
            dataset=dataset[item]
            df=dataset.to_pandas()
            df['output']=df.apply(lambda row: self.get_row_response(self.model,self.tokenizer,row,item,self.test_type),
                                      axis=1)
            df=df[['input','response','output']]
            filename=item+'-output.csv'
            savepath=os.path.join(save_dir,filename)
            df.to_csv(savepath)

    def get_y(self,row,label_list):
        y_true=np.zeros((len(label_list)+1,1))
        y_pred=np.zeros((len(label_list)+1,1))
        response=set([item.strip() for item in str(row['response']).replace('，', ',').strip().split(',') if item])
        output=set([item.strip() for item in str(row['output']).replace('，', ',').strip().split(',') if item])   

        for i in range(len(label_list)):
            if label_list[i] in response:
                y_true[i]=1
            if label_list[i] in output:
                y_pred[i]=1
        
        if y_pred.sum()==0 or len(output)>y_pred.sum():
            y_pred[-1]=1
        return y_true,y_pred

    def get_f1_score(self,row,label_list):
        y_true,y_pred=self.get_y(row,label_list=label_list)
        prec = (y_true * y_pred).sum() / y_true.sum()
        reca = (y_true * y_pred).sum() / y_pred.sum()
        if prec == 0 or reca == 0:
            f1 = 0
        else:
            f1 = 2 * prec * reca / (prec+reca)
        return f1

    def get_cosine_similarities(self,row):
        sentences_1 = str(row['output'])
        sentences_2 = str(row['response'])
        try:
            encoded_input = self.t2v_tokenizer([sentences_1,sentences_2], padding=True, truncation=True, return_tensors='pt',max_length=512).to('cuda:0')
        except Exception as e:
            print(f"An exception occurred: {str(e)}")
            return 0

        with torch.no_grad():
            model_output = self.t2v_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        cosine_sim = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)
        return cosine_sim.item()

    def get_test_scores(self):
        result_directory = os.path.join(self.scores_path,self.test_type, self.modelname)
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        for classes in self.classifications:
            filename=classes+'-output.csv'
            response_path=os.path.join(response_path,self.test_type,self.modelname,filename)
            df=pd.read_csv(response_path)
            if classes=='suggestion' or classes=='summary' or classes=='risk':
                df['cosine_s']=df.apply(lambda row:self.get_cosine_similarities(row),
                                        axis=1)
                score1=df['cosine_s'].sum()/len(df)
                print("{}的{} cosine_similarity为{}".format(self.modelname,classes,score1))
            elif classes=='company' or classes=='product':
                df['f1score']=df.apply(lambda row:self.get_f1_score(row,row['response'].split('，')),
                                        axis=1)
                score1=df['f1score'].sum()/len(df)
                print("{}的{} f1 score 为{}".format(self.modelname,classes,score1))
            else:
                df['f1score']=df.apply(lambda row:self.get_f1_score(row,self.labels[classes]),
                                        axis=1)
                score1=df['f1score'].sum()/len(df)
                print("{}的{} f1 score 为{}".format(self.modelname,classes,score1))
            filename=classes+'-scores.csv'
            df.to_csv(os.path.join(result_directory,filename))

