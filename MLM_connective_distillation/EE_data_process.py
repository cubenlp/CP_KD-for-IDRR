from copyreg import pickle
import os
from re import I
import sys
import json
from collections import defaultdict
from debugpy import connect
from tqdm import tqdm
from transformers import DataProcessor,BertTokenizerFast
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import pandas as pd
import heapq
from collections import defaultdict
import pickle



class EEProcessor(DataProcessor):
    """
        从数据文件读取数据，生成训练数据dataloader，返回给模型
    """
    def __init__(self, config, tokenizer=None):
        self.train_data = self.read_data(config.train_path)
        # exp_train_data = self.read_data(config.exp_train_path)
        # exp_train_data = exp_train_data[exp_train_data["subtype_label"].isin(["Comparison.Concession","Contingency.Pragmatic cause","Expansion.Alternative","Expansion.List",'Temporal.Synchrony', 'Temporal.Asynchronous.Succession','Temporal.Asynchronous.Precedence'])]
        # exp_train_data = exp_train_data[:int(len(exp_train_data)*0.25)]

        # self.train_data = pd.concat([train_data,exp_train_data],ignore_index=True)
        self.dev_data = self.read_data(config.dev_path) 
        self.test_data = self.read_data(config.test_path)
        self.data_type = config.data_type
        
        self.tokenizer = tokenizer
        self.template_version = config.template_version

        if self.data_type in ["pdtb2_top","pdtb2_second"]:

            self.subtype_label = ['Comparison.Concession', 'Comparison.Contrast', 'Comparison',
                    'Contingency.Cause.Reason','Contingency.Cause.Result', 'Contingency.Pragmatic cause',
                    'Expansion.Alternative', 'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.List', 'Expansion.Restatement','Expansion',
                    'Temporal.Synchrony', 'Temporal.Asynchronous.Succession','Temporal.Asynchronous.Precedence'
                    ]

            self.subtype_label_word = ['although','but', 'but',
                        'because','so','since',
                        'instead','and','instance','first','specifically', 'and', 
                        'simultaneously', 'previously','then'  
                    ]
            
            self.ans_word = ['nevertheless', 'but', 'however', 'although',
                'therefore', 'thus', 'because', 'so','since',
                'instead', 'or', 'furthermore', 'specifically', 'and', 'first', 'instance',
                'simultaneously', 'previously', 'then']

            self.Token_id = [21507, 53, 959, 1712,
                3891, 4634, 142, 98, 187,
                1386, 50, 43681, 4010, 8, 78, 4327,
                11586, 1433, 172] #deberta/roberta
        elif self.data_type=="conll2016":
            self.subtype_label = [
                "Comparison.Contrast","Comparison.Concession",
                "Contingency.Cause.Reason","Contingency.Cause.Result",
                "Expansion.Alternative.Chosen alternative","Expansion.Conjunction","Expansion.Instantiation","Expansion.Restatement",
                "Temporal.Asynchronous.Precedence","Temporal.Asynchronous.Succession","Temporal.Synchrony",
                # "EntRel"
                ]
            self.subtype_label_word = [
                    "but","although",
                    "because","so",
                    "instead","and","example","specifically",
                    "then","previously","meanwhile",
                    # " "
                ]
            self.ans_word = [
                'although', 'but', 'however',
                'so', 'thus', 'consequently', 'because', 'as',
                'instead','rather','specifically', 'instance', 'example', 'and', 'while',
                'meanwhile', 'previously', 'then',
                # ' '
            ]

            self.Token_id = [1712, 53, 959,
                98, 4634, 28766, 142, 25,
                1386, 1195, 4010, 4327, 1246, 8, 150,
                6637, 1433, 172]

        else:

            # new 18
            self.subtype_label = ['Comparison.Similarity', 'Comparison.Contrast', 'Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct', 'Comparison.Concession.Arg2-as-denier', 'Comparison.Concession.Arg1-as-denier', 
                 'Contingency.Purpose.Arg2-as-goal', 'Contingency.Purpose.Arg1-as-goal', 'Contingency.Condition+SpeechAct', 'Contingency.Condition.Arg2-as-cond', 'Contingency.Condition.Arg1-as-cond', 'Contingency.Cause+SpeechAct.Result+SpeechAct', 'Contingency.Cause+SpeechAct.Reason+SpeechAct', 'Contingency.Cause+Belief.Result+Belief', 'Contingency.Cause+Belief.Reason+Belief', 'Contingency.Cause.Result', 'Contingency.Cause.Reason', 
                 'Expansion.Substitution.Arg2-as-subst', 'Expansion.Manner.Arg2-as-manner', 'Expansion.Manner.Arg1-as-manner', 'Expansion.Level-of-detail.Arg2-as-detail', 'Expansion.Level-of-detail.Arg1-as-detail', 'Expansion.Instantiation.Arg2-as-instance', 'Expansion.Instantiation.Arg1-as-instance', 'Expansion.Exception.Arg2-as-excpt', 'Expansion.Exception.Arg1-as-excpt', 'Expansion.Equivalence', 'Expansion.Disjunction', 'Expansion.Conjunction', 
                 'Temporal.Synchronous', 'Temporal.Asynchronous.Succession', 'Temporal.Asynchronous.Precedence']

            self.subtype_label_word = ['similarly', 'but', 'but', 'however', 'although',
                                'for', 'for', 'if', 'if', 'if', 'so','because', 'so', 'because', 'so', 'because', 
                                'instead', 'by', 'thereby', 'specifically', 'specifically', 'instance', 'instance', 'and', 'and', 'namely', 'and', 'and',
                                'simultaneously', 'previously', 'then']

            self.ans_word = ['similarly', 'but', 'however', 'although',
                        'for', 'if', 'because', 'so',
                        'instead', 'by', 'thereby', 'specifically', 'and', 'instance','namely',
                        'simultaneously', 'previously', 'then']

            self.Token_id = [11401, 53, 959, 1712,
                13, 114, 142, 98, 
                1386, 30, 12679, 4010, 8,4327,13953,
                11586, 1433, 172]


        
        self.ans2secondlabel = [
            'Comparison.Concession','Comparison.Contrast','Comparison.Contrast','Comparison.Concession',
            'Contingency.Cause','Contingency.Cause','Contingency.Cause','Contingency.Cause', 'Contingency.Pragmatic cause',
            'Expansion.Alternative','Expansion.Alternative','Expansion.Conjunction','Expansion.Restatement','Expansion.Conjunction','Expansion.List','Expansion.Instantiation',
            'Temporal.Synchrony', 'Temporal.Asynchronous','Temporal.Asynchronous'
        ]
        self.ans2toplabel = [
            'Comparison','Comparison','Comparison','Comparison',
            'Contingency','Contingency','Contingency','Contingency', 'Contingency',
            'Expansion','Expansion','Expansion','Expansion','Expansion','Expansion','Expansion',
            'Temporal', 'Temporal','Temporal'
        ]
        self.ans2conll16label = [
            'Comparison.Concession','Comparison.Contrast','Comparison.Contrast',
            'Contingency.Cause.Result','Contingency.Cause.Result','Contingency.Cause.Result','Contingency.Cause.Reason', 'Contingency.Cause.Reason',
            'Expansion.Alternative.Chosen alternative','Expansion.Alternative.Chosen alternative','Expansion.Restatement','Expansion.Instantiation','Expansion.Instantiation','Expansion.Conjunction','Expansion.Conjunction',
            'Temporal.Synchrony', 'Temporal.Asynchronous.Succession','Temporal.Asynchronous.Precedence',
            # 'EntRel'
        ]


        # new 18
        self.ans2pdtb3toplabel = [
            'Comparison','Comparison','Comparison','Comparison',
            'Contingency','Contingency','Contingency','Contingency', 
            'Expansion','Expansion','Expansion','Expansion','Expansion','Expansion','Expansion',
            'Temporal', 'Temporal','Temporal'
        ]

        self.ans2pdtb3secondlabel = [
            'Comparison.Similarity','Comparison.Contrast','Comparison.Concession','Comparison.Concession',
            'Contingency.Purpose','Contingency.Condition','Contingency.Cause','Contingency.Cause', 
            'Expansion.Substitution','Expansion.Manner','Expansion.Manner','Expansion.Level-of-detail','Expansion.Conjunction','Expansion.Instantiation','Expansion.Equivalence',
            'Temporal.Synchronous', 'Temporal.Asynchronous','Temporal.Asynchronous'
        ]

        self.Comp = ['nevertheless', 'but', 'however', 'although']
        self.Cont = ['therefore', 'thus', 'because', 'so', 'since']
        self.Expa = ['instead', 'or', 'furthermore', 'specifically', 'and', 'first', 'instance']
        self.Temp = ['simultaneously', 'previously', 'then']  

        self.label2ids, self.id2labels,self.rel_label2ids, self.rel_id2labels = self._load_schema()


    def _load_schema(self):
        label2ids = {}
        id2labels = {}
        rel_label2ids = {}
        rel_id2labels = {}
        # label2ids = {"0":0}
        # id2labels = {0:"0"}
          
        if self.data_type in ["pdtb2_top","pdtb3_top"]:
            type_list = [ 
            'Comparison',
            'Contingency',
            'Expansion',
            'Temporal',]
        elif self.data_type == "pdtb2_second":
            type_list = [
            "Comparison.Contrast",
		    "Comparison.Concession",
            "Contingency.Cause",
            "Contingency.Pragmatic cause",
            "Expansion.Alternative",
            "Expansion.Conjunction",
            "Expansion.Instantiation",
            "Expansion.List",
            "Expansion.Restatement",
            "Temporal.Asynchronous",
            "Temporal.Synchrony"
            ]
        elif self.data_type == "pdtb3_second":
            type_list=[
                'Comparison.Concession', 
                # 'Comparison.Concession+SpeechAct', 
                'Comparison.Contrast', 
                'Comparison.Similarity', 
                'Contingency.Cause', 
                'Contingency.Cause+Belief', 
                'Contingency.Cause+SpeechAct', 
                'Contingency.Condition', 
                # 'Contingency.Condition+SpeechAct', 
                'Contingency.Purpose', 
                'Expansion.Conjunction', 
                'Expansion.Disjunction', 
                'Expansion.Equivalence', 
                # 'Expansion.Exception', 
                'Expansion.Instantiation', 
                'Expansion.Level-of-detail', 
                'Expansion.Manner', 
                'Expansion.Substitution', 
                'Temporal.Asynchronous', 
                'Temporal.Synchronous']
        else:
            type_list = [
            "Comparison.Contrast",
            "Comparison.Concession",
            "Contingency.Cause.Reason",
            "Contingency.Cause.Result",
            # "Contingency.Condition",
            # "Expansion.Alternative",
            "Expansion.Alternative.Chosen alternative",
            "Expansion.Conjunction",
            # "Expansion.Exception",
            "Expansion.Instantiation",
            "Expansion.Restatement",
            "Temporal.Asynchronous.Precedence",
            "Temporal.Asynchronous.Succession",
            "Temporal.Synchrony",
            # "EntRel"
            ]

        self.type_list = type_list

        for index,role in enumerate(self.ans_word):
            label2ids[role] = index
            id2labels[index] = role

        for index,role in enumerate(self.type_list):
            rel_label2ids[role] = index
            rel_id2labels[index] = role 

        return label2ids,id2labels,rel_label2ids,rel_id2labels


    def read_data(self,path):
        data = pd.read_csv(path,sep="\t")
        return data

    def process_data_from_csv(self,data):
        # text1 = []
        # text2 = []
        mask_text = []
        # connective = []
        label_conn = []
        label = []

        text = [] # teacher model text
        print(self.template_version,type(self.template_version))

        for i,x in data.iterrows():
          
            if self.data_type not in ["pdtb2_second","pdtb3_second"]:
                # label.append(self.label2ids[x["label"]])
                label_word = x["label"]
            else:
                curr_label = ".".join(x["subtype_label"].split(".")[:2])
                if curr_label not in self.rel_label2ids:
                    continue
                # label.append(self.label2ids[curr_label])
                label_word = curr_label

            if x["subtype_label"] not in self.subtype_label:
                continue
            
            label.append(self.rel_label2ids[label_word])


            # text1.append(x["Arg1_RawText"])
            # text2.append(x["Arg2_RawText"])
            # mask_text.append("{} [MASK] {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))
            mask_text.append("{} <mask> {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))

            # connective.append(x["Conn1"])  

            if x["Conn1"] not in self.ans_word:
                subtype_index = self.subtype_label.index(x["subtype_label"])
                word = self.subtype_label_word[subtype_index]
            else:
                word = x["Conn1"]
            list0 = [0]*len(self.ans_word)
            list0[self.ans_word.index(word)] = 1
            label_conn.append(list0)
            # text.append("{} <Conn-start> {} <Conn-end> {}".format(x["Arg1_RawText"],label_word.split(".")[0],x["Arg2_RawText"]))
            
            if self.template_version == 777:
                if i%10==0:
                    text.append("{} <mask> {} Answer: {}".format(x["Arg1_RawText"],x["Arg2_RawText"],x["label"].split(".")[0].lower()))
                else:
                    text.append("{} <mask> {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))
            elif self.template_version == 888:
                text.append("{} <mask> {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))
            elif self.template_version == 999:
                if i%10 in [1,4,7,0]:
                    text.append("{} <mask> {} Answer: {}".format(x["Arg1_RawText"],x["Arg2_RawText"],x["label"].split(".")[0].lower()))
                else:
                    text.append("{} <mask> {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))
            elif self.template_version == 1111:
                if i%10 in [0,2,3,5,6,8,9]:
                    text.append("{} <mask> {} Answer: {}".format(x["Arg1_RawText"],x["Arg2_RawText"],x["label"].split(".")[0].lower()))
                else:
                    text.append("{} <mask> {}".format(x["Arg1_RawText"],x["Arg2_RawText"]))
            elif self.template_version == 2222:
                text.append("{} <mask> {} Answer: {}".format(x["Arg1_RawText"],x["Arg2_RawText"],x["label"].split(".")[0].lower()))
            
            else:
                text.append("The label is {} . {} <mask> {}".format(x["label"].split(".")[0].lower(),x["Arg1_RawText"],x["Arg2_RawText"]))


        return text,mask_text,label,label_conn


    def create_dataloader(self,data,batch_size,shuffle=False,max_length=250):
        # origin max_length=250
        tokenizer = self.tokenizer

        # data = data[:100]
       
        text,mask_text,label,label_conn= self.process_data_from_csv(data)
        
        df = pd.DataFrame(text)
        df.columns = ["text"]
        df.to_csv("test.csv",sep="\t",index=0)
        
        max_length = min(max_length,max([len(tokenizer.encode(s)) for s in mask_text]))

        # max_length = min(max_length,max([len(tokenizer.encode(s1,s2)) for s1,s2 in zip(text1,text2)]))
        print("max sentence length: ", max_length)

        max_conn_length = min(max_length,max([len(tokenizer.encode(c)) for c in text]))
       
        # max_length = 512

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            mask_text,
            max_length=max_length,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True,
        
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            )

        teacher_model_inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            max_length=max_conn_length,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True,
        
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
            )
       
        # 4. 将得到的句子编码和BIO转为dataloader，供模型使用
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(inputs["input_ids"]),          # 句子字符id
            # torch.LongTensor(inputs["token_type_ids"]),     # 区分两句话，此任务中全为0，表示输入只有一句话
            torch.LongTensor(inputs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0

            torch.LongTensor(teacher_model_inputs["input_ids"]),          # 句子字符id
            # torch.LongTensor(teacher_model_inputs["token_type_ids"]),     # 区分两句话，此任务中全为0，表示输入只有一句话
            torch.LongTensor(teacher_model_inputs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0

            # torch.LongTensor(inputs["offset_mapping"]),  
            torch.LongTensor(label),
            torch.tensor(label_conn,dtype=torch.float),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=0,
        )
        return dataloader

