# coding=utf-8
from cProfile import label
from tkinter import Label
from typing import DefaultDict, Sequence
from torch._C import device, set_flush_denormal
from EE_data_process import EEProcessor

import pytorch_lightning as pl
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import csv
from transformers import (
    BertTokenizerFast,
    BertTokenizer,
    BertForTokenClassification,
    BertModel,
    DebertaModel
)
from transformers import DebertaForMaskedLM

from transformers import AutoTokenizer, AutoModelForMaskedLM,BertModel,BertForSequenceClassification,RobertaForSequenceClassification
from transformers import DebertaTokenizerFast, DebertaForTokenClassification, DebertaForSequenceClassification
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM,RobertaModel
from sklearn.metrics import classification_report


from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn import CrossEntropyLoss

import re
import json
import pandas as pd

class EETeacherModel(pl.LightningModule):
    def __init__(self, config,):
        # 1. Init parameters
        super(EETeacherModel, self).__init__()
        
        self.config=config

        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.teacher_pretrained_path)


        # self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        special_tokens_dict = { 'additional_special_tokens': ["<Conn-start>", "<Conn-end>","[SEP1]","<sense_sep>"] }  # 在词典中增加特殊字符

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.processor = EEProcessor(config, self.tokenizer)

        self.labels = len(self.processor.label2ids)
        # self.model = DebertaForSequenceClassification.from_pretrained(
        #     config.pretrained_path, num_labels=self.labels)
        # self.model = DebertaForMaskedLM.from_pretrained(
        #     config.pretrained_path, num_labels=self.labels,output_hidden_states=True)
        self.model = RobertaForMaskedLM.from_pretrained(
            config.teacher_pretrained_path, num_labels=self.labels,output_hidden_states=False)
       
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size,self.labels)

        self.loss_fct = CrossEntropyLoss(label_smoothing=0.05,)
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr = config.lr
        self.alpha = config.alpha
        self.temperature_rate = config.temperature_rate
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.kl_weight = config.kl_weight


        self.len_comp = len(self.processor.Comp)
        self.len_cont = len(self.processor.Cont)
        self.len_expa = len(self.processor.Expa)
        self.len_temp = len(self.processor.Temp)

        # self.teacher_model = EETeacherModel.load_from_checkpoint(config.teacher_checkpoint_path, config=config)
        # self.teacher_model.eval()


    def prepare_data(self):
        train_data = self.processor.train_data
        dev_data = self.processor.dev_data



        # dev_data = self.processor.train_data
        if self.config.train_num>0:
            train_data=train_data[:self.config.train_num]
        if self.config.dev_num>0:
            dev_data=dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = self.processor.create_dataloader(
            dev_data, batch_size=self.batch_size, shuffle=False)

    
    def forward(self, input_ids,attention_mask, training_flag=False):
       
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_mask_indices = [0]*len(input_ids)
        for idx,indice in (input_ids==50264).nonzero().tolist():
            token_mask_indices[idx] = indice

        # token_mask_indices = [l[1] for l in (input_ids==50264).nonzero().tolist()]
        mask_arg = output[0]
        # pooler_output = output[1][-1][:,0,:]

        out_vocab = torch.zeros(len(mask_arg), self.model.vocab_size).cuda()
        for i in range(len(mask_arg)):
            out_vocab[i] = mask_arg[i][token_mask_indices[i]]

        out_ans = out_vocab[:, self.processor.Token_id] # Tensor.cuda()
 
        return out_ans
        
      

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask,label_conn = batch

        out_ans = self.forward(input_ids,attention_mask)
        mask_loss = self.loss_fct(out_ans, label_conn)
        loss = mask_loss

        self.log('train_loss', loss.item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask,label_conn = batch
        out_ans= self.forward(input_ids,attention_mask)
        mask_loss = self.loss_fct(out_ans, label_conn)
        loss = mask_loss

        # loss = self.loss_fct(logits,label)
        # pred = logits.argmax(dim=-1)
        pred = out_ans.argmax(dim=-1)


        gold = label_conn.argmax(dim=-1)
        pre = torch.tensor(pred).cuda()

        return loss.cpu(),gold.cpu(),pre.cpu()

    def validation_epoch_end(self, outputs):
        val_loss,gold,pre = zip(*outputs)

        val_loss = torch.stack(val_loss).mean()
        gold = torch.cat(gold)
        pre = torch.cat(pre)

        # print("")
        def to_label(label):
            # if label==0 :return "O"
            # return "B-"+self.processor.ids2conn[con.item()]
            return self.processor.id2labels[label.item()]

        # true_seqs = [self.processor.id2labels[int(g)] for g,con in zip(gold,cons)]
        # pred_seqs = [self.processor.id2labels[int(g)] for g in zip(pre,cons)]

        true_seqs = [to_label(g) for g in gold]
        pred_seqs = [to_label(g) for g in pre]


        print("true_seqs",len(true_seqs),true_seqs[:5])
        print("pred_seqs",len(pred_seqs),pred_seqs[:5])

        print('\n')
        # prec, rec, f1 = RE_evaluate(true_seqs, pred_seqs)

        print(classification_report(
                y_pred=pre, y_true=gold, digits=4, 
                labels = [x for x in range(0, self.labels)],
                target_names = list(self.processor.label2ids.keys())
                ))
        res = classification_report(
                y_pred=pre, y_true=gold, digits=4, 
                labels = [x for x in range(0, self.labels)],
                target_names = list(self.processor.label2ids.keys()),output_dict=True
                )
        if "accuracy" in res:
            acc = res["accuracy"]
        else:
            acc = res["micro avg"]["f1-score"]
        f1 = res["macro avg"]

        self.log('val_loss', val_loss)

        self.log('val_acc', torch.tensor(acc*100))
        self.log('val_f1', torch.tensor(f1['f1-score']*100))

    def configure_optimizers(self):
        # if self.use_crf:
        #     crf_params_ids = list(map(id, self.crf.parameters()))
        #     base_params = filter(lambda p: id(p) not in crf_params_ids, [
        #                          p for p in self.parameters() if p.requires_grad])

        #     arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]
        # else:
        #     # label_embed_and_attention_params = list(map(id, self.label_embedding.parameters())) + list(map(id, self.self_attention.parameters()))
        #     # arg_list = [{'params': list(self.label_embedding.parameters()) + list(self.self_attention.parameters()), 'lr': self.lr}]
        #     arg_list = [p for p in self.parameters() if p.requires_grad]
        arg_list = [p for key, p in self.named_parameters() if p.requires_grad and not key.startswith("teacher_model")]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        # return self.train_loader
        return self.processor.create_dataloader(
            self.processor.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # return self.valid_loader
        return self.processor.create_dataloader(
            self.processor.dev_data, batch_size=self.batch_size, shuffle=False)


class EEModel(pl.LightningModule):
    def __init__(self, config,):
        # 1. Init parameters
        super(EEModel, self).__init__()
        
        self.config=config

        # self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        # self.tokenizer = DebertaTokenizerFast.from_pretrained(config.pretrained_path)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.pretrained_path)


        # self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        special_tokens_dict = { 'additional_special_tokens': ["<Conn-start>", "<Conn-end>","[SEP1]","<sense_sep>"] }  # 在词典中增加特殊字符

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.processor = EEProcessor(config, self.tokenizer)

        self.labels = len(self.processor.label2ids)
        # self.model = DebertaForSequenceClassification.from_pretrained(
        #     config.pretrained_path, num_labels=self.labels)
        # self.model = DebertaForMaskedLM.from_pretrained(
        #     config.pretrained_path, num_labels=self.labels,output_hidden_states=True)
        self.model = RobertaForMaskedLM.from_pretrained(
            config.pretrained_path, num_labels=self.labels,output_hidden_states=True)
       
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # print(11111,self.model.config.hidden_size)
        self.layer_norm = torch.nn.LayerNorm(self.model.config.hidden_size)
        self.classifier = nn.Linear(self.model.config.hidden_size,self.labels)

        self.loss_fct = CrossEntropyLoss(label_smoothing=0.05,)
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr = config.lr
        self.alpha = config.alpha
        self.temperature_rate = config.temperature_rate
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.kl_weight = config.kl_weight


        self.len_comp = len(self.processor.Comp)
        self.len_cont = len(self.processor.Cont)
        self.len_expa = len(self.processor.Expa)
        self.len_temp = len(self.processor.Temp)

        self.teacher_model = EETeacherModel.load_from_checkpoint(config.teacher_checkpoint_path, config=config)
        # self.teacher_model.eval()


    def prepare_data(self):
        train_data = self.processor.train_data
        dev_data = self.processor.dev_data



        # dev_data = self.processor.train_data
        if self.config.train_num>0:
            train_data=train_data[:self.config.train_num]
        if self.config.dev_num>0:
            dev_data=dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = self.processor.create_dataloader(
            dev_data, batch_size=self.batch_size, shuffle=False)

    
    def forward(self, input_ids,attention_mask, training_flag=False):
       
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_mask_indices = [0]*len(input_ids)
        for idx,indice in (input_ids==50264).nonzero().tolist():
            token_mask_indices[idx] = indice

        # token_mask_indices = [l[1] for l in (input_ids==50264).nonzero().tolist()]
        mask_arg = output[0]

        out_vocab = torch.zeros(len(mask_arg), self.model.vocab_size).cuda()
        for i in range(len(mask_arg)):
            out_vocab[i] = mask_arg[i][token_mask_indices[i]]

        out_ans = out_vocab[:, self.processor.Token_id] # Tensor.cuda()

        pred_word = torch.argmax(out_ans, dim=1).tolist() # list

    
        pred = torch.IntTensor(len(mask_arg), len(self.processor.rel_label2ids)).cuda()

        for tid, idx in enumerate(pred_word, 0):
            if self.processor.data_type=="pdtb2_top":
                if idx <= (self.len_comp - 1):
                    pred[tid] = torch.IntTensor([1, 0, 0, 0])
                elif  (self.len_comp - 1) < idx <= (self.len_comp + self.len_cont - 1):
                    pred[tid] = torch.IntTensor([0, 1, 0, 0])
                elif  (self.len_comp + self.len_cont - 1) < idx <= (self.len_comp + self.len_cont + self.len_expa - 1):
                    pred[tid] = torch.IntTensor([0, 0, 1, 0])
                elif  (self.len_comp + self.len_cont + self.len_expa - 1) < idx <= (self.len_comp + self.len_cont + self.len_expa + self.len_temp - 1):
                    pred[tid] = torch.IntTensor([0, 0, 0, 1])
            elif self.processor.data_type=="pdtb3_top":
                tmp = [0]*len(self.processor.rel_label2ids)
                tmp[self.processor.rel_label2ids[self.processor.ans2pdtb3toplabel[idx]]] = 1
                pred[tid] = torch.IntTensor(tmp)
            elif self.processor.data_type=="pdtb2_second":
                tmp = [0]*len(self.processor.rel_label2ids)
                tmp[self.processor.rel_label2ids[self.processor.ans2secondlabel[idx]]] = 1
                pred[tid] = torch.IntTensor(tmp)
            elif self.processor.data_type=="pdtb3_second":
                tmp = [0]*len(self.processor.rel_label2ids)
                tmp[self.processor.rel_label2ids[self.processor.ans2pdtb3secondlabel[idx]]] = 1
                pred[tid] = torch.IntTensor(tmp)
            else:
                tmp = [0]*len(self.processor.rel_label2ids)
                tmp[self.processor.rel_label2ids[self.processor.ans2conll16label[idx]]] = 1
                pred[tid] = torch.IntTensor(tmp)
       
      
        return pred,out_ans
        
      

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask,teacher_input_ids,teacher_attention_mask,label, label_conn = batch
      

        '''knowledge distillation start'''
        pred,out_ans = self.forward(input_ids,attention_mask)
        mask_loss = self.loss_fct(out_ans, label_conn)

        teacher_logits = self.teacher_model(teacher_input_ids,teacher_attention_mask)
        ts_loss = self.kld(F.log_softmax(teacher_logits/self.temperature_rate, dim=-1), F.softmax(out_ans/self.temperature_rate, dim=-1)).sum(-1)


        loss = self.alpha*mask_loss+(1-self.alpha)*(self.temperature_rate**2)*ts_loss


        '''knowledge distillation end'''


        self.log('train_loss', loss.item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask,teacher_input_ids,teacher_attention_mask,label, label_conn = batch
        pred,out_ans = self.forward(input_ids,attention_mask)
        mask_loss = self.loss_fct(out_ans, label_conn)
        loss = mask_loss

        # loss = self.loss_fct(logits,label)
        # pred = logits.argmax(dim=-1)
        pred = pred.argmax(dim=-1)


        gold = label
        pre = torch.tensor(pred).cuda()

        return loss.cpu(),gold.cpu(),pre.cpu()

    def validation_epoch_end(self, outputs):
        val_loss,gold,pre = zip(*outputs)

        val_loss = torch.stack(val_loss).mean()
        gold = torch.cat(gold)
        pre = torch.cat(pre)

        # print("")
        def to_label(label):
            # if label==0 :return "O"
            # return "B-"+self.processor.ids2conn[con.item()]
            return self.processor.rel_id2labels[label.item()]

        # true_seqs = [self.processor.id2labels[int(g)] for g,con in zip(gold,cons)]
        # pred_seqs = [self.processor.id2labels[int(g)] for g in zip(pre,cons)]

        true_seqs = [to_label(g) for g in gold]
        pred_seqs = [to_label(g) for g in pre]


        print("true_seqs",len(true_seqs),true_seqs[:5])
        print("pred_seqs",len(pred_seqs),pred_seqs[:5])

        print('\n')
        # prec, rec, f1 = RE_evaluate(true_seqs, pred_seqs)

        print(classification_report(
                y_pred=pre, y_true=gold, digits=4, 
                labels = [x for x in range(0, len(self.processor.rel_label2ids))],
                target_names = list(self.processor.rel_label2ids.keys())
                ))
        res = classification_report(
                y_pred=pre, y_true=gold, digits=4, 
                labels = [x for x in range(0, len(self.processor.rel_label2ids))],
                target_names = list(self.processor.rel_label2ids.keys()),output_dict=True
                )
        if "accuracy" in res:
            acc = res["accuracy"]
        else:
            acc = res["micro avg"]["f1-score"]
        f1 = res["macro avg"]

        self.log('val_loss', val_loss)

        self.log('val_acc', torch.tensor(acc*100))
        self.log('val_f1', torch.tensor(f1['f1-score']*100))

    def configure_optimizers(self):
        arg_list = [p for key, p in self.named_parameters() if p.requires_grad and not key.startswith("teacher_model")]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        return self.train_loader
        # return self.processor.create_dataloader(
        #     self.processor.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return self.valid_loader
        # return self.processor.create_dataloader(
        #     self.processor.dev_data, batch_size=self.batch_size, shuffle=False)

class EEPredictor:
    def __init__(self, checkpoint_path, config, test_data=[]):
        self.model = EEModel.load_from_checkpoint(checkpoint_path, config=config)
        self.config = config
        if len(test_data):
            self.test_data = test_data
        else:
            self.test_data = self.model.processor.test_data

        
        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
            self.test_data,batch_size=config.batch_size,shuffle=False)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def predict(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        print(len(self.test_data),len(self.dataloader))
        pred_list = []
        label_list = []
        logits_list = []

        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            # input_ids, attention_mask, conn_input_ids, conn_attention_mask, label = batch
            # logits = self.model(input_ids,attention_mask)

            input_ids, attention_mask,_,_,label, _ = batch
            pred,out_ans = self.model(input_ids,attention_mask)
            logits_list.extend(pred)
           
            # loss = self.loss_fct(logits,label)
            # pred = logits.argmax(dim=-1)
            preds = pred.argmax(dim=-1)
            pred_list.extend(preds.cpu())
            label_list.extend(label.cpu())

        if self.config.test_path == "data/conll2016/blind_test.tsv":
            self.config.data_type+="blind"
        torch.save(torch.stack(logits_list),"pth/{}.pth".format(self.config.data_type))
        torch.save(torch.stack(label_list),"pth/truth_{}.pth".format(self.config.data_type))


        # print(torch.stack(logits_list))


        print(classification_report(
                        y_pred=pred_list, y_true=label_list, digits=4, 
                        labels = [x for x in range(0, len(self.model.processor.rel_label2ids.keys()))],
                        target_names = list(self.model.processor.rel_label2ids.keys())
                        ))
        # prec, rec, f1 = RE_evaluate(pred_list, label_list)
        return pred_list







