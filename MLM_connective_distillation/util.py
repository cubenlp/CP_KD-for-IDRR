#!/usr/bin/env python
import random
import re
from typing import List
import argparse
import pandas as pd

import torch
import numpy as np
from datetime import datetime
import string



def str2bool(v):
    '''
    将字符转化为bool类型
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def split_sentence(content, max_sen_length=512):
    '''
    将正文按句切分，返回句子列表
    '''
    re_sentence_sp = re.compile('([﹒﹔；﹖﹗．。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    s = content
    slist = []
    for i in re_sentence_sp.split(s):  # 将句子按照正则表达式切分
        if re_sentence_sp.match(i) and slist:  # 如果是标点符号，则添加到上一句末尾
            slist[-1] += i
        elif i.strip():  # 不是标点符号，也不是空字符串，则将句子添加到句子列表
            while len(i) >= max_sen_length - 2:  # 按句切分后句子长度大于BERT最大长度 (-2是因为句子开头和结尾要添加CLS和SEP)
                sub_i, i = i[:max_sen_length - 2], i[max_sen_length - 2:]
                slist.append(sub_i)
            slist.append(i)
    return slist


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

