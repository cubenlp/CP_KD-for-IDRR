# coding=utf-8
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from EE_model import EEModel,EEPredictor
# from EE_model_arg import EEModel,EEPredictor

import util
import warnings
warnings.filterwarnings("ignore")


os.environ["TOKENIZERS_PARALLELISM"] = "True"

def gen_args():
    WORKING_DIR = "/home/hongyi/discourse/prompt_distillation_template3_noS"

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20221026, help="random seed")

    parser.add_argument("--is_train", type=util.str2bool, default=True, help="train the EE model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=32, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=10, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--crf_lr", type=float, default=0.1, help="crf learning rate (default: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")

    parser.add_argument("--use_bert", type=util.str2bool, default=True,
                        help="whether to use bert training or not (default: True)")
    parser.add_argument("--use_crf", type=util.str2bool, default=True,
                        help="whether to use crf layer training or not (default: True)")

    parser.add_argument("--use-bilstm", type=util.str2bool, default=True,
                        help="whether to use bilstm training or not (default: True)")
    parser.add_argument("--lstm_dropout_prob", type=float, default=0.5, help="lstm dropout probability")
    parser.add_argument("--max_arg", type=int, default=512, help="max arg length")
    parser.add_argument("--hidden_dim",type=int, default=128, help="hidden dimension")
    parser.add_argument("--num_perspectives",type=int, default=16, help="num_perspectives")
    parser.add_argument("--activation",type=str, default="relu", help="activation")
    parser.add_argument("--num_rels",type=int, default=15, help="num_rels")
    parser.add_argument("--num_filters",type=int, default=64, help="num_filters")

    parser.add_argument("--hidden_size", type=int, default=768, help="bilstm hidden size")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2, help="hidden dropout probability")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="kl weight")


    # 下面参数基本默认
    parser.add_argument("--data_type", type=str, default="pdtb2_top", help="dataset type")
    parser.add_argument("--version", type=str, default="roberta_masked_model", help="dataset type")
    
    
    parser.add_argument("--train_path", type=str, default="{}/data/pdtb2/train.tsv".format(WORKING_DIR),
                        help="train_path")

    parser.add_argument("--exp_train_path", type=str, default="{}/data/pdtb2/explicit_train.tsv".format(WORKING_DIR),
                        help="train_path")

    
    parser.add_argument("--dev_path", type=str, default="{}/data/pdtb2/dev.tsv".format(WORKING_DIR),
                        help="dev_path")
    parser.add_argument("--train_num", type=int, default=-1,help="train data number")
    parser.add_argument("--dev_num", type=int, default=-1,help="train data number")
    parser.add_argument("--test_path", type=str, default="{}/data/pdtb2/test.tsv".format(WORKING_DIR),
                        help="test_path")
    
    
    parser.add_argument("--ee_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ee_result_path")
    parser.add_argument("--ckpt_save_path", type=str,
                        default="{}/mlm_test_all_weights/".format(WORKING_DIR), help="ckpt_save_path")
    parser.add_argument("--resume_ckpt", type=str,
                        default=None, help="checkpoint file name for resume")
    # parser.add_argument("--pretrained_path", type=str,
    #                     default="microsoft/deberta-base", help="pretrained_path")
    # parser.add_argument("--teacher_checkpoint_path",type=str,
    #                     default="/home/hongyi/discourse/distillation/mlm_teacher_all_weights/basepdtb2_top_roberta_model+exp_onlytem_teacher+ls+rdrop+promptconn_epoch=44_val_acc=98.9_val_f1=98.9.ckpt",help="pretrained_teacher_model")
    parser.add_argument("--teacher_checkpoint_path",type=str,
                        default="/home/hongyi/discourse/distillation/mlm_teacher_all_weights/basepdtb2_top_roberta_model_teacher+ls+rdrop+promptconn_epoch=0_val_acc=98.9_val_f1=98.9.ckpt",help="pretrained_teacher_model")


    parser.add_argument("--alpha", type=float, default=0.5, help="knowledge distillation alpha")
    parser.add_argument("--temperature_rate", type=float, default=1, help="knowledge distillation temperature rate")


    parser.add_argument("--pretrained_path", type=str,
                        default="roberta-base", help="pretrained_path")
    # parser.add_argument("--pretrained_path", type=str,
    #                     default="/home/lawson/program/pretrain_bert/user_data/pretrain_model/checkpoint-21800", help="pretrained_path")

    parser.add_argument("--outfile_txt",  type=str, default="base", help="result output filename")
    parser.add_argument("--ckpt_name",  type=str, default="base", help="ckpt save name")
    parser.add_argument("--ner_save_path",type=str, default="weights", help="ner save path")
    parser.add_argument("--test_ckpt_name",  type=str, default="###_epoch=13_val_f1=70.7.ckpt", help="ckpt name for test")

    args = parser.parse_args()
    return args

def train(args):
    # ============= train 训练模型==============
    print("start train model ...")
    model = EEModel(args)

    # 设置保存模型的路径及参数
    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_save_path,                           # 模型保存路径
        filename=args.ckpt_name + args.data_type+"_"+args.version+"_{epoch}_{val_acc:.2f}_{val_f1:.2f}",   # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
        monitor='val_acc',                                      # 根据验证集上的准确率评估模型优劣
        mode='max',
        save_top_k=3,                                           # 保存得分最高的前三个模型
        verbose=True,
    )

    resume_checkpoint=None
    if args.resume_ckpt:
        resume_checkpoint=os.path.join(args.ckpt_save_path ,args.resume_ckpt)   # 加载已保存的模型继续训练

    # 设置训练器
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        resume_from_checkpoint = resume_checkpoint,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback],
        checkpoint_callback=True,
        gpus=-1,
        strategy="ddp" if len(os.environ["CUDA_VISIBLE_DEVICES"])>1 else None,
        num_nodes=1,

    )

    # 开始训练模型
    trainer.fit(model)

    # 只训练CRF的时候，保存最后的模型
    # if config.use_crf and config.first_train_crf == 1:
    #     trainer.save_checkpoint(os.path.join(config.ner_save_path, 'crf_%d.ckpt' % (config.max_epochs)))


if __name__ == '__main__':

    args = gen_args()

    print('--------config----------')
    print(args)
    print('--------config----------')

    util.set_random_seed(args.seed)


    if args.is_train == True:
       train(args)
    else:
        # ============= test 测试模型==============
        print("\n\nstart test model...")

        outfile_txt = args.outfile_txt   
        # 开始测试，将结果保存至输出文件
        checkpoint_path = args.test_ckpt_name
        predictor = EEPredictor(checkpoint_path, args)
        pred_list = predictor.predict()
        # predictor.generate_result(outfile_txt)
        # print('\n', 'outfile_txt name:', outfile_txt)