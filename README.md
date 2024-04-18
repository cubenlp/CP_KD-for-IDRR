# CP_KD-for-IDRR

## How to Train Teacher model
```bash
CUDA_VISIBLE_DEVICES=0 python MLM/EE_main.py --data_type pdtb2_top --train_path data/pdtb2/train.tsv --dev_path data/pdtb2/dev.tsv
```

## How to Train Student Model 

```bash
CUDA_VISIBLE_DEVICES=0 python MLM_connective_distillation/EE_main.py --template_version 777 --alpha 0.4 --temperature_rate 1 --version KLDLoss_roberta_base_masked_model+ls=0.05+alpha=0.5+tr=1_maxlen=250_bsz=16_agb=2_teacher+temp_true_test_epoch=2_20221026_lr=1e-5_template777_trueS --lr 1e-5 --batch_size 16 --data_type pdtb2_top --pretrained_path roberta-base --teacher_pretrained_path roberta-base --teacher_checkpoint_path mlm_test_all_weights/basepdtb2_top_template_777_epoch=1.ckpt --train_path data/pdtb2/train.tsv --dev_path data/pdtb2/dev.tsv
```

## ACL 2023 paper
Connective Prediction for Implicit Discourse Relation Recognition via Knowledge Distillation (https://aclanthology.org/2023.acl-long.325/)

## Citation
```
@inproceedings{wu-etal-2023-connective,
    title = "Connective Prediction for Implicit Discourse Relation Recognition via Knowledge Distillation",
    author = "Wu, Hongyi  and
      Zhou, Hao  and
      Lan, Man  and
      Wu, Yuanbin  and
      Zhang, Yadong",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.325",
    doi = "10.18653/v1/2023.acl-long.325",
    pages = "5908--5923",
    abstract = "Implicit discourse relation recognition (IDRR) remains a challenging task in discourse analysis due to the absence of connectives. Most existing methods utilize one-hot labels as the sole optimization target, ignoring the internal association among connectives. Besides, these approaches spend lots of effort on template construction, negatively affecting the generalization capability. To address these problems,we propose a novel Connective Prediction via Knowledge Distillation (CP-KD) approach to instruct large-scale pre-trained language models (PLMs) mining the latent correlations between connectives and discourse relations, which is meaningful for IDRR. Experimental results on the PDTB 2.0/3.0 and CoNLL2016 datasets show that our method significantly outperforms the state-of-the-art models on coarse-grained and fine-grained discourse relations. Moreover, our approach can be transferred to explicit discourse relation recognition(EDRR) and achieve acceptable performance.",
}
```
