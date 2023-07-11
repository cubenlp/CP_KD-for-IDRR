# CP_KD-for-IDRR

## How to Train Teacher model
```bash
CUDA_VISIBLE_DEVICES=0 python MLM/EE_main.py --data_type pdtb2_top --train_path data/pdtb2/train.tsv --dev_path data/pdtb2/dev.tsv
```

## How to Train Student Model 

```bash
CUDA_VISIBLE_DEVICES=0 python MLM_connective_distillation/EE_main.py --template_version 777 --alpha 0.4 --temperature_rate 1 --version KLDLoss_roberta_base_masked_model+ls=0.05+alpha=0.5+tr=1_maxlen=250_bsz=16_agb=2_teacher+temp_true_test_epoch=2_20221026_lr=1e-5_template777_trueS --lr 1e-5 --batch_size 16 --data_type pdtb2_top --pretrained_path roberta-base --teacher_pretrained_path roberta-base --teacher_checkpoint_path mlm_test_all_weights/basepdtb2_top_template_777_epoch=1.ckpt --train_path data/pdtb2/train.tsv --dev_path data/pdtb2/dev.tsv
```
