# DePro

*Decorrelate Irrelevant, Purify Relevant: Overcome Textual Spurious Correlations from a Feature Perspective*


<img width="520" alt="image" src="https://user-images.githubusercontent.com/111481934/189941995-a56461fa-a4ad-4905-81e2-305b84dc950d.png">


This is the open-source code repository for COLING 2022 paper "DePro: Decorrelate Irrelevant, Purify Relevant: Overcome Textual Spurious Correlations from a Feature Perspective"

### Notes

The code implementation of this paper is mainly referenced in the following two papers.

1. [CVPR21] [Deep Stable Learning for Out-Of-Distribution Generalization](https://arxiv.org/abs/2104.07876)

   Zhang, Xingxuan and Cui, Peng and Xu, Renzhe and Zhou, Linjun and He, Yue and Shen, Zheyan

   Code: https://github.com/xxgege/StableNet

2. [ICLR 2021] [InfoBERT: Improving Robustness of Language Models from An Information Theoretic Perspective](https://openreview.net/forum?id=hpH98mK5Puk)
   Wang, Boxin and Wang, Shuohang and Cheng, Yu and Gan, Zhe and Jia, Ruoxi and Li, Bo and Liu, Jingjing

   Code: https://github.com/AI-secure/InfoBERT

### Getting Started

#### Requirements

```
pytorch==1.7.0 
cudatoolkit=11.1
datasets==1.18.3
transformers==4.16.2
tensorboard==2.8.0
```

#### Code Structure

```
Depro/
|--dataset/                # dataset files
  |--MNLI/
    |--train.csv
    |--id_dev_mismatched.csv
    |--id_test_mismatched.csv
    |-- ...
  |--MNLI_hard/
  |--HANS/
  |-- ...
|--models/                 
  |--model_slabt.py        # model structure implementation
|--ops/                    
  |--config.py             # configuration files such as hyper-parameters
|--training/
  |--reweighting.py        # reweight module of feature decorrelation
  |--schedule.py           # learning schedule
  |--train.py              # training phase
  |--validate.py           # validate phase
|--utilis/
  |--datasets.py           # load datasets
  |--matrix.py             # matrix of feature decorrelation
  |--meters.py             # exp metric
  |--saving.py             # saving models
  |--info_regularizer.py   # infoB regularizer of feature purification
|--loss_reweighting.py     # RFF module of feature decorrelation
|--main_feaDe.py           # running ablation experiment of feature decorrelation
|--main_feaPu.py           # running ablation experiment of feature purification
|--main.py                 # running main experiment
```



### Usage

#### Feature Decorrelation

```bash
python main_featDe.py
```

#### Feature Purification

Soon

### Reference

```
@inproceedings{zhang2021deep,
  title={Deep Stable Learning for Out-Of-Distribution Generalization},
  author={Zhang, Xingxuan and Cui, Peng and Xu, Renzhe and Zhou, Linjun and He, Yue and Shen, Zheyan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5372--5382},
  year={2021}
}

@inproceedings{
wang2021infobert,
title={InfoBERT: Improving Robustness of Language Models from An Information Theoretic Perspective},
author={Wang, Boxin and Wang, Shuohang and Cheng, Yu and Gan, Zhe and Jia, Ruoxi and Li, Bo and Liu, Jingjing},
booktitle={International Conference on Learning Representations},
year={2021}}
```

### Citation

```
@inproceedings
```
