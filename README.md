# TPLinker
This repository contains all the code of the official implementation for the paper: **TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.** \[[PDF](https://drive.google.com/file/d/1fCmbBi_4mzFmzLS-ovtEyRigiM7wWigM/view?usp=sharing)\] 
The paper has been accepted to appear at **COLING 2020**.

- [Model](#model)
- [Results](#results)
- [Usage](#usage)
  * [Prerequisites](#prerequisites)
  * [Data](#data)
    + [download data](#download-data)
    + [build data](#build-data)
  * [Pretrained Model and Word Embeddings](#pretrained-model-and-word-embeddings)
  * [Train](#train)
  * [Evaluation](#evaluation)

## Model
<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95205135-8bf08d80-0817-11eb-80bb-8f559f072c8c.png" alt="framework" width="768"/>
</p>

**The Framework of TPLinker. SH is short for subject head, OH is short for object head, ST is
short for subject tail, and OT is short for object tail.**

## Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95207775-d0c9f380-081a-11eb-95e9-976c58acab84.png" alt="data_statistics" width="768"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95207797-d6273e00-081a-11eb-8d6d-f00b3e20bef7.png" alt="main_res" width="768"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/7437595/95207811-d9bac500-081a-11eb-87b4-2d917cdc6e21.png" alt="res2" width="768"/>
</p>

## Usage
### Prerequisites
Our experiments are conducted on Python 3.6 and Pytorch 1.4. 
The main requirements are:
* tqdm
* glove-python-binary==0.2.0
* transformers==2.10.0
* wandb # for logging the results
* yaml

In the root directory, run
```bash
pip install -e .
```
### Data
#### download data
Get and preprocess NYT* and WebNLG* following [CasRel](https://github.com/weizhepei/CasRel/tree/master/data) (note: named NYT and WebNLG by CasRel).
Take NYT* as an example, rename train_triples.json and dev_triples.json to train_data.json and valid_data.json and move them to `ori_data/nyt_star`, put all test*.json under `ori_data/nyt_star/test_data`. The same process goes for WebNLG*.

Get raw NYT from [CopyRE](https://github.com/xiangrongzeng/copy_re),  rename raw_train.json and raw_valid.json to train_data.json and valid_data.json and move them to `ori_data/nyt`, rename raw_test.json to test_data.json and put it under `ori_data/nyt/test_data`.

Get WebNLG from [ETL-Span](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data), rename train.json and dev.json to train_data.json and valid_data.json and move them to `ori_data/webnlg`, rename test.json to test_data.json and put it under `ori_data/webnlg/test_data`.

#### build data
Build data by `preprocess/BuildData.ipynb`.
Set configuration in `preprocess/build_data_config.yaml`.
In the configuration file, set `exp_name` corresponding to the directory name, set `ori_data_format` corresponding to the source project name. 
e.g. To build NYT*, set `exp_name` to `nyt_star` and set `ori_data_format` to `casrel`. See `build_data_config.yaml` for more details.

### Pretrained Model and Word Embeddings
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `pretrained_model`. Pretrain word embeddings by `preprocess/Pretrain_Word_Embedding.ipynb` and put models under `pretrained_word_emb`.

### Train
Set configuration in `tplinker/config.py` as follows:
```python
common["exp_name"] = nyt_star # webnlg_star, nyt, webnlg
common["device_num"] = 0 # 1, 2, 3 ...
common["encoder"] = "BERT" # BiLSTM
train_config["hyper_parameters"]["batch_size"] = 24 # 6 for webnlg and webnlg_star
train_config["hyper_parameters"]["match_pattern"] = "only_head_text" # "only_head_text" for webnlg_star and nyt_star; "whole_text" for webnlg and nyt.

# if the encoder is set to BiLSTM
bilstm_config["pretrained_word_embedding_path"] = ""../pretrained_word_emb/glove_300_nyt.emb""

# Leave the rest as default
```

Start training
```
cd tplinker
python train.py
```
### Evaluation
Set configuration in `tplinker/config.py` as follows:
```python
eval_config["run_ids"] = ["46qer3r9", ] # run id is recorded in training log
eval_config["last_k_model"] = 1 # only use the last k models in these runs to output results
# Leave the rest as the same as the training
```
Start evaluation by running `tplinker/Evaluation.ipynb`
