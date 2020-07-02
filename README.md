# JointExtraction
## Data
### download data
Get and preprocess NYT* and WebNLG* following [CasRel](https://github.com/weizhepei/CasRel/tree/master/data) (note: named NYT and WebNLG by CasRel).
Take NYT* as an example, rename train_triples.json and dev_triples.json to train_data.json and valid_data.json and move them to `ori_data/nyt_star`, put all test*.json under `ori_data/nyt_star/test_data`. The same process goes for WebNLG*.

Get raw NYT from [CopyRE](https://github.com/xiangrongzeng/copy_re),  rename raw_train.json and raw_valid.json to train_data.json and valid_data.json and move them to `ori_data/nyt`, rename raw_test.json to test_data.json and put it under `ori_data/nyt/test_data`.

Get WebNLG from [ETL-Span](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data), rename train.json and dev.json to train_data.json and valid_data.json and move them to `ori_data/webnlg`, rename test.json to test_data.json and put it under `ori_data/webnlg/test_data`.

### build data
Build data by `preprocess/BuildData.ipynb`.
Set configuration in `preprocess/build_data_config.yaml`.
In the configuration file, set `exp_name` corresponding to the directory name, set `ori_data_format` corresponding to the source project name. 
e.g. To build NYT*, set `exp_name` to `nyt_star` and set `ori_data_format` to `casrel`. See `build_data_config.yaml` for more details.

## Pretrained Model and Word Embeddings
Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `pretrained_model`. Pretrain word embeddings by `preprocess/Pretrain_Word_Embedding.ipynb` and put models under `pretrained_word_emb`.

## Train
Set configuration in `tplinker/config.py` as follows:
```
common["exp_name"] = nyt_star # webnlg_star, nyt, webnlg
common["device_num"] = 0 # 1, 2, 3 ...
train_config["hyper_parameters"]["batch_size"] = 24 # 6 for webnlg and webnlg*
# Leave the rest as default
```

Start training
```
cd tplinker
python train.py
```
## Evaluation
Set configuration in `tplinker/config.py` as follows:
```
eval_config["run_ids"] = ["46qer3r9", ] # run id is recorded in training log
eval_config["last_k_model"] = 1 # only use the last k models in these runs to output results
# Leave the rest as the same as the training
```
Start evaluation by running `tplinker/Evaluation.ipynb`
