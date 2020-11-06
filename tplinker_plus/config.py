import string
import random

common = {
    "exp_name": "duie2", # ace05_lu
    "rel2id": "rel2id.json",
    "ent2id": "ent2id.json",
    "device_num": 1,
#     "encoder": "BiLSTM",
    "encoder": "BERT", 
    "hyper_parameters": {
        "shaking_type": "cln_plus",
        "inner_enc_type": "lstm",
        # match_pattern: only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span, event_extraction
        "match_pattern": "whole_text", 
    },
}
common["run_name"] = "{}+{}+{}".format("TP2", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
train_config = {
    "train_data": "train_data.json",
    "valid_data": "valid_data.json",
    "rel2id": "rel2id.json",
    "logger": "wandb", # if wandb, comment the following four lines
   
#     # if logger is set as default, uncomment the following four lines and comment the line above
#     "logger": "default", 
#     "run_id": run_id,
#     "log_path": "./default_log_dir/default.log",
#     "path_to_save_model": "./default_log_dir/{}".format(run_id),

    # when to save the model state dict
    "f1_2_save": 0,
    # whether train_config from scratch
    "fr_scratch": True,
    # write down notes here if you want, it will be logged
    "note": "start from scratch",
    # if not fr scratch, set a model_state_dict
    "model_state_dict_path": "", # valid only if "fr_scratch" is False
    "hyper_parameters": {
        "batch_size": 32,
        "epochs": 100,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 128,
        "sliding_len": 20,
        "scheduler": "CAWR", # Step
        "ghm": False, # set True if you want to use GHM to adjust the weights of gradients, this will speed up the training process and might improve the results. (Note that ghm in current version is unstable now, may hurt the results)
        "tok_pair_sample_rate": 1, # (0, 1] How many percent of token paris you want to sample for training, this would slow down the training if set to less than 1. It is only helpful when your GPU memory is not enought for the training.
    },
}

eval_config = {
    "model_state_dict_dir": "./wandb", # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "run_ids": ["1a70p109", ],
    "last_k_model": 1,
    "test_data": "*test*.json", # "*test*.json"
    
    # results
    "save_res": False,
    "save_res_dir": "../results",
    
    # score: set true only if test set is tagged
    "score": True,
    
    "hyper_parameters": {
        "batch_size": 32,
        "force_split": False,
        "max_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "../data4bert",
    "bert_path": "../../pretrained_models/chinese-bert-wwm-ext-hit", # bert-base-cased， chinese-bert-wwm-ext-hit
    "hyper_parameters": {
        "lr": 5e-5,
    },
}
bilstm_config = {
    "data_home": "../data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "../../pretrained_emb/glove_300_nyt.emb",
    "hyper_parameters": {
         "lr": 1e-3,
         "enc_hidden_size": 300,
         "dec_hidden_size": 600,
         "emb_dropout": 0.1,
         "rnn_dropout": 0.1,
         "word_embedding_dim": 300,
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------
if common["encoder"] == "BERT":
    hyper_params = {**common["hyper_parameters"], **bert_config["hyper_parameters"]}
    common = {**common, **bert_config}
    common["hyper_parameters"] = hyper_params
elif common["encoder"] == "BiLSTM":
    hyper_params = {**common["hyper_parameters"], **bilstm_config["hyper_parameters"]}
    common = {**common, **bilstm_config}
    common["hyper_parameters"] = hyper_params
    
hyper_params = {**common["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}
    
hyper_params = {**common["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common}
eval_config["hyper_parameters"] = hyper_params