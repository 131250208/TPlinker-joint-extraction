common = {
    "exp_name": "nyt_single",
    "rel2id": "rel2id.json",
    "device_num": 2,
#     "encoder": "BiLSTM",
    "encoder": "BERT", 
    "hyper_parameters": {
        "shaking_type": "cat",
        "dist_emb_size": -1,
        "ent_add_dist": False,
        "rel_add_dist": False,
        "match_pattern": "only_head_text", # only_head_text, whole_text, only_head_index, whole_span
    },
}
common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"])

train_config = {
    "train_data": "train_data.json",
    "valid_data": "valid_data.json",
    "rel2id": "rel2id.json",
    
    # for default logger
#     "logger": "wandb",
    "logger": "default", 
    "log_path": "./default.log",
    "path_to_save_model": "./model_state",

    # when to save the model state dict
    "f1_2_save": 0,
    # whether train_config from scratch
    "fr_scratch": True,
    # note 
    "note": "start from scratch",
    # if not fr scratch, set a model_state_dict
    "model_state_dict_path": "",
    "hyper_parameters": {
        "batch_size": 6,
        "epochs": 200,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 100,
        "sliding_len": 20,
        "loss_weight_recover_steps": 6000,
        "scheduler": "CAWR", # Step
    },
}

eval_config = {
    "model_state_dict_dir": "./wandb",
    "run_ids": ["1vgzwath", ],
    "last_k_model": 1,
    "test_data": "*test*.json", # "*test*.json"
    
    # results
    "save_res": False,
    "save_res_dir": "../results",
    
    # score: set true only if test set is tagged
    "score": True,
    
    "hyper_parameters": {
        "batch_size": 1,
        "force_split": False,
        "max_test_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "../data4bert",
    "bert_path": "../pretrained_model/bert-base-cased",
    "hyper_parameters": {
        "lr": 5e-5,
    },
}
bilstm_config = {
    "data_home": "../data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "../pretrained_word_emb/glove_300_webnlg.emb",
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
