#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from tqdm import tqdm
import re
from IPython.core.debugger import set_trace
from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
import logging
from common.utils import Preprocessor, DefaultLogger
from tplinker import (HandshakingTaggingScheme,
                      DataMaker4Bert, 
                      DataMaker4BiLSTM, 
                      TPLinkerBert, 
                      TPLinkerBiLSTM,
                      MetricsCalculator)
import wandb
import config
from glove import Glove
import numpy as np


# In[ ]:


# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)


# In[ ]:


config = config.train_config
hyper_parameters = config["hyper_parameters"]


# In[ ]:

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# for reproductivity
torch.manual_seed(hyper_parameters["seed"]) # pytorch random seed
torch.backends.cudnn.deterministic = True


# In[ ]:


data_home = config["data_home"]
experiment_name = config["exp_name"]    
train_data_path = os.path.join(data_home, experiment_name, config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name, config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])


# In[ ]:


if config["logger"] == "wandb":
    # init wandb
    wandb.init(project = experiment_name, 
               name = config["run_name"],
               config = hyper_parameters # Initialize config
              )

    wandb.config.note = config["note"]          

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    logger = DefaultLogger(config["log_path"], experiment_name, config["run_name"], config["run_id"], hyper_parameters)
    model_state_dict_dir = config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)


# # Load Data

# In[ ]:


train_data = json.load(open(train_data_path, "r", encoding = "utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding = "utf-8"))


# # Split

# In[ ]:


# @specific
if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping = True, add_special_tokens = False)["offset_mapping"]
elif config["encoder"] in {"BiLSTM", }:
    tokenize = lambda text: text.split(" ")
    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1 # +1: whitespace
        return tok2char_span


# In[ ]:


preprocessor = Preprocessor(tokenize_func = tokenize, 
                            get_tok2char_span_map_func = get_tok2char_span_map)


# In[ ]:


# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data 
    
for sample in all_data:
    tokens = tokenize(sample["text"])
    max_tok_num = max(max_tok_num, len(tokens))
max_tok_num


# In[ ]:


if max_tok_num > hyper_parameters["max_seq_len"]:
    train_data = preprocessor.split_into_short_samples(train_data, 
                                                          hyper_parameters["max_seq_len"], 
                                                          sliding_len = hyper_parameters["sliding_len"], 
                                                          encoder = config["encoder"]
                                                         )
    valid_data = preprocessor.split_into_short_samples(valid_data, 
                                                          hyper_parameters["max_seq_len"], 
                                                          sliding_len = hyper_parameters["sliding_len"], 
                                                          encoder = config["encoder"]
                                                         )


# In[ ]:


print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))


# # Tagger (Decoder)

# In[ ]:


max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))
handshaking_tagger = HandshakingTaggingScheme(rel2id = rel2id, max_seq_len = max_seq_len)


# # Dataset

# In[ ]:


if config["encoder"] == "BERT":
    tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"], add_special_tokens = False, do_lower_case = False)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
    
elif config["encoder"] in {"BiLSTM", }:
    token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
    token2idx = json.load(open(token2idx_path, "r", encoding = "utf-8"))
    idx2token = {idx:tok for tok, idx in token2idx.items()}
    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx['<UNK>'])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids
    data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)


# In[ ]:


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


# In[ ]:


indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)


# In[ ]:


train_dataloader = DataLoader(MyDataset(indexed_train_data), 
                                  batch_size = hyper_parameters["batch_size"], 
                                  shuffle = True, 
                                  num_workers = 6,
                                  drop_last = False,
                                  collate_fn = data_maker.generate_batch,
                                 )
valid_dataloader = DataLoader(MyDataset(indexed_valid_data), 
                          batch_size = hyper_parameters["batch_size"], 
                          shuffle = True, 
                          num_workers = 6,
                          drop_last = False,
                          collate_fn = data_maker.generate_batch,
                         )


# In[ ]:


# # have a look at dataloader
# train_data_iter = iter(train_dataloader)
# batch_data = next(train_data_iter)
# text_id_list, text_list, batch_input_ids, \
# batch_attention_mask, batch_token_type_ids, \
# offset_map_list, batch_ent_shaking_tag, \
# batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_data

# print(text_list[0])
# print()
# print(tokenizer.decode(batch_input_ids[0].tolist()))
# print(batch_input_ids.size())
# print(batch_attention_mask.size())
# print(batch_token_type_ids.size())
# print(len(offset_map_list))
# print(batch_ent_shaking_tag.size())
# print(batch_head_rel_shaking_tag.size())
# print(batch_tail_rel_shaking_tag.size())


# # Model

# In[ ]:


if config["encoder"] == "BERT":
    encoder = AutoModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
    rel_extractor = TPLinkerBert(encoder, 
                                 len(rel2id), 
                                 hyper_parameters["shaking_type"],
                                 hyper_parameters["inner_enc_type"],
                                 hyper_parameters["dist_emb_size"],
                                 hyper_parameters["ent_add_dist"],
                                 hyper_parameters["rel_add_dist"],
                                )
    
elif config["encoder"] in {"BiLSTM", }:
    glove = Glove()
    glove = glove.load(config["pretrained_word_embedding_path"])
    
    # prepare embedding matrix
    word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
    count_in = 0

    # 在预训练词向量中的用该预训练向量
    # 不在预训练集里的用随机向量
    for ind, tok in tqdm(idx2token.items(), desc="Embedding matrix initializing..."):
        if tok in glove.dictionary:
            count_in += 1
            word_embedding_init_matrix[ind] = glove.word_vectors[glove.dictionary[tok]]

    print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(idx2token))) # 命中预训练词向量的比例
    word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)
    
    fake_inputs = torch.zeros([hyper_parameters["batch_size"], max_seq_len, hyper_parameters["dec_hidden_size"]]).to(device)
    rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix, 
                                   hyper_parameters["emb_dropout"], 
                                   hyper_parameters["enc_hidden_size"], 
                                   hyper_parameters["dec_hidden_size"],
                                   hyper_parameters["rnn_dropout"],
                                   len(rel2id), 
                                   hyper_parameters["shaking_type"],
                                   hyper_parameters["inner_enc_type"],
                                   hyper_parameters["dist_emb_size"],
                                   hyper_parameters["ent_add_dist"],
                                   hyper_parameters["rel_add_dist"],
                                  )

rel_extractor = rel_extractor.to(device)


# In[ ]:


# all_paras = sum(x.numel() for x in rel_extractor.parameters())
# enc_paras = sum(x.numel() for x in encoder.parameters())


# In[ ]:


# print(all_paras, enc_paras)
# print(all_paras - enc_paras)


# # Metrics

# In[ ]:


def bias_loss(weights = None):
    if weights is not None:
        weights = torch.FloatTensor(weights).to(device)
    cross_en = nn.CrossEntropyLoss(weight = weights)  
    return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))
loss_func = bias_loss()


# In[ ]:


metrics = MetricsCalculator(handshaking_tagger)


# # Train

# In[ ]:


# train step
def train_step(batch_train_data, optimizer, loss_weights):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids,         batch_attention_mask, batch_token_type_ids,         tok2char_span_list, batch_ent_shaking_tag,         batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data
        
        batch_input_ids,         batch_attention_mask,         batch_token_type_ids,         batch_ent_shaking_tag,         batch_head_rel_shaking_tag,         batch_tail_rel_shaking_tag = (batch_input_ids.to(device), 
                                      batch_attention_mask.to(device), 
                                      batch_token_type_ids.to(device), 
                                      batch_ent_shaking_tag.to(device), 
                                      batch_head_rel_shaking_tag.to(device), 
                                      batch_tail_rel_shaking_tag.to(device)
                                     )
        
    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids,         tok2char_span_list, batch_ent_shaking_tag,         batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data
        
        batch_input_ids,         batch_ent_shaking_tag,         batch_head_rel_shaking_tag,         batch_tail_rel_shaking_tag = (batch_input_ids.to(device), 
                                      batch_ent_shaking_tag.to(device), 
                                      batch_head_rel_shaking_tag.to(device), 
                                      batch_tail_rel_shaking_tag.to(device)
                                     )
    

    # zero the parameter gradients
    optimizer.zero_grad()
    
    if config["encoder"] == "BERT":
        ent_shaking_outputs,         head_rel_shaking_outputs,         tail_rel_shaking_outputs = rel_extractor(batch_input_ids, 
                                                  batch_attention_mask, 
                                                  batch_token_type_ids, 
                                                 )
    elif config["encoder"] in {"BiLSTM", }:
        ent_shaking_outputs,         head_rel_shaking_outputs,         tail_rel_shaking_outputs = rel_extractor(batch_input_ids)
    
    w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
    loss = w_ent * loss_func(ent_shaking_outputs, batch_ent_shaking_tag) +             w_rel * loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) +             w_rel * loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)
    
    loss.backward()
    optimizer.step()
    
    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, 
                                          batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, 
                                             batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, 
                                             batch_tail_rel_shaking_tag)
    
    return loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item()

# valid step
def valid_step(batch_valid_data):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids,         batch_attention_mask, batch_token_type_ids,         tok2char_span_list, batch_ent_shaking_tag,         batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
        
        batch_input_ids,         batch_attention_mask,         batch_token_type_ids,         batch_ent_shaking_tag,         batch_head_rel_shaking_tag,         batch_tail_rel_shaking_tag = (batch_input_ids.to(device), 
                                      batch_attention_mask.to(device), 
                                      batch_token_type_ids.to(device), 
                                      batch_ent_shaking_tag.to(device), 
                                      batch_head_rel_shaking_tag.to(device), 
                                      batch_tail_rel_shaking_tag.to(device)
                                     )
        
    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids,         tok2char_span_list, batch_ent_shaking_tag,         batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
        
        batch_input_ids,         batch_ent_shaking_tag,         batch_head_rel_shaking_tag,         batch_tail_rel_shaking_tag = (batch_input_ids.to(device), 
                                      batch_ent_shaking_tag.to(device), 
                                      batch_head_rel_shaking_tag.to(device), 
                                      batch_tail_rel_shaking_tag.to(device)
                                     )
    
    with torch.no_grad():
        if config["encoder"] == "BERT":
            ent_shaking_outputs,             head_rel_shaking_outputs,             tail_rel_shaking_outputs = rel_extractor(batch_input_ids, 
                                                      batch_attention_mask, 
                                                      batch_token_type_ids, 
                                                     )
        elif config["encoder"] in {"BiLSTM", }:
            ent_shaking_outputs,             head_rel_shaking_outputs,             tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

    
    ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs, 
                                          batch_ent_shaking_tag)
    head_rel_sample_acc = metrics.get_sample_accuracy(head_rel_shaking_outputs, 
                                             batch_head_rel_shaking_tag)
    tail_rel_sample_acc = metrics.get_sample_accuracy(tail_rel_shaking_outputs, 
                                             batch_tail_rel_shaking_tag)
    
    rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list, 
                                    ent_shaking_outputs,
                                    head_rel_shaking_outputs,
                                    tail_rel_shaking_outputs,
                                    hyper_parameters["match_pattern"]
                                    )
    
    return ent_sample_acc.item(), head_rel_sample_acc.item(), tail_rel_sample_acc.item(), rel_cpg


# In[ ]:


max_f1 = 0.
def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):  
    def train(dataloader, ep):
        # train
        rel_extractor.train()
        
        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1 # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}
            
            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = train_step(batch_train_data, optimizer, loss_weights)
            scheduler.step()
            
            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc
            
            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)
            
            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " +                                 "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," +                                  "lr: {}, batch_time: {}, total_time: {} -------------"
                    
            print(batch_print_format.format(experiment_name, config["run_name"], 
                                            ep + 1, num_epoch, 
                                            batch_ind + 1, len(dataloader), 
                                            avg_loss, 
                                            avg_ent_sample_acc,
                                            avg_head_rel_sample_acc,
                                            avg_tail_rel_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                           ), end="")
            
            if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_ent_seq_acc": avg_ent_sample_acc,
                    "train_head_rel_acc": avg_head_rel_sample_acc,
                    "train_tail_rel_acc": avg_tail_rel_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })
                
        if config["logger"] != "wandb": # only log once for training if logger is not wandb
                logger.log({
                    "train_loss": avg_loss,
                    "train_ent_seq_acc": avg_ent_sample_acc,
                    "train_head_rel_acc": avg_head_rel_sample_acc,
                    "train_tail_rel_acc": avg_tail_rel_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                }) 
            
        
    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()
        
        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc = "Validating")):
            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, rel_cpg = valid_step(batch_valid_data)

            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc
            
            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_ent_sample_acc = total_ent_sample_acc / len(dataloader)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / len(dataloader)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(dataloader)
        
        rel_prf = metrics.get_prf_scores(total_rel_correct_num, total_rel_pred_num, total_rel_gold_num)
        
        log_dict = {
                        "val_ent_seq_acc": avg_ent_sample_acc,
                        "val_head_rel_acc": avg_head_rel_sample_acc,
                        "val_tail_rel_acc": avg_tail_rel_sample_acc,
                        "val_prec": rel_prf[0],
                        "val_recall": rel_prf[1],
                        "val_f1": rel_prf[2],
                        "time": time.time() - t_ep,
                    }
        logger.log(log_dict)
        pprint(log_dict)
        
        return rel_prf[2]
        
    for ep in range(num_epoch):
        train(train_dataloader, ep)   
        valid_f1 = valid(valid_dataloader, ep)
        
        global max_f1
        if valid_f1 >= max_f1: 
            max_f1 = valid_f1
            if valid_f1 > config["f1_2_save"]: # save the best model
                modle_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                torch.save(rel_extractor.state_dict(), os.path.join(model_state_dict_dir, "model_state_dict_{}.pt".format(modle_state_num)))
#                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
#                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num))) 
        print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))


# In[ ]:


# optimizer
init_learning_rate = float(hyper_parameters["lr"])
optimizer = torch.optim.Adam(rel_extractor.parameters(), lr = init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    
elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_steps, gamma = decay_rate)


# In[ ]:


if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))

train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])

