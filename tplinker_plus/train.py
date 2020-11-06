#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import os
from tqdm import tqdm
import re
from IPython.core.debugger import set_trace
from pprint import pprint
import unicodedata
from transformers import BertModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
import logging
from common.utils import Preprocessor, DefaultLogger
from tplinker_plus import (HandshakingTaggingScheme,
                          DataMaker4Bert, 
                          DataMaker4BiLSTM, 
                          TPLinkerPlusBert, 
                          TPLinkerPlusBiLSTM,
                          MetricsCalculator)
import wandb
from glove import Glove
import numpy as np
import config


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
ent2id_path = os.path.join(data_home, experiment_name, config["ent2id"])


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


# In[ ]:


# count_neg = 0 # 74.8% are neg samples 0.7485367594575303
# for example in train_data + valid_data:
#     if len(example["relation_list"]) == 0 and len(example["entity_list"]) == 0:
#         count_neg += 1
# print(count_neg/len(indexed_train_data + indexed_valid_data))


# # Tagger (Decoder)

# In[ ]:


max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
rel2id = json.load(open(rel2id_path, "r", encoding = "utf-8"))
ent2id = json.load(open(ent2id_path, "r", encoding = "utf-8"))
handshaking_tagger = HandshakingTaggingScheme(rel2id, max_seq_len, ent2id)
tag_size = handshaking_tagger.get_tag_size()


# In[ ]:


def sample_equal_to(sample1, sample2):
    assert sample1["id"] == sample2["id"]
    assert sample1["text"] == sample2["text"]
    memory_set = set()
    for rel in sample2["relation_list"]:
        memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subject"], 
                                                             rel["predicate"], 
                                                             rel["object"], 
                                                             *rel["subj_tok_span"], 
                                                             *rel["obj_tok_span"])
        memory_set.add(memory)
    for rel in sample1["relation_list"]:
        memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subject"], 
                                                             rel["predicate"], 
                                                             rel["object"], 
                                                             *rel["subj_tok_span"], 
                                                             *rel["obj_tok_span"])
        if memory not in memory_set:
            set_trace()
            return False
    return True


# In[ ]:


# # check tagging and decoding
# batch_size = hyper_parameters["batch_size"]
# for idx in tqdm(range(0, len(train_data), batch_size), desc = "check tagging and decoding"):
#     batch_matrix_spots = []
#     batch_data = train_data[idx:idx + batch_size]
#     for sample in batch_data:
#         matrix_spots = handshaking_tagger.get_spots(sample)
# #         %timeit shaking_tagger.get_spots(sample)
#         batch_matrix_spots.append(matrix_spots)
    
#     # tagging
#     # batch_shaking_tag: (batch_size, rel_id, seq_len, seq_len)
#     batch_shaking_tag = handshaking_tagger.spots2shaking_tag4batch(batch_matrix_spots)
# #     %timeit shaking_tagger.spots2shaking_tag4batch(batch_matrix_spots) #0.3s
    
#     for batch_idx in range(len(batch_data)):
#         gold_sample = batch_data[batch_idx]
#         shaking_tag = batch_shaking_tag[batch_idx]
#         # decode
#         text = batch_data[batch_idx]["text"]
#         tok2char_span = get_tok2char_span_map(text)
#         rel_list = handshaking_tagger.decode_rel(text, shaking_tag, tok2char_span)
#         pred_sample = {
#             "text": text,
#             "id": gold_sample["id"],
#             "relation_list": rel_list,
#         }
        
        
#         if not sample_equal_to(pred_sample, gold_sample) or not sample_equal_to(gold_sample, pred_sample):
#             set_trace()


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
# tok2char_span_list, batch_shaking_tag = batch_data

# print(text_list[0])
# print()
# print(tokenizer.decode(batch_input_ids[0].tolist()))
# print(batch_input_ids.size())
# print(batch_attention_mask.size())
# print(batch_token_type_ids.size())
# print(len(tok2char_span_list))
# print(batch_shaking_tag.size())

# # decode 
# idx = 2
# print(text_list[idx])
# shaking_tag = batch_shaking_tag[idx]
# text = text_list[idx]
# tok2char_span = tok2char_span_list[idx]
# handshaking_tagger.decode_rel(text, shaking_tag, tok2char_span)


# # Model

# In[ ]:


if config["encoder"] == "BERT":
    encoder = BertModel.from_pretrained(config["bert_path"])
    hidden_size = encoder.config.hidden_size
    
    rel_extractor = TPLinkerPlusBert(encoder, 
                                     tag_size,
                                     hyper_parameters["shaking_type"],
                                     hyper_parameters["inner_enc_type"],
                                     hyper_parameters["tok_pair_sample_rate"]
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
    
    rel_extractor = TPLinkerPlusBiLSTM(word_embedding_init_matrix, 
                                       hyper_parameters["emb_dropout"], 
                                       hyper_parameters["enc_hidden_size"], 
                                       hyper_parameters["dec_hidden_size"],
                                       hyper_parameters["rnn_dropout"],
                                       tag_size, 
                                       hyper_parameters["shaking_type"],
                                       hyper_parameters["inner_enc_type"],
                                       hyper_parameters["tok_pair_sample_rate"],
                                      )

rel_extractor = rel_extractor.to(device)


# In[ ]:


# # test outputs
# rel_extractor.train()
# with torch.no_grad():
#     outputs, sampled_tok_pair_indices = rel_extractor(batch_input_ids.to(device), 
#                                    batch_attention_mask.to(device), 
#                                    batch_token_type_ids.to(device),
#                                 )
# print(outputs.size())
# if rel_extractor.training:
#     print(sampled_tok_pair_indices.size())


# # Metrics

# In[ ]:


metrics = MetricsCalculator(handshaking_tagger)
loss_func = lambda y_pred, y_true: metrics.loss_func(y_pred, y_true, ghm = hyper_parameters["ghm"])


# # Train

# In[ ]:


# train step
def train_step(batch_train_data, optimizer):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids,         batch_attention_mask, batch_token_type_ids,         tok2char_span_list, batch_shaking_tag = batch_train_data
        
        batch_input_ids,         batch_attention_mask,         batch_token_type_ids,         batch_shaking_tag = (batch_input_ids.to(device), 
                              batch_attention_mask.to(device), 
                              batch_token_type_ids.to(device), 
                              batch_shaking_tag.to(device)
                             )
        
    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids,         tok2char_span_list, batch_shaking_tag = batch_train_data
        
        batch_input_ids,         batch_shaking_tag = (batch_input_ids.to(device), 
                              batch_shaking_tag.to(device)
                            )
    

    # zero the parameter gradients
    optimizer.zero_grad()
    if config["encoder"] == "BERT":
        pred_small_shaking_outputs, sampled_tok_pair_indices = rel_extractor(batch_input_ids, 
                                                                            batch_attention_mask, 
                                                                            batch_token_type_ids
                                                                             )
    elif config["encoder"] in {"BiLSTM", }:
        pred_small_shaking_outputs, sampled_tok_pair_indices = rel_extractor(batch_input_ids)

    # sampled_tok_pair_indices: (batch_size, ~segment_len)
    # batch_small_shaking_tag: (batch_size, ~segment_len, tag_size)
    batch_small_shaking_tag = batch_shaking_tag.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, tag_size))
    
    loss = loss_func(pred_small_shaking_outputs, batch_small_shaking_tag)
#     set_trace()
#     t1 = time.time()
    loss.backward()
    optimizer.step()
#     print("bp: {}".format(time.time() - t1))
    pred_small_shaking_tag = (pred_small_shaking_outputs > 0.).long()
    sample_acc = metrics.get_sample_accuracy(pred_small_shaking_tag,
                                             batch_small_shaking_tag)
    
    return loss.item(), sample_acc.item()

# valid step
def valid_step(batch_valid_data):
    if config["encoder"] == "BERT":
        sample_list, batch_input_ids,         batch_attention_mask, batch_token_type_ids,         tok2char_span_list, batch_shaking_tag = batch_valid_data
        
        batch_input_ids,         batch_attention_mask,         batch_token_type_ids,         batch_shaking_tag = (batch_input_ids.to(device), 
                              batch_attention_mask.to(device), 
                              batch_token_type_ids.to(device), 
                              batch_shaking_tag.to(device)
                             )
        
    elif config["encoder"] in {"BiLSTM", }:
        sample_list, batch_input_ids,         tok2char_span_list, batch_shaking_tag = batch_valid_data
        
        batch_input_ids,         batch_shaking_tag = (batch_input_ids.to(device), 
                              batch_shaking_tag.to(device)
                             )
    with torch.no_grad():
        if config["encoder"] == "BERT":
            pred_shaking_outputs, _ = rel_extractor(batch_input_ids, 
                                      batch_attention_mask, 
                                      batch_token_type_ids, 
                                     )
        elif config["encoder"] in {"BiLSTM", }:
            pred_shaking_outputs, _ = rel_extractor(batch_input_ids)
            
    pred_shaking_tag = (pred_shaking_outputs > 0.).long()
    sample_acc = metrics.get_sample_accuracy(pred_shaking_tag,
                                             batch_shaking_tag)
    
    cpg_dict = metrics.get_cpg(sample_list, 
                                          tok2char_span_list, 
                                          pred_shaking_tag,
                                          hyper_parameters["match_pattern"])
    return sample_acc.item(), cpg_dict


# In[ ]:


max_f1 = 0.
def train_n_valid(train_dataloader, dev_dataloader, optimizer, scheduler, num_epoch):  
    def train(dataloader, ep):
        # train
        rel_extractor.train()
        
        t_ep = time.time()
        total_loss, total_sample_acc = 0., 0.
        for batch_ind, batch_train_data in enumerate(dataloader):
            t_batch = time.time()
            
            loss, sample_acc = train_step(batch_train_data, optimizer)
            
            total_loss += loss
            total_sample_acc += sample_acc
            
            avg_loss = total_loss / (batch_ind + 1)
            
            # scheduler
            if hyper_parameters["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()
                
            avg_sample_acc = total_sample_acc / (batch_ind + 1)
            
            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " +                                 "t_sample_acc: {}," +                                  "lr: {}, batch_time: {}, total_time: {} -------------"
                    
            print(batch_print_format.format(experiment_name, config["run_name"], 
                                            ep + 1, num_epoch, 
                                            batch_ind + 1, len(dataloader), 
                                            avg_loss, 
                                            avg_sample_acc,
                                            optimizer.param_groups[0]['lr'],
                                            time.time() - t_batch,
                                            time.time() - t_ep,
                                           ), end="")
            
            if config["logger"] == "wandb" and batch_ind % hyper_parameters["log_interval"] == 0:
                logger.log({
                    "train_loss": avg_loss,
                    "train_small_shaking_seq_acc": avg_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                })
                
        if config["logger"] != "wandb": # only log once for training if logger is not wandb
                logger.log({
                    "train_loss": avg_loss,
                    "train_small_shaking_seq_acc": avg_sample_acc,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "time": time.time() - t_ep,
                }) 
            
        
    def valid(dataloader, ep):
        # valid
        rel_extractor.eval()
        
        t_ep = time.time()
        total_sample_acc = 0.
#         total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
#         total_ent_correct_num, total_ent_pred_num, total_ent_gold_num = 0, 0, 0
        total_cpg_dict = {}
        for batch_ind, batch_valid_data in enumerate(tqdm(dataloader, desc = "Validating")):
            sample_acc, cpg_dict = valid_step(batch_valid_data)
            total_sample_acc += sample_acc
            
            # init total_cpg_dict
            for k in cpg_dict.keys():
                if k not in total_cpg_dict:
                    total_cpg_dict[k] = [0, 0, 0]
           
            for k, cpg in cpg_dict.items():
                for idx, n in enumerate(cpg):
                    total_cpg_dict[k][idx] += cpg[idx]
                    
#             total_rel_correct_num += rel_cpg[0]
#             total_rel_pred_num += rel_cpg[1]
#             total_rel_gold_num += rel_cpg[2]
            
#             total_ent_correct_num += ent_cpg[0]
#             total_ent_pred_num += ent_cpg[1]
#             total_ent_gold_num += ent_cpg[2]

        avg_sample_acc = total_sample_acc / len(dataloader)
        
        if "rel_cpg" in total_cpg_dict:
            rel_prf = metrics.get_prf_scores(total_cpg_dict["rel_cpg"][0], total_cpg_dict["rel_cpg"][1], total_cpg_dict["rel_cpg"][2])
            ent_prf = metrics.get_prf_scores(total_cpg_dict["ent_cpg"][0], total_cpg_dict["ent_cpg"][1], total_cpg_dict["ent_cpg"][2])
            final_score = rel_prf[2]
            log_dict = {
                            "val_shaking_tag_acc": avg_sample_acc,
                            "val_rel_prec": rel_prf[0],
                            "val_rel_recall": rel_prf[1],
                            "val_rel_f1": rel_prf[2],
                            "val_ent_prec": ent_prf[0],
                            "val_ent_recall": ent_prf[1],
                            "val_ent_f1": ent_prf[2],
                            "time": time.time() - t_ep,
                        }
        elif "trigger_iden_cpg" in total_cpg_dict:
            trigger_iden_prf = metrics.get_prf_scores(total_cpg_dict["trigger_iden_cpg"][0], 
                                                      total_cpg_dict["trigger_iden_cpg"][1], 
                                                      total_cpg_dict["trigger_iden_cpg"][2])
            trigger_class_prf = metrics.get_prf_scores(total_cpg_dict["trigger_class_cpg"][0], 
                                                      total_cpg_dict["trigger_class_cpg"][1], 
                                                      total_cpg_dict["trigger_class_cpg"][2])
            arg_iden_prf = metrics.get_prf_scores(total_cpg_dict["arg_iden_cpg"][0], total_cpg_dict["arg_iden_cpg"][1], total_cpg_dict["arg_iden_cpg"][2])
            arg_class_prf = metrics.get_prf_scores(total_cpg_dict["arg_class_cpg"][0], total_cpg_dict["arg_class_cpg"][1], total_cpg_dict["arg_class_cpg"][2])
            final_score = arg_class_prf[2]
            log_dict = {
                            "val_shaking_tag_acc": avg_sample_acc,
                            "val_trigger_iden_prec": trigger_iden_prf[0],
                            "val_trigger_iden_recall": trigger_iden_prf[1],
                            "val_trigger_iden_f1": trigger_iden_prf[2],
                            "val_trigger_class_prec": trigger_class_prf[0],
                            "val_trigger_class_recall": trigger_class_prf[1],
                            "val_trigger_class_f1": trigger_class_prf[2],
                            "val_arg_iden_prec": arg_iden_prf[0],
                            "val_arg_iden_recall": arg_iden_prf[1],
                            "val_arg_iden_f1": arg_iden_prf[2],
                            "val_arg_class_prec": arg_class_prf[0],
                            "val_arg_class_recall": arg_class_prf[1],
                            "val_arg_class_f1": arg_class_prf[2],
                            "time": time.time() - t_ep,
                        }
            
        logger.log(log_dict)
        pprint(log_dict)
        
        return final_score
        
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


# In[ ]:


if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_dataloader) * rewarm_epoch_num, T_mult)
    
elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = decay_steps, gamma = decay_rate)

elif hyper_parameters["scheduler"] == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", verbose = True, patience = 6)


# In[ ]:


if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(model_state_path.split("/")[-1]))


train_n_valid(train_dataloader, valid_dataloader, optimizer, scheduler, hyper_parameters["epochs"])

