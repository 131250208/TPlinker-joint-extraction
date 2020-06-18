import re
from tqdm import tqdm
import torch
from IPython.core.debugger import set_trace
import copy
import torch
import torch.nn as nn
import json
from torch.nn.parameter import Parameter

class DefaultLogger:
    def __init__(self, log_path, project, run_name, hyperparameter):
        self.log_path = log_path
        
        self.log("project: {}, run_name: {}\n".format(project, run_name))
        hyperparameters_format = "--------------hypter_parameters------------------- \n{}\n-----------------------------------------"
        self.log(hyperparameters_format.format(json.dumps(hyperparameter, indent = 4)))

    def log(self, text):
        print(text)
        open(self.log_path, "a", encoding = "utf-8").write("{}\n".format(text))
        
class Preprocessor:
    '''
    1. transform the dataset to normal format, which can fit in our codes
    2. add token level span to all entities in the relations, which will be used in tagging phase
    '''
    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func
    
    def transform_data(self, data, ori_format, dataset_type, add_id = True):
        '''
        This function can only deal with three original format used in the experiments. 
        If you want to feed new dataset to the model, just define your own function to transform data.
        data: original data
        ori_format: "casrel", "joint_re", "raw_nyt"
        dataset_type: "train", "valid", "test"; only for generate id for the data
        '''
        normal_sample_list = []
        for ind, sample in tqdm(enumerate(data), desc = "Transforming data format"):
            if ori_format == "casrel":
                text = sample["text"]
                rel_list = sample["triple_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "joint_re":
                text = " ".join(sample["tokens"])
                rel_list = sample["spo_list"]
                subj_key, pred_key, obj_key = 0, 1, 2
            elif ori_format == "raw_nyt":
                text = sample["sentText"]
                rel_list = sample["relationMentions"]
                subj_key, pred_key, obj_key = "em1Text", "label", "em2Text"

            normal_sample = {
                "text": text,
            }
            if add_id:
                normal_sample["id"] = "{}_{}".format(dataset_type, ind)
            normal_rel_list = []
            for rel in rel_list:
                normal_rel = {
                    "subject": rel[subj_key],
                    "predicate": rel[pred_key],
                    "object": rel[obj_key],
                }
                normal_rel_list.append(normal_rel)
            normal_sample["relation_list"] = normal_rel_list
            normal_sample_list.append(normal_sample)
            
        return self._clean(normal_sample_list)
    
    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len = 50, encoder = "BERT"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc = "Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)

            # sliding on token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT": # if use bert, try not to split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = (char_span_list[0][0], char_span_list[-1][1])
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "relation_list": [],
                    }
                    
                for rel in sample["relation_list"]:
                    subj_span = rel["subj_span"]
                    obj_span = rel["obj_span"]
                    # if subject and object are included this subtext, save this rel in new sample
                    if subj_span[0] >= start_ind and subj_span[1] <= end_ind \
                        and obj_span[0] >= start_ind and obj_span[1] <= end_ind: 
                        new_rel = copy.deepcopy(rel)
                        new_rel["subj_span"] = (subj_span[0] - start_ind, subj_span[1] - start_ind)
                        new_rel["obj_span"] = (obj_span[0] - start_ind, obj_span[1] - start_ind)
                        new_sample["relation_list"].append(new_rel)

                if len(new_sample["relation_list"]) > 0:
                    split_sample_list.append(new_sample)
                
                if end_ind > len(tokens):
                    break
                    
            new_sample_list.extend(split_sample_list)
        return new_sample_list
    
    def _clean(self, dataset):
        def clean_text(text):
            text = re.sub("�", "", text)
#             text = re.sub("([A-Za-z]+)", r" \1 ", text)
#             text = re.sub("(\d+)", r" \1 ", text)
#             text = re.sub("\s+", " ", text).strip()
            return text 
        for sample in tqdm(dataset, desc = "Clean"):
            sample["text"] = clean_text(sample["text"])
            for rel in sample["relation_list"]:
                rel["subject"] = clean_text(rel["subject"])
                rel["object"] = clean_text(rel["object"])
        return dataset
    
    def _get_char2tok_span(self, text):
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)] # 除了空格，其他字符均有对应token
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1
        return char2tok_span

    def _get_ent2char_spans(self, text, entities):
        entities = sorted(entities, key = lambda x: len(x), reverse = True)
        text_cp = " {} ".format(text[:])
        ent2char_spans = {}
        for ent in entities:
            spans = []
            for m in re.finditer(re.escape(" {} ".format(ent)), text_cp):
                span = (m.span()[0], m.span()[1] - 2)
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans

    def _get_ent2tok_spans(self, text, entities):
        ent2char_spans = self._get_ent2char_spans(text, entities)
        char2tok_span = self._get_char2tok_span(text)
        ent2tok_spans = {}
        for ent, char_spans in ent2char_spans.items():
            tok_spans = []
            for char_sp in char_spans:
                tok_span_list = char2tok_span[char_sp[0]:char_sp[1]]
                tok_span = (tok_span_list[0][0], tok_span_list[-1][1])
                tok_spans.append(tok_span)
            ent2tok_spans[ent] = tok_spans
        return ent2tok_spans

    def add_tok_spans(self, dataset):
        for sample in tqdm(dataset, desc = "Adding token level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            ent2tok_spans = self._get_ent2tok_spans(sample["text"], entities)
            
            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_spans = ent2tok_spans[rel["subject"]]
                obj_spans = ent2tok_spans[rel["object"]]
                for subj_sp in subj_spans:
                    for obj_sp in obj_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_span": subj_sp,
                            "obj_span": obj_sp,
                            "predicate": rel["predicate"],
                        })
            sample["relation_list"] = new_relation_list

class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id, max_seq_len):
        super(HandshakingTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind:rel for rel, ind in rel2id.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_:tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_:tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,    
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_:tag for tag, id_ in self.tag2id_tail_rel.items()}

        # mapping shaking sequence and matrix
        self.matrix_size = max_seq_len
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        self.shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(self.matrix_size) for end_ind in list(range(self.matrix_size))[ind:]]

        self.matrix_ind2shaking_ind = [[0 for i in range(self.matrix_size)] for j in range(self.matrix_size)]
        for shaking_ind, matrix_ind in enumerate(self.shaking_ind2matrix_ind):
            self.matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind

    def get_spots(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], [] 

        for rel in sample["relation_list"]:
            subj_span = rel["subj_span"]
            obj_span = rel["obj_span"]
            ent_matrix_spots.append((subj_span[0], subj_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_span[0], obj_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if  subj_span[0] <= obj_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_span[0], obj_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_span[0], subj_span[0], self.tag2id_head_rel["REL-OH2SH"]))
                
            if subj_span[1] <= obj_span[1]:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_span[1] - 1, obj_span[1] - 1, self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_span[1] - 1, subj_span[1] - 1, self.tag2id_tail_rel["REL-OT2ST"]))
                
        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            shake_seq_tag: (shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    def spots2shaking_tag(self, spots):
        '''
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return: 
            shake_seq_tag: (rel_size, shaking_seq_len, )
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        shaking_seq_tag = torch.zeros(len(self.rel2id), shaking_seq_len).long()
        for sp in spots:
            shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
            shaking_seq_tag[sp[0]][shaking_ind] = sp[3]
        return shaking_seq_tag


    def sharing_spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return: 
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def spots2shaking_tag4batch(self, batch_spots):
        '''
        convert spots to batch shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail_relation
        return: 
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''
        shaking_seq_len = self.matrix_size * (self.matrix_size + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), len(self.rel2id), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = self.matrix_ind2shaking_ind[sp[1]][sp[2]]
                tag_id = sp[3]
                rel_id = sp[0]
                batch_shaking_seq_tag[batch_id][rel_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    def get_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = self.shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = self.shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots


    def decode_rel_fr_shaking_tag(self,
                                  text, 
                                  ent_shaking_tag, 
                                  head_rel_shaking_tag, 
                                  tail_rel_shaking_tag, 
                                  tok2char_span):
        '''
        ent shaking tag: (shaking_seq_len, )
        head rel and tail rel shaking_tag: size = (rel_size, shaking_seq_len, )
        '''
        rel_list = []
        
        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(ent_shaking_tag)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(head_rel_shaking_tag)
        tail_rel_matrix_spots = self.get_spots_fr_shaking_tag(tail_rel_shaking_tag)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2]
            if tag_id != self.tag2id_ent["ENT-H2T"]:
                continue
            
            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = (char_span_list[0][0], char_span_list[-1][1])
            ent_text = text[char_sp[0]:char_sp[1]] 
            
            head_key = sp[0] # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append({
                "text": ent_text,
                "span": (sp[0], sp[1] + 1),
            })
            
        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_tail_rel["REL-ST2OT"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[1], sp[2])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel["REL-OT2ST"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[2], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            
            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                subj_head_key, obj_head_key = sp[1], sp[2]
            elif tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                subj_head_key, obj_head_key = sp[2], sp[1]
                
            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[subj_head_key] # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key] # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = "{}-{}-{}".format(rel_id, subj["span"][1] - 1, obj["span"][1] - 1)
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation 
                        continue
                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_span": subj["span"],
                        "obj_span": obj["span"],
                        "predicate": self.id2rel[rel_id],
                    })
        return rel_list

class DataMaker4Bert():
    def __init__(self, tokenizer, handshaking_tagger):
        self.tokenizer = tokenizer
        self.handshaking_tagger = handshaking_tagger
    
    def get_indexed_train_valid_data(self, data, max_seq_len):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            text_id = sample["id"]
            # codes for bert input
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = False,
                                    max_length = max_seq_len, 
                                    pad_to_max_length = True)


            # tagging
            spots_tuple = self.handshaking_tagger.get_spots(sample)

            # get codes
            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (text_id,
                     text, 
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     spots_tuple,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def get_indexed_pred_data(self, data, max_seq_len):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed pred data"):
            text = sample["text"] 
            text_id = sample["id"]
            # @specific
            codes = self.tokenizer.encode_plus(text, 
                                    return_offsets_mapping = True, 
                                    add_special_tokens = False,
                                    max_length = max_seq_len, 
                                    pad_to_max_length = True)

            input_ids = torch.tensor(codes["input_ids"]).long()
            attention_mask = torch.tensor(codes["attention_mask"]).long()
            token_type_ids = torch.tensor(codes["token_type_ids"]).long()
            tok2char_span = codes["offset_mapping"]

            sample_tp = (text_id,
                     text, 
                     input_ids,
                     attention_mask,
                     token_type_ids,
                     tok2char_span,
                     )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_train_dev_batch(self, batch_data):
        text_id_list = []
        text_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for sample in batch_data:
            text_id_list.append(sample[0])
            text_list.append(sample[1])
            input_ids_list.append(sample[2])
            attention_mask_list.append(sample[3])        
            token_type_ids_list.append(sample[4])        
            tok2char_span_list.append(sample[5])

            ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = sample[6]
            ent_spots_list.append(ent_matrix_spots)
            head_rel_spots_list.append(head_rel_matrix_spots)
            tail_rel_spots_list.append(tail_rel_matrix_spots)

        # @specific: indexed by bert tokenizer
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)

        batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
        batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
        batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return text_id_list, text_list, \
              batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, \
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag
    
    def generate_pred_batch(self, batch_data):
        text_ids = []
        text_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = [] 
        tok2char_span_list = []
        for sample in batch_data:
            text_ids.append(sample[0])
            text_list.append(sample[1])
            input_ids_list.append(sample[2])
            attention_mask_list.append(sample[3])        
            token_type_ids_list.append(sample[4])        
            tok2char_span_list.append(sample[5])
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        batch_attention_mask = torch.stack(attention_mask_list, dim = 0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim = 0)
        return text_ids, text_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list

class DataMaker4BiLSTM():
    def __init__(self, text2indices, get_tok2char_span_map, handshaking_tagger):
        self.text2indices = text2indices
        self.handshaking_tagger = handshaking_tagger
        self.get_tok2char_span_map = get_tok2char_span_map
        
    def get_indexed_train_valid_data(self, data, max_seq_len):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed train or valid data"):
            text = sample["text"]
            text_id = sample["id"]

            # tagging
            spots_tuple = self.handshaking_tagger.get_spots(sample)
            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (text_id,
                     text, 
                     input_ids,
                     tok2char_span,
                     spots_tuple,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def get_indexed_pred_data(self, data, max_seq_len):
        indexed_samples = []
        for ind, sample in tqdm(enumerate(data), desc = "Generate indexed pred data"):
            text = sample["text"]
            text_id = sample["id"]

            tok2char_span = self.get_tok2char_span_map(text)
            tok2char_span.extend([(-1, -1)] * (max_seq_len - len(tok2char_span)))
            input_ids = self.text2indices(text, max_seq_len)

            sample_tp = (text_id,
                     text, 
                     input_ids,
                     tok2char_span,
                    )
            indexed_samples.append(sample_tp)       
        return indexed_samples
    
    def generate_train_dev_batch(self, batch_data):
        text_id_list = []
        text_list = []
        input_ids_list = []
        tok2char_span_list = []
        ent_spots_list = []
        head_rel_spots_list = []
        tail_rel_spots_list = []

        for sample in batch_data:
            text_id_list.append(sample[0])
            text_list.append(sample[1])
            input_ids_list.append(sample[2])    
            tok2char_span_list.append(sample[3])

            ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = sample[4]
            ent_spots_list.append(ent_matrix_spots)
            head_rel_spots_list.append(head_rel_matrix_spots)
            tail_rel_spots_list.append(tail_rel_matrix_spots)

        batch_input_ids = torch.stack(input_ids_list, dim = 0)

        batch_ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag4batch(ent_spots_list)
        batch_head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(head_rel_spots_list)
        batch_tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag4batch(tail_rel_spots_list)

        return text_id_list, text_list, \
                batch_input_ids, tok2char_span_list, \
                batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag
    
    def generate_pred_batch(self, batch_data):
        text_ids = []
        text_list = []
        input_ids_list = []
        tok2char_span_list = []
        for sample in batch_data:
            text_ids.append(sample[0])
            text_list.append(sample[1])
            input_ids_list.append(sample[2])       
            tok2char_span_list.append(sample[3])
        batch_input_ids = torch.stack(input_ids_list, dim = 0)
        return text_ids, text_list, batch_input_ids, tok2char_span_list

class LayerNorm(nn.Module):
    def __init__(self, shape, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        shape: inputs.shape
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.shape = (shape[-1],)
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(self.shape))
        if self.scale:
            self.gamma = Parameter(torch.ones(self.shape))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=self.shape[0], bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢?
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            if self.center:
                # print(self.beta_dense.weight.shape, cond.shape)
                self.beta_dense(cond)
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

class HandshakingKernel(nn.Module):
    def __init__(self, fake_inputs, shaking_type):
        super().__init__()
        hidden_size = fake_inputs.size()[-1]
        self.cond_layer_norm = LayerNorm(fake_inputs.size(), hidden_size, conditional = True)
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "res_gate":
            self.Wg = nn.Linear(hidden_size, hidden_size)
            self.Wo = nn.Linear(hidden_size * 3, hidden_size)
            
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: 
                cln: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
                cat: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size * 2)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            # seq_len - ind: only shake afterwards
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1) 
            after_hiddens = seq_hiddens[:, ind:, :]
            if self.shaking_type == "cln":
                shaking_hiddens = self.cond_layer_norm(after_hiddens, repeat_hiddens)
            elif self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, after_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "res_gate":
                gate = torch.sigmoid(self.Wg(repeat_hiddens))
                cond_hiddens = after_hiddens * gate
                res_hiddens = torch.cat([repeat_hiddens, after_hiddens, cond_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.Wo(res_hiddens))
            shaking_hiddens_list.append(shaking_hiddens)
        shaking_hiddenss = torch.cat(shaking_hiddens_list, dim = 1)
        return shaking_hiddenss
    
class TPLinkerBert(nn.Module):
    def __init__(self, encoder, 
                 rel_size, 
                 fake_inputs, 
                 shaking_type
                ):
        super().__init__()
        self.encoder = encoder
        self.shaking_type = shaking_type
        shaking_hidden_size = encoder.config.hidden_size
        
        self.ent_fc = nn.Linear(shaking_hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(shaking_hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(shaking_hidden_size, 3) for _ in range(rel_size)]
        
        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(fake_inputs, shaking_type)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids, attention_mask, token_type_ids: (batch_size, seq_len)
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        
        ent_shaking_outputs = self.ent_fc(shaking_hiddens)
            
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens))
            
        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens))
        
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim = 1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim = 1)
        
        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs

class TPLinkerBiLSTM(nn.Module):
    def __init__(self, init_word_embedding_matrix, 
                 emb_dropout_rate, 
                 enc_hidden_size, 
                 dec_hidden_size, 
                 rnn_dropout_rate,
                 rel_size, 
                 fake_inputs, 
                 shaking_type):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze = False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1], 
                        enc_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.dec_lstm = nn.LSTM(enc_hidden_size, 
                        dec_hidden_size // 2, 
                        num_layers = 1, 
                        bidirectional = True, 
                        batch_first = True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        
        shaking_hidden_size = dec_hidden_size
        if shaking_type == "cat":
            shaking_hidden_size *= 2
            
        self.ent_fc = nn.Linear(shaking_hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(shaking_hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(shaking_hidden_size, 3) for _ in range(rel_size)]
        
        for ind, fc in enumerate(self.head_rel_fc_list):
            self.register_parameter("weight_4_head_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_head_rel{}".format(ind), fc.bias)
        for ind, fc in enumerate(self.tail_rel_fc_list):
            self.register_parameter("weight_4_tail_rel{}".format(ind), fc.weight)
            self.register_parameter("bias_4_tail_rel{}".format(ind), fc.bias)
            
        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(fake_inputs, shaking_type)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # embedding: (batch_size, seq_len, emb_dim)
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        # lstm_outputs: (batch_size, seq_len, enc_hidden_size)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        # lstm_outputs: (batch_size, seq_len, dec_hidden_size)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        
        ent_shaking_outputs = self.ent_fc(shaking_hiddens)
            
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens))
            
        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens))
        
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim = 1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim = 1)
        
        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs
    
class MetricsCalculator():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
        
    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)

        return sample_acc 
    def get_rel_cpg(self, text_list, offset_map_list, 
                 batch_pred_ent_shaking_outputs,
                 batch_pred_head_rel_shaking_outputs,
                 batch_pred_tail_rel_shaking_outputs,
                 batch_gold_ent_shaking_tag,
                 batch_gold_head_rel_shaking_tag,
                 batch_gold_tail_rel_shaking_tag):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim = -1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim = -1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim = -1)

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(text_list)):
            text = text_list[ind]
            offset_map = offset_map_list[ind]
            gold_ent_shaking_tag, pred_ent_shaking_tag = batch_gold_ent_shaking_tag[ind], batch_pred_ent_shaking_tag[ind]
            gold_head_rel_shaking_tag, pred_head_rel_shaking_tag = batch_gold_head_rel_shaking_tag[ind], batch_pred_head_rel_shaking_tag[ind]
            gold_tail_rel_shaking_tag, pred_tail_rel_shaking_tag = batch_gold_tail_rel_shaking_tag[ind], batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_ent_shaking_tag, 
                                                      pred_head_rel_shaking_tag, 
                                                      pred_tail_rel_shaking_tag, 
                                                      offset_map)
            gold_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      gold_ent_shaking_tag, 
                                                      gold_head_rel_shaking_tag, 
                                                      gold_tail_rel_shaking_tag, 
                                                      offset_map)

            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])

            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1