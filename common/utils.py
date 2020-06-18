import re
from tqdm import tqdm
from IPython.core.debugger import set_trace
import copy
import json

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
        This function can only deal with three original format used in the previous works. 
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
    
    def split_into_short_samples(self, sample_list, max_seq_len, sliding_len = 50, encoder = "BERT", data_type = "train"):
        new_sample_list = []
        for sample in tqdm(sample_list, desc = "Splitting into subtexts"):
            text_id = sample["id"]
            text = sample["text"]
            tokens = self._tokenize(text)
            tok2char_span = self._get_tok2char_span_map(text)

            # sliding at token level
            split_sample_list = []
            for start_ind in range(0, len(tokens), sliding_len):
                if encoder == "BERT": # if use bert, do not split a word into two samples
                    while "##" in tokens[start_ind]:
                        start_ind -= 1
                end_ind = start_ind + max_seq_len

                char_span_list = tok2char_span[start_ind:end_ind]
                char_level_span = [char_span_list[0][0], char_span_list[-1][1]]
                sub_text = text[char_level_span[0]:char_level_span[1]]

                new_sample = {
                    "id": text_id,
                    "text": sub_text,
                    "tok_offset": start_ind,
                    "char_offset": char_level_span[0],
                    }
                if data_type == "test":
                    if len(sub_text) > 0:
                        split_sample_list.append(new_sample)
                else:
                    sub_rel_list = []
                    for rel in sample["relation_list"]:
                        subj_tok_span = rel["subj_tok_span"]
                        obj_tok_span = rel["obj_tok_span"]
                        # if subject and object are included this subtext, save this rel in new sample
                        if subj_tok_span[0] >= start_ind and subj_tok_span[1] <= end_ind \
                            and obj_tok_span[0] >= start_ind and obj_tok_span[1] <= end_ind: 
                            new_rel = copy.deepcopy(rel)
                            new_rel["subj_tok_span"] = [subj_tok_span[0] - start_ind, subj_tok_span[1] - start_ind]
                            new_rel["obj_tok_span"] = [obj_tok_span[0] - start_ind, obj_tok_span[1] - start_ind]
                            new_rel["subj_char_span"][0] -= char_level_span[0]
                            new_rel["subj_char_span"][1] -= char_level_span[0]
                            new_rel["obj_char_span"][0] -= char_level_span[0]
                            new_rel["obj_char_span"][1] -= char_level_span[0]
                            sub_rel_list.append(new_rel)

                    if len(sub_rel_list) > 0:
                        new_sample["relation_list"] = sub_rel_list
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
        '''
        map character index to token level span
        '''
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)] # [-1, -1] is whitespace
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
    
    def add_char_spans(self, dataset):
        for sample in tqdm(dataset, desc = "Adding char level spans"):
            entities = [rel["subject"] for rel in sample["relation_list"]]
            entities.extend([rel["object"] for rel in sample["relation_list"]])
            ent2char_spans = self._get_ent2char_spans(sample["text"], entities)
            
            new_relation_list = []
            for rel in sample["relation_list"]:
                subj_tok_spans = ent2char_spans[rel["subject"]]
                obj_tok_spans = ent2char_spans[rel["object"]]
                for subj_sp in subj_tok_spans:
                    for obj_sp in obj_tok_spans:
                        new_relation_list.append({
                            "subject": rel["subject"],
                            "object": rel["object"],
                            "subj_char_span": subj_sp,
                            "obj_char_span": obj_sp,
                            "predicate": rel["predicate"],
                        })
            sample["relation_list"] = new_relation_list
                           
    def add_tok_spans(self, dataset):
        '''
        dataset must has char level span
        '''      
        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span
        
        for sample in tqdm(dataset, desc = "Adding token level spans"):
            text = sample["text"]
            char2tok_span = self._get_char2tok_span(sample["text"])
            for rel in sample["relation_list"]:
                subj_char_span = rel["subj_char_span"]
                obj_char_span = rel["obj_char_span"]
                rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
                rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
            