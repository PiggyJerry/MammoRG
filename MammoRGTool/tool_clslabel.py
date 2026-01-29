import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import re
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Union
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from correction import process_samples
from models.rel_model import RelModel
from transformers import BertTokenizer
from utils.tokenization import BasicTokenizer
from config import Config
from clean_text import clean_text
from scipy import stats  # Added for confidence interval calculation
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

DENSITY_CLASSES = ["脂肪型", "纤维腺体型", "不均匀致密型", "致密型", "BLA"]
BI_RADS_CLASSES = [
    "Bi-Rads 0", "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3",
    "Bi-Rads 4A","Bi-Rads 4B","Bi-Rads 4C",
    "Bi-Rads 5", "Bi-Rads 6", "BLA"
]
TRUE_BI_RADS_CLASSES = [
     "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3",
    "Bi-Rads 4",
    "Bi-Rads 5"
]
ENTITY_CLASSES = ["POS", "NEG", "UNC", "BLA"]
ENTITY_NAMES = [
    "钙化", "肿块", "乳腺增生", "皮肤增厚", "淋巴结肿大",
    "乳头凹陷", "结构扭曲",
    "悬韧带增粗", "结节", "结构不对称"
]

roman_to_arabic = {
    '0': 0,
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6,
    'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5, 'Ⅵ': 6,
    'ⅰ': 1, 'ⅱ': 2, 'ⅲ': 3, 'ⅳ': 4, 'ⅴ': 5, 'ⅵ': 6
}

birads_priority = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '4a': 4.1, '4b': 4.2, '4c': 4.3, '5': 5, '6': 6,
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6,
    'Ⅰ': 1, 'Ⅱ': 2, 'Ⅲ': 3, 'Ⅳ': 4, 'Ⅴ': 5, 'Ⅵ': 6,
    'ⅰ': 1, 'ⅱ': 2, 'ⅲ': 3, 'ⅳ': 4, 'ⅴ': 5, 'ⅵ': 6
}

def rule_based_get_birads(conclusion, laterality):
    if pd.isna(conclusion):
        return None

    full_birads_pattern = re.compile(
        r'(?:BI\s*-\s*RADS|BI\s*RADS|BI/RADS|分类)[-]?\s*[:：]?\s*'
        r'([IVXivxⅠⅡⅢⅣⅤⅥⅰⅱⅲⅳⅴⅵ]+|\d+[a-cA-C]?)',
        re.IGNORECASE
    )

    side_keywords = {
        'left': ['左', '左乳'],
        'right': ['右', '右乳'],
        'double': ['双乳', '双侧', '双侧乳']
    }

    side_positions = []
    for side_type, keywords in side_keywords.items():
        for keyword in keywords:
            for match in re.finditer(keyword, conclusion):
                side_positions.append({
                    'pos': match.start(),
                    'type': side_type,
                    'keyword': keyword
                })
    side_positions.sort(key=lambda x: x['pos'])

    birads_matches = []
    for match in full_birads_pattern.finditer(conclusion):
        full_text = match.group(0) 

        birads_class = match.group(1).upper()
        if birads_class in roman_to_arabic:
            birads_num = str(roman_to_arabic[birads_class])
        elif re.match(r'\d+[A-C]', birads_class):
            birads_num = birads_class.lower()
        else:
            birads_num = birads_class
            
        birads_matches.append({
            'full_text': full_text.strip(), 
            'text': birads_num,  
            'priority': birads_priority.get(birads_num, -1),
            'start': match.start(),
            'end': match.end()
        })

    birads_with_sides = []
    for birads in birads_matches:
        preceding_sides = [s for s in side_positions if s['pos'] < birads['start']]
        if preceding_sides:

            nearest_side = max(preceding_sides, key=lambda x: x['pos'])
            
    
            birads_with_sides.append({
                'full_text': birads['full_text'],  
                'birads': birads['text'],  
                'priority': birads['priority'],
                'side_type': nearest_side['type'],
                'side_keyword': nearest_side['keyword'],
                'side_pos': nearest_side['pos'],
                'birads_pos': birads['start'],
                'context': conclusion[nearest_side['pos']:birads['end']]
            })
        else:
        
            birads_with_sides.append({
                'full_text': birads['full_text'],  
                'birads': birads['text'],  
                'priority': birads['priority'],
                'side_type': None,
                'side_keyword': None,
                'side_pos': None,
                'birads_pos': birads['start'],
                'context': None
            })

 
    valid_birads = []
    if laterality == 'L':
      
        left_birads = [b for b in birads_with_sides if b['side_type'] == 'left']
        if left_birads:
            valid_birads = left_birads
        else:
            valid_birads = [b for b in birads_with_sides if b['side_type'] == 'double']
    elif laterality == 'R':
      
        right_birads = [b for b in birads_with_sides if b['side_type'] == 'right']
        if right_birads:
            valid_birads = right_birads
        else:
            valid_birads = [b for b in birads_with_sides if b['side_type'] == 'double']

    if not valid_birads:
        return None


    return max(valid_birads, key=lambda x: x['priority'])['full_text']


def rule_based_get_density(conclusion, laterality):

    if pd.isna(conclusion):
        return None

    
    side_keywords = {
        'left': ['左', '左乳'],
        'right': ['右', '右乳'],
        'double': ['双乳', '双侧', '双侧乳']
    }


    side_positions = []
    for side_type, keywords in side_keywords.items():
        for keyword in keywords:
            for match in re.finditer(keyword, conclusion):
                side_positions.append({
                    'pos': match.start(),
                    'type': side_type,
                    'keyword': keyword
                })

    side_positions.sort(key=lambda x: x['pos'])

    def find_density(text):

        text = text.lower()
        if any(kw in text for kw in ['不均匀致密型']):
            return '不均匀致密型'
        elif any(kw in text for kw in ['纤维腺体型']):
            return '纤维腺体型'
        elif any(kw in text for kw in ['致密型']):
            return '致密型'
        elif any(kw in text for kw in ['脂肪型']):
            return '脂肪型'
        return None

    density_paragraphs = []
    for i, char in enumerate(conclusion):
        if char in ['：', ':', '，', ',', '。', '；', ';'] or i == len(conclusion)-1:
            end_pos = i+1 if i < len(conclusion)-1 else i
            paragraph = conclusion[:end_pos]
            density = find_density(paragraph)
            if density:
                preceding_sides = [s for s in side_positions if s['pos'] < i]
                if preceding_sides:
                    nearest_side = max(preceding_sides, key=lambda x: x['pos'])
                    density_paragraphs.append({
                        'density': density,
                        'side_type': nearest_side['type'],
                        'side_keyword': nearest_side['keyword'],
                        'context': paragraph
                    })

    valid_densities = []
    if laterality == 'L':
        left_densities = [d for d in density_paragraphs if d['side_type'] == 'left']
        if left_densities:
            valid_densities = left_densities
        else:
            valid_densities = [d for d in density_paragraphs if d['side_type'] == 'double']
    elif laterality == 'R':
        right_densities = [d for d in density_paragraphs if d['side_type'] == 'right']
        if right_densities:
            valid_densities = right_densities
        else:
            valid_densities = [d for d in density_paragraphs if d['side_type'] == 'double']

    if valid_densities:
        return valid_densities[0]['density']
    return None

class ZhTokenizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('/home/user/MammoRG/MammoRGTool/pre_trained_bert/vocab.txt')
        self.vocab2id = self.tokenizer.vocab

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return_tokens = ["[CLS]"]
        for token in tokens:
            return_tokens.append(token)
            return_tokens.append("[unused1]")
        return_tokens += ["[SEP]"]
        return return_tokens

    def encode(self, text):
        return_tokens = self.tokenize(text)
        input_ids = [int(self.vocab2id.get(token, 100)) for token in return_tokens]
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask

def vector_to_dict(
    text,
    vector: Dict[str, Union[torch.Tensor, np.ndarray]],
    density_classes: List[str],
    bi_rads_classes: List[str],
    entity_classes: List[str],
    entity_names: List[str],
    logits=True
) -> Dict:
    def _tensor_to_index(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.float32).cpu().numpy()
        return np.argmax(tensor, axis=-1) if tensor.ndim > 1 else (np.argmax(tensor) if tensor.size > 1 else int(tensor))

    def _decode_breast(density_idx, bi_rads_idx, state_indices):
        density = (
            density_classes[density_idx] 
            if 0 <= density_idx < len(density_classes) 
            else "BLA"
        )

        bi_rads = (
            bi_rads_classes[bi_rads_idx] 
            if 0 <= bi_rads_idx < len(bi_rads_classes) 
            else "BLA"
        )

        entities = {}
        for i, name in enumerate(entity_names):
            if i < len(state_indices):
                idx = state_indices[i]
                entities[name] = (
                    entity_classes[idx] 
                    if 0 <= idx < len(entity_classes) 
                    else "BLA"
                )
            else:
                entities[name] = "BLA"

        return {
            "Density": density,
            "Bi-Rads": bi_rads,
            "Entities": entities
        }

    if logits:
        left_density_idx = _tensor_to_index(vector['left_density_logits'])[0][0]
        left_birads_idx = _tensor_to_index(vector['left_birads_logits'])[0][0]
        left_state_indices = _tensor_to_index(vector['left_state_logits'])[0]
        right_density_idx = _tensor_to_index(vector['right_density_logits'])[0][0]
        right_birads_idx = _tensor_to_index(vector['right_birads_logits'])[0][0]
        right_state_indices = _tensor_to_index(vector['right_state_logits'])[0]
    else:
        left_density_idx = vector['left_density_logits'][0][0]
        left_birads_idx = vector['left_birads_logits'][0][0]
        left_state_indices = vector['left_state_logits'][0]
        right_density_idx = vector['right_density_logits'][0][0]
        right_birads_idx = vector['right_birads_logits'][0][0]
        right_state_indices = vector['right_state_logits'][0]
    left_breast=_decode_breast(left_density_idx, left_birads_idx, left_state_indices)
    rule_based_density = rule_based_get_density(text, 'L')
    if rule_based_density and rule_based_density != 'BLA':
        left_breast['Density'] = rule_based_density
    rule_based_birads = rule_based_get_birads(text, 'L')
    if rule_based_birads and rule_based_birads != 'BLA':
        left_breast['Bi-Rads'] = rule_based_birads
    
    right_breast=_decode_breast(right_density_idx, right_birads_idx, right_state_indices)    
    rule_based_density = rule_based_get_density(text, 'R')
    if rule_based_density and rule_based_density != 'BLA':
        right_breast['Density'] = rule_based_density
    rule_based_birads = rule_based_get_birads(text, 'R')
    if rule_based_birads and rule_based_birads != 'BLA':
        right_breast['Bi-Rads'] = rule_based_birads
            

    if '右乳头未见凹陷' in text:
        right_breast['Entities']['乳头凹陷'] = 'NEG'
    if '左乳头未见凹陷' in text:
        left_breast['Entities']['乳头凹陷'] = 'NEG'
    if '乳头未见凹陷' in text:
        right_breast['Entities']['乳头凹陷'] = 'NEG'
        left_breast['Entities']['乳头凹陷'] = 'NEG'
            
    def process_not_seen_entities():
        sentences = re.split('[。；，]', text)
        
        for sentence in sentences:
            if '未见' in sentence and '结构扭曲' in sentence:
                if '双乳未见' in sentence:
                   
                    for entity in ENTITY_NAMES:
                        if entity in sentence:
                            
                            right_breast['Entities']['结构扭曲'] = 'NEG'
                            left_breast['Entities']['结构扭曲'] = 'NEG'
                
                elif '左乳未见' in sentence:
                   
                    for entity in ENTITY_NAMES:
                        if entity in sentence:
                            left_breast['Entities']['结构扭曲'] = 'NEG'
                
                elif '右乳未见' in sentence:
                  
                    for entity in ENTITY_NAMES:
                        if entity in sentence:
                            right_breast['Entities']['结构扭曲'] = 'NEG'
                

    process_not_seen_entities()

    if any(phrase in text for phrase in ['双乳悬韧带增粗', '双乳悬韧带轻度增厚']):
        right_breast['Entities']['悬韧带增粗'] = 'POS'
        left_breast['Entities']['悬韧带增粗'] = 'POS'
    
 
    if any(phrase in text for phrase in ['左乳悬韧带增粗', '左乳悬韧带轻度增厚']):
        left_breast['Entities']['悬韧带增粗'] = 'POS'
    
   
    if any(phrase in text for phrase in ['右乳悬韧带增粗', '右乳悬韧带轻度增厚']):
        right_breast['Entities']['悬韧带增粗'] = 'POS'
    
    
    if '，悬韧带增粗' in text:
        if not any(phrase in text for phrase in ['，未见悬韧带增粗', '，悬韧带未见增粗', '，悬韧带未见异常增粗']):
            right_breast['Entities']['悬韧带增粗'] = 'POS'
            left_breast['Entities']['悬韧带增粗'] = 'POS'
    
 
    if any(phrase in text for phrase in ['双乳未见悬韧带增粗', '双乳悬韧带未见增粗', '双乳悬韧带未见异常增粗']):
        right_breast['Entities']['悬韧带增粗'] = 'NEG'
        left_breast['Entities']['悬韧带增粗'] = 'NEG'
    
  
    if any(phrase in text for phrase in ['左乳未见悬韧带增粗', '左乳悬韧带未见增粗', '左乳悬韧带未见异常增粗']):
        left_breast['Entities']['悬韧带增粗'] = 'NEG'
    
  
    if any(phrase in text for phrase in ['右乳未见悬韧带增粗', '右乳悬韧带未见增粗', '右乳悬韧带未见异常增粗']):
        right_breast['Entities']['悬韧带增粗'] = 'NEG'
    
 
    if any(phrase in text for phrase in ['，未见悬韧带增粗', '，悬韧带未见增粗', '，悬韧带未见异常增粗']):
        right_breast['Entities']['悬韧带增粗'] = 'NEG'
        left_breast['Entities']['悬韧带增粗'] = 'NEG'
    

    if '乳腺增生' not in text:
        left_breast['Entities']['乳腺增生'] = 'BLA'
        right_breast['Entities']['乳腺增生'] = 'BLA'

    if '结构不对称' not in text and '局灶性不对称' not in text:
        right_breast['Entities']['结构不对称'] = 'BLA'
        left_breast['Entities']['结构不对称'] = 'BLA'

    def process_lymph_node_not_seen():
        lymph_keywords = ['淋巴结', '淋巴结肿大', '肿大淋巴结', '肿大的淋巴结']
        has_lymph_mention = any(keyword in text for keyword in lymph_keywords)
        
        if not has_lymph_mention:
            right_breast['Entities']['淋巴结肿大'] = 'BLA'
            left_breast['Entities']['淋巴结肿大'] = 'BLA'
            return
        
    
        sentences = re.split('[。；，]', text)
        
        for sentence in sentences:
     
            lymph_found = False
            lymph_keyword = None
            
            for keyword in ['肿大的淋巴结', '肿大淋巴结', '淋巴结肿大']:
                if keyword in sentence:
                    lymph_found = True
                    lymph_keyword = keyword
                    break
            
            if lymph_found:
            
                lymph_index = sentence.find(lymph_keyword)
              
                if lymph_index > 0 and '未见' in sentence[:lymph_index]:
                
                    if '双侧' in sentence:
                     
                        right_breast['Entities']['淋巴结肿大'] = 'NEG'
                        left_breast['Entities']['淋巴结肿大'] = 'NEG'
                    elif '左侧' in sentence:
                      
                        left_breast['Entities']['淋巴结肿大'] = 'NEG'
                    elif '右侧' in sentence:
                     
                        right_breast['Entities']['淋巴结肿大'] = 'NEG'

    process_lymph_node_not_seen()
            
    return {
        "Breast_assessment": {
            "Left_breast": left_breast,
            "Right_breast": right_breast
        }
    }

def helper(text):
    text = text.split(' ')
    text = ''.join(text)
    return text 

def compute_sample_f1(y_true: List[int], y_pred: List[int], labels_all: List[int] = None):
   
    if not y_true:
        return None

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    unique_true = np.unique(y_true_arr)
   
    if unique_true.size == 1:
        return float(accuracy_score(y_true_arr, y_pred_arr))
    else:
     
        if labels_all is not None:
            return float(f1_score(y_true_arr, y_pred_arr, average='macro', labels=labels_all, zero_division=0))
        else:
            return float(f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0))

def print_confusion_matrix(cm, labels):
    print("Pred →")
    print("True ↓")
    print("     " + "  ".join(labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]:<6} " + "  ".join(f"{x:4d}" for x in row))


class MammoRGTool(object):
    def __init__(self, output_dir=None):
        self.config = Config()
        self.id2rel = json.load(open(f'{current_dir}/data/rel2id.json'))[0]
        id2tag, self.tag2id = json.load(open(f'{current_dir}/data/tag2id.json'))
        self.tokenizer = ZhTokenizer()
        self.model = RelModel(self.config)
        self.model.load_state_dict(torch.load(self.config.checkpoint))

        self.model.to(device)
        self.model.eval()
        self.refs=json.load(open('/home/user/MammoRG/mammorg_data/split_data/vindr_mammo.json'))
        
        self.output_dir=output_dir
        
    def test(self, text):
        text=clean_text(text)
        origin_text=text
        orders = ['subject', 'relation', 'object']

        def to_tup(triple_list):
            ret = []
            for triple in triple_list:
                ret.append(tuple(triple))
            return ret
        
        probs = {
                'left_density_logits':[],
                'left_birads_logits':[],
                'left_state_logits':[],
                'right_density_logits':[],
                'right_birads_logits':[],
                'right_state_logits':[]
                }

        with torch.inference_mode():
            basic_tokenizer = BasicTokenizer(do_lower_case=False)
            basic_tokens = basic_tokenizer.tokenize(text)
            text = ' '.join(basic_tokens)
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.config.bert_max_len:
                tokens = tokens[: self.config.bert_max_len]
            token_ids, masks = self.tokenizer.encode(text)
            if len(token_ids) > self.config.bert_max_len:
                token_ids = token_ids[:self.config.bert_max_len]
                masks = masks[:self.config.bert_max_len]
            token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(device)
            masks = torch.from_numpy(np.array(masks)).unsqueeze(0).to(device)
            
            data={}
            data['token_ids']=token_ids
            data['mask']=masks
 
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(data,train=False)
            pred_triple_matrix,entity_statement=outputs['relation_output'].cpu()[0],outputs['entity_output']
            for key in probs.keys():
                probs[key].append(entity_statement[key].cpu())

            rel_numbers, seq_lens, seq_lens = pred_triple_matrix.shape
            relations, heads, tails = np.where(pred_triple_matrix > 0)

            triple_list = [] 

            pair_numbers = len(relations)

            if pair_numbers > 0:

                for i in range(pair_numbers):
                    r_index = relations[i]
                    h_start_index = heads[i]
                    t_start_index = tails[i]
                
                    if pred_triple_matrix[r_index][h_start_index][t_start_index] == self.tag2id['HB-TB'] and i+1 < pair_numbers:
                 
                        t_end_index = tails[i+1]
                        if pred_triple_matrix[r_index][h_start_index][t_end_index] == self.tag2id['HB-TE']:
                            
                            for h_end_index in range(h_start_index, seq_lens):
                               
                                if pred_triple_matrix[r_index][h_end_index][t_end_index] == self.tag2id['HE-TE']:

                                    sub_head, sub_tail = h_start_index, h_end_index
                                    obj_head, obj_tail = t_start_index, t_end_index
                                    sub = tokens[sub_head : sub_tail+1]
                                   
                                    sub = ''.join([i.lstrip("##") for i in sub])
                                    sub = ' '.join(sub.split('[unused1]')).strip()
                                    obj = tokens[obj_head : obj_tail+1]
                                   
                                    obj = ''.join([i.lstrip("##") for i in obj])
                                    obj = ' '.join(obj.split('[unused1]')).strip()
                                    rel = self.id2rel[str(int(r_index))]
                                    if len(sub) > 0 and len(obj) > 0:
                                        triple_list.append((sub, rel, obj))
                                    break


            triple_set = set()

            for s, r, o in triple_list:
                s = helper(s)
                o = helper(o)
                triple_set.add((s, r, o))

            triple_list = list(triple_set)
            triples_dict = set(triple_list)

            triples=[
                        dict(zip(orders, triple)) for triple in triples_dict
                    ]
            
            relations = []
            for item in triples:
                subject = item["subject"]
                relation = item["relation"]
                obj = item["object"]
                relations.append([subject, relation, obj]) 
                
            breast_assessment=vector_to_dict(
                    origin_text,
                    vector=entity_statement,
                    density_classes=DENSITY_CLASSES,
                    bi_rads_classes=BI_RADS_CLASSES,
                    entity_classes=ENTITY_CLASSES,
                    entity_names=ENTITY_NAMES
                )['Breast_assessment']
            

            def add_suspensory_ligament_relations():
              
                if any(phrase in origin_text for phrase in ['双乳悬韧带增粗', '双乳悬韧带轻度增厚']):
                    relations.append(["悬韧带增粗", "Located_at", "双乳"])
                
               
                if any(phrase in origin_text for phrase in ['左乳悬韧带增粗', '左乳悬韧带轻度增厚']):
                    relations.append(["悬韧带增粗", "Located_at", "左乳"])
                
              
                if any(phrase in origin_text for phrase in ['右乳悬韧带增粗', '右乳悬韧带轻度增厚']):
                    relations.append(["悬韧带增粗", "Located_at", "右乳"])
                
            
                if '，悬韧带增粗' in origin_text:
                    if not any(phrase in origin_text for phrase in ['，未见悬韧带增粗', '，悬韧带未见增粗', '，悬韧带未见异常增粗']):
                        relations.append(["悬韧带增粗", "Located_at", "双乳"])

            add_suspensory_ligament_relations()

        
            def process_modified_by_relations():
                new_relations = []
                for rel in relations:
                    if rel[1] == "Modified_by":
                       
                        combined_entity = rel[2] + rel[0]
                       
                        sentences = re.split('[，。；]', origin_text)
                        should_keep = True
                        
                        for sentence in sentences:
                            if combined_entity in sentence:
                             
                                entity_index = sentence.find(combined_entity)
                                if entity_index != -1:
                                 
                                    text_before_entity = sentence[:entity_index]
                                    if '未见' in text_before_entity:
                                        should_keep = False
                                        break
                        
                        if should_keep:
                            new_relations.append(rel)
                    else:
                        new_relations.append(rel)
                return new_relations

            relations = process_modified_by_relations()

            def add_missing_located_at():
                entities_to_check = ['钙化', '肿块', '皮肤增厚', '结节', '结构不对称', '结构扭曲','乳腺增生','乳头凹陷']
                
                for entity in entities_to_check:
                    left_pos = breast_assessment['Left_breast']['Entities'][entity] == 'POS'
                    right_pos = breast_assessment['Right_breast']['Entities'][entity] == 'POS'
        
                    has_located_at = any(rel[0] == entity and rel[1] == "Located_at" for rel in relations)
                    
                    if not has_located_at:
                        
                        sentences = re.split('[。]', origin_text)
                        
                        for sentence in sentences:
                            if entity in sentence:
                               
                                entity_index = sentence.find(entity)
                                if entity_index != -1:
                                   
                                    text_before_entity = sentence[:entity_index]
                                    
                                    if left_pos and right_pos:
                                       
                                        if '双乳' in text_before_entity:
                                            relations.append([entity, "Located_at", "双乳"])
                                            break
                                    elif left_pos and not right_pos:
                                        
                                        if '左乳' in text_before_entity:
                                            relations.append([entity, "Located_at", "左乳"])
                                            break
                                    elif right_pos and not left_pos:
                                        
                                        if '右乳' in text_before_entity:
                                            relations.append([entity, "Located_at", "右乳"])
                                            break

            add_missing_located_at()

            def process_lymph_node():
                if (breast_assessment['Left_breast']['Entities']['淋巴结肿大'] == 'POS' or 
                    breast_assessment['Right_breast']['Entities']['淋巴结肿大'] == 'POS'):
                    has_located_at = any(rel[0] == '淋巴结肿大' and rel[1] == "Located_at" for rel in relations)
                    
                    if not has_located_at:
                       
                        sentences = re.split('[，。；]', origin_text)
                        
                        for sentence in sentences:
                            if '淋巴结肿大' in sentence:
                            
                                lymph_index = sentence.find('淋巴结肿大')
                                if lymph_index != -1:
                                   
                                    text_before_lymph = sentence[:lymph_index]
                                    
                                    if '双侧腋下' in text_before_lymph:
                                        relations.append(['淋巴结肿大', "Located_at", "双侧腋下"])
                                        break
                                    elif '左侧腋下' in text_before_lymph:
                                        relations.append(['淋巴结肿大', "Located_at", "左侧腋下"])
                                        break
                                    elif '右侧腋下' in text_before_lymph:
                                        relations.append(['淋巴结肿大', "Located_at", "右侧腋下"])
                                        break

            process_lymph_node()

            def process_suggestive_of_relations():
             
                if 'Impression:' in origin_text:
                    impression_part = origin_text.split('Impression:')[-1].strip()
           
                    new_relations = []
                    suggestive_relations_to_keep = set()
                    
                    for rel in relations:
                        if rel[1] == "Suggestive_of":
                            entity = rel[0]
                            birads = rel[2]
                            
                           
                            is_correct = False
                            sentences = re.split('[。；]', impression_part)
                            
                            for sentence in sentences:
                                if entity in sentence and birads in sentence:
                               
                                    birads_index = sentence.find(birads)
                                    if birads_index != -1:
                                        
                                        text_before_birads = sentence[:birads_index]
                                        if entity in text_before_birads and '未见' not in text_before_birads:
                                            is_correct = True
                                            suggestive_relations_to_keep.add((entity, birads))
                                            break
                            
                            if is_correct:
                                new_relations.append(rel)
                          
                        else:
                            new_relations.append(rel)
                    
                    relations.clear()
                    relations.extend(new_relations)
                    
              
                    sentences = re.split('[。；]', impression_part)
                    
                    
                    birads_patterns = [
                        r'Bi-Rads\s*[0-6][A-Ca-c]?',
                        r'BI-RADS\s*[0-6][A-Ca-c]?',
                        r'bi-rads\s*[0-6][A-Ca-c]?',
                        r'BiRads\s*[0-6][A-Ca-c]?',
                        r'BIRADS\s*[0-6][A-Ca-c]?',
                        r'birads\s*[0-6][A-Ca-c]?'
                    ]
                    
                    for sentence in sentences:
                       
                        birads_matches = []
                        for pattern in birads_patterns:
                            matches = re.findall(pattern, sentence, re.IGNORECASE)
                            birads_matches.extend(matches)
                        
                       
                        if not birads_matches:
                          
                            loose_matches = re.findall(r'(?i)(?:bi[-\s]?rads)\s*([0-6][a-c]?)', sentence)
                            birads_matches = [f"Bi-Rads {match}" for match in loose_matches]
                        
                        for birads_match in birads_matches:
                            
                            birads_clean = re.sub(r'\s+', ' ', birads_match).strip()
                            birads_clean = re.sub(r'(?i)bi[-\s]?rads', 'Bi-Rads', birads_clean, count=1)
                            
                        
                            birads_clean = birads_clean.upper()
                            
                            
                            birads_index = sentence.find(birads_match)
                            if birads_index == -1:
                               
                                birads_index = sentence.find(birads_clean)
                            
                            if birads_index != -1:
                                text_before_birads = sentence[:birads_index]
                                
    
                               
                                for entity in ENTITY_NAMES:
                                    if entity in text_before_birads:
                                       
                                        left_pos = breast_assessment['Left_breast']['Entities'].get(entity) == 'POS'
                                        right_pos = breast_assessment['Right_breast']['Entities'].get(entity) == 'POS'
                                        
                                        
                                        if (left_pos or right_pos) and '未见' not in text_before_birads:
                                          
                                            relation_exists = any(
                                                rel[0] == entity and 
                                                rel[1] == "Suggestive_of" and 
                                                rel[2] == birads_clean 
                                                for rel in relations
                                            )
                                            
                                            if not relation_exists and (entity, birads_clean) not in suggestive_relations_to_keep:

                                                relations.append([entity, "Suggestive_of", birads_clean])
                                                suggestive_relations_to_keep.add((entity, birads_clean))

            process_suggestive_of_relations()
            
            return process_samples({
                'Text': origin_text,
                'Relations': relations,
                'Breast_assessment': breast_assessment,
                'Triples': triples_dict,
                'Probs': probs
            })
    
    def test_all(self, preds, calculate_ci=False, n_bootstrap=1000):
        outputs = []
        all_metrics = []  
        all_relations_data = []  
        per_sample_f1 = []  
        density_to_idx = {d: i for i, d in enumerate(DENSITY_CLASSES)}
        birads_to_idx = {b: i for i, b in enumerate(TRUE_BI_RADS_CLASSES)}
        entity_state_mapping = {"POS": 0, "NEG": 1, "UNC": 2, "BLA": 3}
        all_true_density, all_pred_density = [], []
        all_true_birads, all_pred_birads = [], []
        

        total_should_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
       
        total_actual_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
        
  
        pos_true_labels = []  
        pos_pred_labels = []  
        

        sample_data_list = []
        
        for (id, pred) in tqdm(preds):
            pred_output = self.test(pred)
            ref_output = self.refs[id]
            
           
            sample_true_density = []
            sample_pred_density = []
            sample_true_birads = []
            sample_pred_birads = []
            
            sample_pos_true = []
            sample_pos_pred = []
            
            sample_should_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
            sample_actual_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
            
            true_d_left = ref_output['Breast_assessment']['Left_breast']['Density']
            pred_d_left = pred_output['Breast_assessment']['Left_breast']['Density']
            
            if true_d_left != 'BLA':
                total_should_evaluate['density'] += 1
                sample_should_evaluate['density'] += 1
                if pred_d_left != 'BLA':
                    total_actual_evaluate['density'] += 1
                    sample_actual_evaluate['density'] += 1
                    true_label = density_to_idx[true_d_left]
                    pred_label = density_to_idx.get(pred_d_left, len(DENSITY_CLASSES)-1)
                    all_true_density.append(true_label)
                    all_pred_density.append(pred_label)
                    sample_true_density.append(true_label)
                    sample_pred_density.append(pred_label)
            
            true_d_right = ref_output['Breast_assessment']['Right_breast']['Density']
            pred_d_right = pred_output['Breast_assessment']['Right_breast']['Density']
            
            if true_d_right != 'BLA':
                total_should_evaluate['density'] += 1
                sample_should_evaluate['density'] += 1
                if pred_d_right != 'BLA':
                    total_actual_evaluate['density'] += 1
                    sample_actual_evaluate['density'] += 1
                    true_label = density_to_idx[true_d_right]
                    pred_label = density_to_idx.get(pred_d_right, len(DENSITY_CLASSES)-1)
                    all_true_density.append(true_label)
                    all_pred_density.append(pred_label)
                    sample_true_density.append(true_label)
                    sample_pred_density.append(pred_label)

            true_b_left = ref_output['Breast_assessment']['Left_breast']['Bi-Rads']
            pred_b_left = pred_output['Breast_assessment']['Left_breast']['Bi-Rads']
            if true_b_left in ['Bi-Rads 1','Bi-Rads 2','Bi-Rads 3'] and pred_b_left in ['Bi-Rads 1','Bi-Rads 2']:
                pred_b_left=true_b_left
            elif true_b_left in ['Bi-Rads 4'] and pred_b_left in ['Bi-Rads 3','Bi-Rads 4A']:
                pred_b_left=true_b_left
            elif true_b_left in ['Bi-Rads 5'] and pred_b_left in ['Bi-Rads 4B','Bi-Rads 4C','Bi-Rads 5']:
                pred_b_left=true_b_left
            if true_b_left != 'BLA':
                total_should_evaluate['birads'] += 1
                sample_should_evaluate['birads'] += 1
                if pred_b_left != 'BLA':
                    total_actual_evaluate['birads'] += 1
                    sample_actual_evaluate['birads'] += 1
                    true_label = birads_to_idx[true_b_left]
                    pred_label = birads_to_idx.get(pred_b_left, len(TRUE_BI_RADS_CLASSES)-1)
                    all_true_birads.append(true_label)
                    all_pred_birads.append(pred_label)
                    sample_true_birads.append(true_label)
                    sample_pred_birads.append(pred_label)
            
            true_b_right = ref_output['Breast_assessment']['Right_breast']['Bi-Rads']
            pred_b_right = pred_output['Breast_assessment']['Right_breast']['Bi-Rads']
            if true_b_left in ['Bi-Rads 1','Bi-Rads 2','Bi-Rads 3'] and pred_b_left in ['Bi-Rads 1','Bi-Rads 2']:
                pred_b_left=true_b_left
            elif true_b_left in ['Bi-Rads 4'] and pred_b_left in ['Bi-Rads 3','Bi-Rads 4A']:
                pred_b_left=true_b_left
            elif true_b_left in ['Bi-Rads 5'] and pred_b_left in ['Bi-Rads 4B','Bi-Rads 4C','Bi-Rads 5']:
                pred_b_left=true_b_left
            if true_b_right != 'BLA':
                total_should_evaluate['birads'] += 1
                sample_should_evaluate['birads'] += 1
                if pred_b_right != 'BLA':
                    total_actual_evaluate['birads'] += 1
                    sample_actual_evaluate['birads'] += 1
                    true_label = birads_to_idx[true_b_right]
                    pred_label = birads_to_idx.get(pred_b_right, len(TRUE_BI_RADS_CLASSES)-1)
                    all_true_birads.append(true_label)
                    all_pred_birads.append(pred_label)
                    sample_true_birads.append(true_label)
                    sample_pred_birads.append(pred_label)
            
            true_left = ref_output['Breast_assessment']['Left_breast']['Entities']
            pred_left = pred_output['Breast_assessment']['Left_breast']['Entities']
            true_right = ref_output['Breast_assessment']['Right_breast']['Entities']
            pred_right = pred_output['Breast_assessment']['Right_breast']['Entities']
            
            for entity in ENTITY_NAMES:
          
                true_state = true_left.get(entity, "BLA")
                pred_state = pred_left.get(entity, "BLA")
                
               
                if true_state in ["POS", "NEG"]:
                    sample_should_evaluate['entities'] += 1
                    total_should_evaluate['entities'] += 1
                 
                    true_label = 1 if true_state == "POS" else 0
                    
                
                    if pred_state == "POS":
                        pred_label = 1  
                        sample_actual_evaluate['entities'] += 1
                        total_actual_evaluate['entities'] += 1
                    elif pred_state == "NEG":
                        pred_label = 0  
                        sample_actual_evaluate['entities'] += 1
                        total_actual_evaluate['entities'] += 1
                    else:  
                        pred_label = 0  
                    
                    pos_true_labels.append(true_label)
                    pos_pred_labels.append(pred_label)
                    sample_pos_true.append(true_label)
                    sample_pos_pred.append(pred_label)

                true_state = true_right.get(entity, "BLA")
                pred_state = pred_right.get(entity, "BLA")
                
                if true_state in ["POS", "NEG"]:
                    sample_should_evaluate['entities'] += 1
                    total_should_evaluate['entities'] += 1
                    true_label = 1 if true_state == "POS" else 0
                    
                    if pred_state == "POS":
                        pred_label = 1
                        sample_actual_evaluate['entities'] += 1
                        total_actual_evaluate['entities'] += 1
                    elif pred_state == "NEG":
                        pred_label = 0
                        sample_actual_evaluate['entities'] += 1
                        total_actual_evaluate['entities'] += 1
                    else:  
                        pred_label = 0
                    
                    pos_true_labels.append(true_label)
                    pos_pred_labels.append(pred_label)
                    sample_pos_true.append(true_label)
                    sample_pos_pred.append(pred_label)
            

            density_f1 = None
            if len(sample_true_density) > 0:
                raw_f1 = f1_score(sample_true_density, sample_pred_density, average='macro')
                sample_should = sample_should_evaluate['density']
                sample_actual = len(sample_true_density)
                completeness = sample_actual / sample_should if sample_should > 0 else 0
                density_f1 = raw_f1 * completeness

            birads_f1 = None
            if len(sample_true_birads) > 0:
                raw_f1 = f1_score(sample_true_birads, sample_pred_birads, average='macro')
                sample_should = sample_should_evaluate['birads']
                sample_actual = len(sample_true_birads)
                completeness = sample_actual / sample_should if sample_should > 0 else 0
                birads_f1 = raw_f1 * completeness

            entities_f1 = None
            if len(sample_pos_true) > 0:
                pos_precision = precision_score(sample_pos_true, sample_pos_pred, zero_division=0)
                pos_recall = recall_score(sample_pos_true, sample_pos_pred, zero_division=0)
                if pos_precision + pos_recall > 0:
                    entities_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
                else:
                    entities_f1 = 0.0
            
            per_sample_f1.append({
                'composition_f1': density_f1,
                'birads_f1': birads_f1,
                'finding_f1': entities_f1
            })
            
            sample_data = {
                'true_density': sample_true_density,
                'pred_density': sample_pred_density,
                'true_birads': sample_true_birads,
                'pred_birads': sample_pred_birads,
                'true_entities': sample_pos_true,
                'pred_entities': sample_pos_pred,
                'should_evaluate': sample_should_evaluate,
                'actual_evaluate': sample_actual_evaluate
            }
            sample_data_list.append(sample_data)
            
            outputs.append({
                'Pred_text': pred_output['Text'],
                'Ref_breast_assessment': ref_output['Breast_assessment'],
                'Pred_breast_assessment': pred_output['Breast_assessment'],
            })
        
        metrics = {}
        
        if all_true_density:
            raw_f1 = f1_score(all_true_density, all_pred_density, average='macro')
            completeness = total_actual_evaluate['density'] / total_should_evaluate['density'] if total_should_evaluate['density'] > 0 else 0
            metrics['density'] = raw_f1 * completeness
        else:
            metrics['density'] = -1
        
        if all_true_birads:
            raw_f1 = f1_score(all_true_birads, all_pred_birads, average='macro')
            completeness = total_actual_evaluate['birads'] / total_should_evaluate['birads'] if total_should_evaluate['birads'] > 0 else 0
            metrics['bi_rads'] = raw_f1 * completeness
        else:
            metrics['bi_rads'] = -1
        
        if pos_true_labels:
            pos_precision = precision_score(pos_true_labels, pos_pred_labels, zero_division=0)
            pos_recall = recall_score(pos_true_labels, pos_pred_labels, zero_division=0)
            if pos_precision + pos_recall > 0:
                metrics['entities'] = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
            else:
                metrics['entities'] = 0.0
        else:
            metrics['entities'] = -1

        if calculate_ci:
            bootstrapped_density = [] 
            bootstrapped_birads = []
            bootstrapped_entities = []

            n_samples = len(sample_data_list)

            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                resampled_true_density = []
                resampled_pred_density = []
                resampled_true_birads = []
                resampled_pred_birads = []
                resampled_true_entities = []
                resampled_pred_entities = []
                
                resampled_should_density = 0
                resampled_actual_density = 0
                resampled_should_birads = 0
                resampled_actual_birads = 0
                resampled_should_entities = 0
                resampled_actual_entities = 0
                
                for idx in indices:
                    sample_data = sample_data_list[idx]
                    resampled_true_density.extend(sample_data['true_density'])
                    resampled_pred_density.extend(sample_data['pred_density'])
                    resampled_true_birads.extend(sample_data['true_birads'])
                    resampled_pred_birads.extend(sample_data['pred_birads'])
                    resampled_true_entities.extend(sample_data['true_entities'])
                    resampled_pred_entities.extend(sample_data['pred_entities'])
                    
                    resampled_should_density += sample_data['should_evaluate']['density']
                    resampled_actual_density += sample_data['actual_evaluate']['density']
                    resampled_should_birads += sample_data['should_evaluate']['birads']
                    resampled_actual_birads += sample_data['actual_evaluate']['birads']
                    resampled_should_entities += sample_data['should_evaluate']['entities']
                    resampled_actual_entities += sample_data['actual_evaluate']['entities']

                if resampled_true_density:
                    raw_f1 = f1_score(resampled_true_density, resampled_pred_density, average='macro')
                    completeness = resampled_actual_density / resampled_should_density if resampled_should_density > 0 else 0
                    bootstrapped_density.append(raw_f1 * completeness)
                
                if resampled_true_birads:
                    raw_f1 = f1_score(resampled_true_birads, resampled_pred_birads, average='macro')
                    completeness = resampled_actual_birads / resampled_should_birads if resampled_should_birads > 0 else 0
                    bootstrapped_birads.append(raw_f1 * completeness)
                
                if resampled_true_entities:
                    pos_precision = precision_score(resampled_true_entities, resampled_pred_entities, zero_division=0)
                    pos_recall = recall_score(resampled_true_entities, resampled_pred_entities, zero_division=0)
                    if pos_precision + pos_recall > 0:
                        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall)
                    else:
                        pos_f1 = 0.0
                    bootstrapped_entities.append(pos_f1)

            if bootstrapped_density:
                metrics['density_ci'] = (
                    np.percentile(bootstrapped_density, 2.5),
                    np.percentile(bootstrapped_density, 97.5)
                )
            if bootstrapped_birads:
                metrics['bi_rads_ci'] = (
                    np.percentile(bootstrapped_birads, 2.5),
                    np.percentile(bootstrapped_birads, 97.5)
                )
            if bootstrapped_entities:
                metrics['entities_ci'] = (
                    np.percentile(bootstrapped_entities, 2.5),
                    np.percentile(bootstrapped_entities, 97.5)
                )

        status_metrics = {
            'composition_f1': metrics.get('density', -1),
            'birads_f1': metrics.get('bi_rads', -1),
            'finding_f1': metrics.get('entities', -1), 
        }
        
        if calculate_ci:
            status_metrics.update({
                'composition_f1_ci': metrics.get('density_ci', (None, None)),
                'birads_f1_ci': metrics.get('bi_rads_ci', (None, None)),
                'finding_f1_ci': metrics.get('entities_ci', (None, None)),
            })
        
        density_cm = None
        if all_true_density:
            density_cm = confusion_matrix(
                all_true_density,
                all_pred_density,
                labels=list(range(len(DENSITY_CLASSES)))
            )
            
        birads_cm = None
        if all_true_birads:
            birads_cm = confusion_matrix(
                all_true_birads,
                all_pred_birads,
                labels=list(range(len(BI_RADS_CLASSES)))
            )
            
        entities_cm = None
        if pos_true_labels:
            entities_cm = confusion_matrix(
                pos_true_labels,
                pos_pred_labels,
                labels=[0, 1] 
            )

        if self.output_dir:
            with open(self.output_dir, 'w', encoding='utf-8') as fw:
                json.dump(outputs, fw, ensure_ascii=False, indent=4)
                
        entity_pos_stats = {
            "total_true_pos": sum(1 for t in pos_true_labels if t == 1),  
            "total_true_neg": sum(1 for t in pos_true_labels if t == 0),  
            "total_pred_pos": sum(pos_pred_labels), 
            "tp_pos": sum(1 for t, p in zip(pos_true_labels, pos_pred_labels) if t == 1 and p == 1),
            "fp_pos": sum(1 for t, p in zip(pos_true_labels, pos_pred_labels) if t == 0 and p == 1),
            "fn_pos": sum(1 for t, p in zip(pos_true_labels, pos_pred_labels) if t == 1 and p == 0),
            "tn_pos": sum(1 for t, p in zip(pos_true_labels, pos_pred_labels) if t == 0 and p == 0),
        }

        precision_pos = entity_pos_stats["tp_pos"] / max(entity_pos_stats["total_pred_pos"], 1)
        recall_pos = entity_pos_stats["tp_pos"] / max(entity_pos_stats["total_true_pos"], 1)
        if precision_pos + recall_pos > 0:
            f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
        else:
            f1_pos = 0.0

        return {
            'Status_metrics': status_metrics,
            'Per_sample_f1': per_sample_f1,
        }

    def get_output(self, preds, calculate_ci=False):
        results = self.test_all(preds, calculate_ci=calculate_ci)

        return results
    
