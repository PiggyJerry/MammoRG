import os
import pandas as pd
import re
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import json
import time
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Union
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from correction import process_samples
from models.rel_model import RelModel
from transformers import BertTokenizer
from utils.tokenization import BasicTokenizer
from config import Config
from clean_text import clean_text

from scipy import stats  # Added for confidence interval calculation
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

DENSITY_CLASSES = ["脂肪型", "纤维腺体型", "不均匀致密型", "致密型", "BLA"]
BI_RADS_CLASSES = [
    "BI-RADS 0", "BI-RADS 1", "BI-RADS 2", "BI-RADS 3",
    "BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C",
    "BI-RADS 5", "BLA"
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
    if pd.isna(conclusion) or conclusion is None:
        return None
    conclusion = str(conclusion)

    # 1) 只用 Impression 段，避免 Findings 的“左乳/右乳/双乳”干扰侧别关联
    m = re.search(r'Impression\s*[:：]', conclusion, flags=re.IGNORECASE)
    text = conclusion[m.end():] if m else conclusion

    # 2) 更宽松的 BI-RADS 识别：BI-RADS分类：3 / BI RADS 3 / BI/RADS 3 / BI-RADS 4A / BIRADS 2 / 分类：2
    # birads_pattern = re.compile(
    #     r'(?:'
    #     r'BI\s*[-/\s]?\s*RADS'      # BI-RADS / BI RADS / BI/RADS
    #     r'|BIRADS'                  # BIRADS
    #     r'|Bi\s*[-/\s]?\s*Rads'     # BI-RADS / Bi Rads
    #     r'|分类'
    #     r')'
    #     r'(?:\s*分类)?\s*[:：]?\s*'
    #     r'([IVXivxⅠⅡⅢⅣⅤⅥⅰⅱⅲⅳⅴⅵ]+|\d+(?:[a-cA-C])?)',
    #     re.IGNORECASE
    # )
    birads_pattern = re.compile(
        r'(?:BI\s*[-/\s]?\s*RADS|BIRADS|Bi\s*[-/\s]?\s*Rads|分类)'
        r'(?:\s*分类)?\s*[:：]?\s*'
        r'(0|[1-6](?:[A-Ca-c])?|[IVXivxⅠⅡⅢⅣⅤⅥⅰⅱⅲⅳⅴⅵ]+)',
        re.IGNORECASE
    )

    side_keywords = {
        'left':   ['左乳', '左'],
        'right':  ['右乳', '右'],
        'double': ['双侧乳', '双乳', '双侧'],
    }

    # 3) 找 text 内所有侧别位置（长词优先）
    side_positions = []
    for side_type, kws in side_keywords.items():
        for kw in sorted(kws, key=len, reverse=True):
            for sm in re.finditer(re.escape(kw), text):
                side_positions.append({'pos': sm.start(), 'type': side_type})
    side_positions.sort(key=lambda x: x['pos'])

    # 4) 找所有 BI-RADS，并做标准化 + 打分
    birads_matches = []
    for bm in birads_pattern.finditer(text):
        raw = bm.group(1).upper()

        if raw in roman_to_arabic:
            norm_num = str(roman_to_arabic[raw])
        elif re.match(r'^\d+[A-C]$', raw):
            norm_num = raw.lower()   # 4A -> 4a 用于 priority
        else:
            norm_num = raw

        birads_matches.append({
            'norm_num': norm_num,  # 用于优先级
            'priority': birads_priority.get(norm_num, -1),
            'start': bm.start(),
        })

    if not birads_matches:
        return None

    # 5) 侧别关联：只在 Impression 段里做“前面最近侧别”
    birads_with_sides = []
    for b in birads_matches:
        preceding = [s for s in side_positions if s['pos'] < b['start']]
        nearest = max(preceding, key=lambda x: x['pos']) if preceding else None
        birads_with_sides.append({
            'norm_num': b['norm_num'],
            'priority': b['priority'],
            'side_type': nearest['type'] if nearest else None
        })

    # 6) laterality 过滤（优先单侧，其次 double）
    if laterality == 'L':
        valid = (
            [x for x in birads_with_sides if x['side_type'] == 'left'] +
            [x for x in birads_with_sides if x['side_type'] == 'double']
        )
    elif laterality == 'R':
        valid = (
            [x for x in birads_with_sides if x['side_type'] == 'right'] +
            [x for x in birads_with_sides if x['side_type'] == 'double']
        )
    else:
        return None

    if not valid:
        return None

    best = max(valid, key=lambda x: x['priority'])

    # 7) 返回标准类名（必须匹配你的 BI_RADS_CLASSES）
    # norm_num 可能是 '4a' -> 输出 '4A'
    num = best['norm_num']
    if re.match(r'^\d+[a-c]$', num):
        num_out = num[:-1] + num[-1].upper()
    else:
        num_out = num

    return f"BI-RADS {num_out}"

def merge_birads_sentences(sentences):
    merged = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        has_birads = bool(
            re.search(r"BI\s*[-/\s]?\s*RADS", s, re.IGNORECASE)
        )
        has_entity = any(entity in s for entity in ENTITY_NAMES)

        # 如果当前句已经明确写了左乳、右乳或双乳，它本身就是一条完整的
        # 乳房级评估，不能因为没有异常实体而并入上一条病灶描述。
        has_laterality = bool(
            re.search(r"左乳|右乳|双乳|左侧乳|右侧乳|双侧乳", s)
        )

        # 只合并真正独立的“BI-RADS 3”式补充句。像
        # “左乳符合 BI-RADS 3，右乳符合 BI-RADS 2”这样的完整评估句
        # 必须保留为独立句，否则会把上一句的所有病灶与两个 BI-RADS
        # 交叉配对。
        standalone_birads = bool(
            re.fullmatch(
                r"\s*[（(]?\s*"
                r"BI\s*[-/\s]?\s*RADS\s*[:：]?\s*[0-6](?:[A-Ca-c])?\s*[类级]?"
                r"(?:\s*[，,、]\s*(?:建议)?(?:短期)?(?:随访|复查).*)?"
                r"\s*[）)]?\s*",
                s,
                re.IGNORECASE,
            )
        )

        if (
            has_birads
            and not has_entity
            and not has_laterality
            and standalone_birads
            and merged
        ):
            merged[-1] += "。" + s
        else:
            merged.append(s)

    return merged

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
        self.tokenizer = BertTokenizer.from_pretrained('/home/jiayi/MammoRG/MammoRGTool/pre_trained_bert/vocab.txt')
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
            "BI-RADS": bi_rads,
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
        left_breast['BI-RADS'] = rule_based_birads
    
    right_breast=_decode_breast(right_density_idx, right_birads_idx, right_state_indices)    
    rule_based_density = rule_based_get_density(text, 'R')
    if rule_based_density and rule_based_density != 'BLA':
        right_breast['Density'] = rule_based_density
    rule_based_birads = rule_based_get_birads(text, 'R')
    if rule_based_birads and rule_based_birads != 'BLA':
        right_breast['BI-RADS'] = rule_based_birads
            

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

    def reset_unmentioned_positive_entities():
        """
        如果正文中完全没有出现指定异常名称，但模型将其预测为 POS，
        则将该侧状态改为 BLA。已经被规则修正为 NEG 的状态不会受影响。
        """
        entities_requiring_explicit_mention = [
            '淋巴结肿大',
            '乳头凹陷',
            '结构扭曲',
            '悬韧带增粗',
            '皮肤增厚',
            '结构不对称',
        ]

        for entity in entities_requiring_explicit_mention:
            if entity in text:
                continue

            for breast in (left_breast, right_breast):
                if breast['Entities'].get(entity) == 'POS':
                    breast['Entities'][entity] = 'BLA'

    # 放在所有实体规则之后执行，清除模型对未提及异常的 POS 误报。
    reset_unmentioned_positive_entities()
    
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


def set_level_f1(ref_set, pred_set) -> float:
    """
    Symmetric set-level F1 / Dice similarity:

        F1 = 2 * |ref ∩ pred| / (|ref| + |pred|)

    Empty-set convention:
      - both empty -> 1.0
      - only one empty -> 0.0
    """
    ref_set = set(ref_set)
    pred_set = set(pred_set)

    if not ref_set and not pred_set:
        return 1.0

    return float(
        2.0 * len(ref_set & pred_set) /
        (len(ref_set) + len(pred_set))
    )


def compute_side_aware_finding_set_f1(
    true_left: Dict[str, str],
    pred_left: Dict[str, str],
    true_right: Dict[str, str],
    pred_right: Dict[str, str],
    entity_names: List[str] = None,
) -> float:
    """
    POS-only, side-aware set-level F1 for Findings.

    Each set element is (side, finding), so:
        ('L', '肿块') != ('R', '肿块')

    NEG / UNC / BLA do not enter the sets.

        S_findings = 2 * |F_ref ∩ F_pred| / (|F_ref| + |F_pred|)

    If both reports contain no POS findings, score = 1.0.
    """
    if entity_names is None:
        entity_names = ENTITY_NAMES

    ref_findings = set()
    pred_findings = set()

    for side, true_entities, pred_entities in (
        ("L", true_left, pred_left),
        ("R", true_right, pred_right),
    ):
        for entity in entity_names:
            if true_entities.get(entity, "BLA") == "POS":
                ref_findings.add((side, entity))
            if pred_entities.get(entity, "BLA") == "POS":
                pred_findings.add((side, entity))

    return set_level_f1(ref_findings, pred_findings)


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

                    # “；/。”以及编号项是硬边界。只有真正独立的
                    # “BI-RADS 3”补充句才允许由 merge_birads_sentences
                    # 合并回上一条病灶描述。
                    sentences = re.split(r'[。；;\n]+', impression_part)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    sentences = merge_birads_sentences(sentences)

                    birads_pattern = re.compile(
                        r'BI\s*[-/\s]?\s*RADS\s*[:：]?\s*([0-6](?:[A-Ca-c])?)',
                        re.IGNORECASE,
                    )

                    def nearest_side(text):
                        """返回文本中最后出现的乳房侧别。"""
                        side_hits = []
                        for side, pattern in (
                            ('double', r'双侧乳|双乳|双侧'),
                            ('left', r'左侧乳|左乳'),
                            ('right', r'右侧乳|右乳'),
                        ):
                            for match in re.finditer(pattern, text):
                                side_hits.append((match.start(), side))
                        return max(side_hits)[1] if side_hits else None

                    def entity_is_positive(entity, side):
                        left_pos = (
                            breast_assessment['Left_breast']['Entities'].get(entity)
                            == 'POS'
                        )
                        right_pos = (
                            breast_assessment['Right_breast']['Entities'].get(entity)
                            == 'POS'
                        )
                        if side == 'left':
                            return left_pos
                        if side == 'right':
                            return right_pos
                        if side == 'double':
                            return left_pos or right_pos
                        return left_pos or right_pos

                    # 丢弃模型直接给出的 Suggestive_of，只保留下面能够由
                    # Impression 局部文本明确验证/重建的关系。
                    relations[:] = [
                        rel for rel in relations if rel[1] != "Suggestive_of"
                    ]
                    extracted_suggestive = set()

                    for sentence in sentences:
                        matches = list(birads_pattern.finditer(sentence))
                        previous_birads_end = 0

                        for match in matches:
                            # 第二个及后续 BI-RADS 只能查看前一个 BI-RADS
                            # 之后的局部文本，避免“左结节→右乳 BI-RADS”的
                            # 跨侧、跨病灶配对。
                            local_context = sentence[
                                previous_birads_end:match.start()
                            ]
                            previous_birads_end = match.end()

                            birads_clean = f"BI-RADS{match.group(1).upper()}"
                            birads_side = nearest_side(local_context)

                            for entity in ENTITY_NAMES:
                                entity_index = local_context.rfind(entity)
                                if entity_index == -1:
                                    continue

                                # 只检查距离该实体最近的局部描述，防止更早的
                                # “未见”错误否定后面的阳性实体。
                                prefix_start = max(
                                    local_context.rfind('，', 0, entity_index),
                                    local_context.rfind(',', 0, entity_index),
                                    local_context.rfind('：', 0, entity_index),
                                    local_context.rfind(':', 0, entity_index),
                                )
                                entity_prefix = local_context[
                                    prefix_start + 1:entity_index
                                ]
                                if '未见' in entity_prefix:
                                    continue

                                entity_side = nearest_side(
                                    local_context[:entity_index + len(entity)]
                                )
                                if (
                                    entity_side in {'left', 'right'}
                                    and birads_side in {'left', 'right'}
                                    and entity_side != birads_side
                                ):
                                    continue

                                side_for_state = entity_side or birads_side
                                if not entity_is_positive(entity, side_for_state):
                                    continue

                                extracted_suggestive.add(
                                    (entity, "Suggestive_of", birads_clean)
                                )

                    relations.extend([list(rel) for rel in extracted_suggestive])

            process_suggestive_of_relations()
            relations = list(set(tuple(r) for r in relations))
            relations = [list(r) for r in relations]
            triples_dict = set(tuple(r) for r in relations)
            return process_samples({
                'Text': origin_text,
                'Relations': relations,
                'Breast_assessment': breast_assessment,
                'Triples': triples_dict,
                'Probs': probs
            })
    
    def test_all(
        self,
        preds,
        refs,
        calculate_ci=False,
        n_bootstrap=1000,
        bootstrap_seed=42,
        bootstrap_indices=None,
        return_bootstrap=False
    ):
        outputs = []
        all_metrics = [] 
        all_relations_data = [] 
        per_sample_f1 = []
        density_to_idx = {d: i for i, d in enumerate(DENSITY_CLASSES)}
        birads_to_idx = {b: i for i, b in enumerate(BI_RADS_CLASSES)}
        entity_state_mapping = {"POS": 0, "NEG": 1, "UNC": 2, "BLA": 3}
        all_true_density, all_pred_density = [], []
        all_true_birads, all_pred_birads = [], []

        # Findings are evaluated per report using a POS-only, side-aware
        # set-level F1, then averaged across reports.
        
        total_should_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
        total_actual_evaluate = {'density': 0, 'birads': 0, 'entities': 0}
        sample_data_list = []
        
        for pred, ref in tqdm(zip(preds, refs)):
            pred_output = self.test(pred)
            ref_output = self.test(ref)
            
            relation_correct = len(pred_output['Triples'] & ref_output['Triples'])
            pred_count = len(pred_output['Triples'])
            gold_count = len(ref_output['Triples'])

            # Relation-set F1. If both reports have no relations, score = 1.0.
            relations_f1 = set_level_f1(
                ref_output['Triples'],
                pred_output['Triples'],
            )
            all_relations_data.append(
                (relation_correct, pred_count, gold_count, relations_f1)
            )
        
            sample_true_density = []
            sample_pred_density = []
            sample_true_birads = []
            sample_pred_birads = []
            sample_true_entities = {entity: [] for entity in ENTITY_NAMES}
            sample_pred_entities = {entity: [] for entity in ENTITY_NAMES}
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

            true_b_left = ref_output['Breast_assessment']['Left_breast']['BI-RADS']
            pred_b_left = pred_output['Breast_assessment']['Left_breast']['BI-RADS']

            if true_b_left != 'BLA':
                total_should_evaluate['birads'] += 1
                sample_should_evaluate['birads'] += 1
                if pred_b_left != 'BLA':
                    total_actual_evaluate['birads'] += 1
                    sample_actual_evaluate['birads'] += 1
                    true_label = birads_to_idx[true_b_left]
                    pred_label = birads_to_idx.get(pred_b_left, len(BI_RADS_CLASSES)-1)
                    all_true_birads.append(true_label)
                    all_pred_birads.append(pred_label)
                    sample_true_birads.append(true_label)
                    sample_pred_birads.append(pred_label)
            
            true_b_right = ref_output['Breast_assessment']['Right_breast']['BI-RADS']
            pred_b_right = pred_output['Breast_assessment']['Right_breast']['BI-RADS']
 
            if true_b_right != 'BLA':
                total_should_evaluate['birads'] += 1
                sample_should_evaluate['birads'] += 1
                if pred_b_right != 'BLA':
                    total_actual_evaluate['birads'] += 1
                    sample_actual_evaluate['birads'] += 1
                    true_label = birads_to_idx[true_b_right]
                    pred_label = birads_to_idx.get(pred_b_right, len(BI_RADS_CLASSES)-1)
                    all_true_birads.append(true_label)
                    all_pred_birads.append(pred_label)
                    sample_true_birads.append(true_label)
                    sample_pred_birads.append(pred_label)

            true_left = ref_output['Breast_assessment']['Left_breast']['Entities']
            pred_left = pred_output['Breast_assessment']['Left_breast']['Entities']
            true_right = ref_output['Breast_assessment']['Right_breast']['Entities']
            pred_right = pred_output['Breast_assessment']['Right_breast']['Entities']

            # POS-only, side-aware Finding representation.
            # (L, finding) and (R, finding) are distinct set elements.
            for entity in ENTITY_NAMES:
                for true_entities, pred_entities in (
                    (true_left, pred_left),
                    (true_right, pred_right),
                ):
                    true_state = true_entities.get(entity, "BLA")
                    pred_state = pred_entities.get(entity, "BLA")

                    true_label = 1 if true_state == "POS" else 0
                    pred_label = 1 if pred_state == "POS" else 0

                    sample_true_entities[entity].append(true_label)
                    sample_pred_entities[entity].append(pred_label)

                    # Retained for bookkeeping only.
                    if true_label == 1 or pred_label == 1:
                        sample_should_evaluate['entities'] += 1
                        total_should_evaluate['entities'] += 1
                        sample_actual_evaluate['entities'] += 1
                        total_actual_evaluate['entities'] += 1

            density_f1 = None
            if len(sample_true_density) > 0:
                correct = sum(
                    1 for t,p in zip(sample_true_density, sample_pred_density)
                    if t == p
                )

                density_f1 = correct / len(sample_true_density)

            birads_f1 = None
            if len(sample_true_birads) > 0:
                correct = sum(
                    1 for t,p in zip(sample_true_birads, sample_pred_birads)
                    if t == p
                )

                birads_f1 = correct / len(sample_true_birads)

            entities_f1 = compute_side_aware_finding_set_f1(
                true_left=true_left,
                pred_left=pred_left,
                true_right=true_right,
                pred_right=pred_right,
            )
            
            per_sample_f1.append({
                'relation_f1': relations_f1,
                'composition_f1': density_f1,
                'birads_f1': birads_f1,
                'finding_f1': entities_f1
            })
            
            sample_data = {
                'true_density': sample_true_density,
                'pred_density': sample_pred_density,
                'true_birads': sample_true_birads,
                'pred_birads': sample_pred_birads,
                'true_entities': sample_true_entities,
                'pred_entities': sample_pred_entities,
                'relations_data': (relation_correct, pred_count, gold_count),
                'finding_f1': entities_f1,
                'relation_f1': relations_f1,
                'should_evaluate': sample_should_evaluate,
                'actual_evaluate': sample_actual_evaluate
            }
            sample_data_list.append(sample_data)
            
            outputs.append({
                'Ref_text': ref_output['Text'],
                'Pred_text': pred_output['Text'],
                'Ref_breast_assessment': ref_output['Breast_assessment'],
                'Pred_breast_assessment': pred_output['Breast_assessment'],
                'Ref_relations': ref_output['Relations'],
                'Pred_relations': pred_output['Relations'],
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
        
        # Mean report-level POS-only, side-aware Finding set-F1.
        finding_scores = [x['finding_f1'] for x in per_sample_f1]
        metrics['entities'] = (
            float(np.mean(finding_scores))
            if finding_scores
            else -1
        )

        # Mean report-level Relation set-F1.
        # Empty-vs-empty relation sets contribute 1.0.
        relation_scores = [x['relation_f1'] for x in per_sample_f1]
        f1 = (
            float(np.mean(relation_scores))
            if relation_scores
            else -1
        )

        if calculate_ci or return_bootstrap:
            n_samples = len(sample_data_list)

            if bootstrap_indices is None:
                rng = np.random.default_rng(bootstrap_seed)
                bootstrap_indices = rng.integers(
                    low=0,
                    high=n_samples,
                    size=(n_bootstrap, n_samples),
                )
            else:
                bootstrap_indices = np.asarray(
                    bootstrap_indices,
                    dtype=np.int64,
                )

                if bootstrap_indices.ndim != 2:
                    raise ValueError(
                        "bootstrap_indices must have shape "
                        "(n_bootstrap, n_samples)."
                    )

                if bootstrap_indices.shape[1] != n_samples:
                    raise ValueError(
                        "bootstrap_indices sample dimension does not "
                        f"match data: {bootstrap_indices.shape[1]} vs "
                        f"{n_samples}."
                    )

                n_bootstrap = bootstrap_indices.shape[0]

            # IMPORTANT: preserve bootstrap_id positions. The previous code
            # appended only valid density/BI-RADS/finding replicates. If one
            # model skipped bootstrap_id=17 and another did not, every later
            # array element became misaligned despite using the same seed.
            bootstrapped_density = np.full(n_bootstrap, np.nan, dtype=np.float64)
            bootstrapped_birads = np.full(n_bootstrap, np.nan, dtype=np.float64)
            bootstrapped_entities = np.full(n_bootstrap, np.nan, dtype=np.float64)
            bootstrapped_relations_f1 = np.full(n_bootstrap, np.nan, dtype=np.float64)

            for bootstrap_id in range(n_bootstrap):
                indices = bootstrap_indices[bootstrap_id]

                resampled_true_density = []
                resampled_pred_density = []
                resampled_true_birads = []
                resampled_pred_birads = []
                resampled_finding_scores = []
                resampled_relation_scores = []

                resampled_should_density = 0
                resampled_actual_density = 0
                resampled_should_birads = 0
                resampled_actual_birads = 0
                resampled_should_entities = 0
                resampled_actual_entities = 0

                for idx in indices:
                    sample_data = sample_data_list[int(idx)]
                    resampled_true_density.extend(sample_data['true_density'])
                    resampled_pred_density.extend(sample_data['pred_density'])
                    resampled_true_birads.extend(sample_data['true_birads'])
                    resampled_pred_birads.extend(sample_data['pred_birads'])

                    resampled_finding_scores.append(
                        sample_data['finding_f1']
                    )
                    resampled_relation_scores.append(
                        sample_data['relation_f1']
                    )

                    resampled_should_density += sample_data['should_evaluate']['density']
                    resampled_actual_density += sample_data['actual_evaluate']['density']
                    resampled_should_birads += sample_data['should_evaluate']['birads']
                    resampled_actual_birads += sample_data['actual_evaluate']['birads']
                    resampled_should_entities += sample_data['should_evaluate']['entities']
                    resampled_actual_entities += sample_data['actual_evaluate']['entities']

                if resampled_true_density:
                    raw_f1 = f1_score(
                        resampled_true_density,
                        resampled_pred_density,
                        average='macro',
                        zero_division=0,
                    )
                    completeness = (
                        resampled_actual_density / resampled_should_density
                        if resampled_should_density > 0
                        else 0.0
                    )
                    bootstrapped_density[bootstrap_id] = raw_f1 * completeness

                if resampled_true_birads:
                    raw_f1 = f1_score(
                        resampled_true_birads,
                        resampled_pred_birads,
                        average='macro',
                        zero_division=0,
                    )
                    completeness = (
                        resampled_actual_birads / resampled_should_birads
                        if resampled_should_birads > 0
                        else 0.0
                    )
                    bootstrapped_birads[bootstrap_id] = raw_f1 * completeness

                if resampled_finding_scores:
                    bootstrapped_entities[bootstrap_id] = float(
                        np.mean(resampled_finding_scores)
                    )

                if resampled_relation_scores:
                    bootstrapped_relations_f1[bootstrap_id] = float(
                        np.mean(resampled_relation_scores)
                    )

            def finite_ci(values):
                values = np.asarray(values, dtype=np.float64)
                values = values[np.isfinite(values)]
                if len(values) == 0:
                    return (None, None)
                return (
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                )

            if calculate_ci:
                density_ci = finite_ci(bootstrapped_density)
                birads_ci = finite_ci(bootstrapped_birads)
                entities_ci = finite_ci(bootstrapped_entities)
                relations_ci = finite_ci(bootstrapped_relations_f1)

                if density_ci[0] is not None:
                    metrics['density_ci'] = density_ci
                if birads_ci[0] is not None:
                    metrics['bi_rads_ci'] = birads_ci
                if entities_ci[0] is not None:
                    metrics['entities_ci'] = entities_ci

            relations_metrics = {'f1': f1}
            if calculate_ci:
                relations_metrics['f1_ci'] = relations_ci
        else:
            relations_metrics = {'f1': f1}

        status_metrics = {
            'composition_f1': metrics.get('density', None),
            'birads_f1': metrics.get('bi_rads', None),
            # Mean report-level POS-only, side-aware set F1.
            'finding_f1': metrics.get('entities', None),
        }

        if calculate_ci:
            status_metrics.update({
                'composition_f1_ci': metrics.get('density_ci', (None, None)),
                'birads_f1_ci': metrics.get('bi_rads_ci', (None, None)),
                'finding_f1_ci': metrics.get('entities_ci', (None, None)),
            })
        
        if self.output_dir:
            with open(self.output_dir, 'w', encoding='utf-8') as fw:
                json.dump(outputs, fw, ensure_ascii=False, indent=4)
        
        result = {
            'Status_metrics': status_metrics,
            'Relations_metrics': relations_metrics,
            'Per_sample_f1': per_sample_f1
        }

        if return_bootstrap:
            result['Bootstrap_metrics'] = {
                'composition_f1': np.asarray(
                    bootstrapped_density,
                    dtype=np.float64,
                ),
                'birads_f1': np.asarray(
                    bootstrapped_birads,
                    dtype=np.float64,
                ),
                'finding_f1': np.asarray(
                    bootstrapped_entities,
                    dtype=np.float64,
                ),
                'relations_f1': np.asarray(
                    bootstrapped_relations_f1,
                    dtype=np.float64,
                ),
            }
            result['Bootstrap_metadata'] = {
                'bootstrap_seed': int(bootstrap_seed),
                'n_bootstrap': int(n_bootstrap),
                'n_samples': int(len(sample_data_list)),
                'index_positions_preserved': True,
            }

        return result


    def get_output(
        self,
        preds,
        refs,
        calculate_ci=False,
        n_bootstrap=1000,
        bootstrap_seed=42,
        bootstrap_indices=None,
        return_bootstrap=False
    ):
        return self.test_all(
            preds,
            refs,
            calculate_ci=calculate_ci,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            bootstrap_indices=bootstrap_indices,
            return_bootstrap=return_bootstrap
        )

if __name__ == "__main__":
    # pred=["Findings: 双乳基本对称，呈不均匀致密型，见斑片状、结节状密影及脂肪组织填充，双乳未见明确肿块影及钙化灶。双乳悬韧带增粗，未见明确血管增多及导管增粗。双乳皮肤、乳晕及乳头未见明显异常。左乳乳内见一淋巴结影，大小约9mm*6mm。; Impression: 1.双乳呈不均匀致密型。 2.右乳符合BI-RADS 0，建议乳腺MRI检查。 3.左乳符合BI-RADS 2，左乳乳内一淋巴结。"]
    # ref=["Findings: 双乳呈致密型，前缘不规则见悬韧带影，腺体密度不均匀，见片状密度增高影，其间夹杂少量乳内脂肪。双乳不对称，左乳较右乳小。右乳内上象限见一卵圆形结节，大小约0.9cm×0.6cm，边缘大部分清晰，密度均与腺体接近，未见异常血管影及恶性钙化。双乳内另见少量散在点状及颗粒状钙化。左乳内未见确切块影。双乳皮下脂肪层清晰，皮肤不厚，乳头正常。右侧腋下见腺体样组织。; Impression: 1、右乳内上象限结节，性质良性，建议短期随访。BI-RADS 3。2、双乳乳腺增生，建议定期复查。BI-RADS 1。3、双乳钙化，考虑良性钙化。BI-RADS 2。4、右侧腋下副乳腺。"]
    # tool=MammoRGTool()
    # output=tool.get_output(pred, ref, calculate_ci=True)
    # print('reference:',ref[0])
    # print('generated-report:',pred[0])
    # print('Metrics:',output)
    pred="Findings: 双侧乳腺显影为不均匀致密类，实质呈索条状、结节样及絮片状，边缘模糊，部分融合； L0片示左侧乳腺上方后1/3见局灶不对称致密影，边缘遮蔽，范围约2.2×1.2cm，内未见钙化及肿 块影; 双侧乳腺皮肤正常，未见厚皮征；乳头无内陷，乳晕区未见异常；皮下脂肪层清晰、透亮；悬韧带 显影正常，未见明显增厚及牵拉征象； 双侧腋前份见淋巴结影，大小及形态未见明显异常。; Impression: 左侧乳腺上方局灶不对称致密影，考虑增生融合所致；双侧乳腺增生、部分增生融合（BI-RADS2 类，建议12个月复查）。"
    
    tool=MammoRGTool()
    output=tool.test(pred)
    # print('Metrics:',output)
    
    
    
