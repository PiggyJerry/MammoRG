import json
import re
from collections import defaultdict
from tqdm import tqdm

def convert_roman_to_arabic(roman):
    roman_map = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6',
        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6',
        'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3', 'ⅳ': '4', 'ⅴ': '5', 'ⅵ': '6',
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5', 'Ⅵ': '6'
    }
    return roman_map.get(roman.strip(), roman)

replacement_rules = {
        r'(?i)bi[-/ ]?rads\s*[:：]?\s*([ivxⅰ-ⅵⅠ-Ⅵ]+)\s*-\s*([ivxⅰ-ⅵⅠ-Ⅵ]+)':
            lambda m: f"BI-RADS {convert_roman_to_arabic(m.group(1))}-{convert_roman_to_arabic(m.group(2))}",
        
        r'(?i)(bi[-/ ]?rads)\s*[:：]?\s*([0-6ⅰⅱⅲⅳⅴⅵⅠⅡⅢⅣⅤⅥIiVv]+)([a-c])?(\d*)\s*类?': 
            lambda m: (
                f"BI-RADS {convert_roman_to_arabic(m.group(2))}" +
                f"{m.group(3).upper() if m.group(3) else ''}" +
                (f"。{m.group(4)}" if m.group(4) else "")
            ),

        r'A腺体型|a腺体型|A型|a型|A类|a类|脂肪腺体型|脂肪为主型|ACR A|ACR a|acr A|acr a': "脂肪型",
        r'B腺体型|b腺体型|B型|b型|B类|b类|散在纤维体型|散在纤维腺体型|少量腺体型|均衡腺体型|散在腺体型|散在稀疏腺体型|少腺体型|疏松腺体型|散在纤维型|均质纤维腺体型|ACR B|ACR b|acr B|acr b': "纤维腺体型",
        r'C腺体型|c腺体型|C型|c型|C类|c类|多量腺体型|中量腺体型|多腺体型|脂肪腺体混合型|不均匀性致密型|不均质致密型|不均匀纤维腺体型|不均匀腺体型|散在纤维不均匀致密型|散在不均匀致密型|不均匀致密线腺体型|混合腺体型|ACR C|ACR c|acr C|acr c': "不均匀致密型",
        r'D腺体型|d腺体型|D型|d型|D类|d类|致密腺体型|极度致密型|十分致密型|ACR D|ACR d|acr D|acr d': "致密型",
        r'欠':"不",
        r'稍':"",
        r'尚':"",
        r'乳腺体': "乳腺",
        r'肿物': "肿块",
        r'淋巴结增大|肿大淋巴结|增大淋巴结|稍大淋巴结': "淋巴结肿大",
        r'乳腺增生症': "乳腺增生",
        r'乳头内陷': "乳头凹陷",
        r'结构紊乱': "结构扭曲",
        r'结构稍紊乱': "结构稍扭曲",
        r'皮肤厚|乳晕增厚': "皮肤增厚",
        r"Cooper's韧带|Cooper韧带|Cooper’s": "悬韧带",
        r"Cooper's韧带增粗|Cooper韧带增粗|悬韧带增厚|Cooper's韧带增厚|Cooper韧带增厚": "悬韧带增粗",
        r'导管扩张|导管增生': "导管增粗",
        r'非对称性致密': "结构不对称",
        r'双乳腺|双侧腺体|双侧乳腺': "双乳",
        r'左乳腺|左侧腺体|左侧乳腺': "左乳",
        r'右乳腺|右侧腺体|右侧乳腺': "右乳",
        r'两侧': "双侧",
        r'双侧腋窝|双腋区|双腋下': "双侧腋下",
        r'左侧腋窝|左腋区|左腋下': "左侧腋下",
        r'右侧腋窝|右腋区|右腋下': "右侧腋下",
        r'双侧(?!乳腺|腋下|乳晕区|中央区)': "双乳",
        r'左侧(?!乳腺|腋下|乳晕区|中央区)': "左乳",
        r'右侧(?!乳腺|腋下|乳晕区|中央区)': "右乳",
        r'左乳上象限|左乳上份|左乳外份': "左乳外上象限",
        r'右乳上象限|右乳上份|右乳外份': "右乳外上象限",
        r'左乳下象限|左乳下份': "左乳外下象限",
        r'右乳下象限|右乳下份': "右乳外下象限",
        r'左乳内份': "左乳内上象限",
        r'右乳内份': "右乳内上象限",
        r'左乳乳晕区': "左侧乳晕区",
        r'右乳乳晕区': "右侧乳晕区",
        r'双乳乳晕区': "双侧乳晕区",
        r'左乳中央区': "左侧中央区",
        r'右乳中央区': "右侧中央区",
        r'双乳中央区': "双侧中央区",
        r'小点状|圆点状': "点状",
        r'细点状': "斑点状",
        r'边界': "边缘",
        r'小簇状': "簇状",
        r'局限性': "局灶性",
        r'欠清|不清': "模糊",
        r'清楚': "清晰",
        r'聚集|密集|群集': "集中",
        r'少许|较少': "少量",
        r'较多': "大量",
        r'数枚': "多枚",
        r'弥散': "弥漫",
        r'凹凸不平|毛糙': "不规则",
        r'边缘部分清': "部分边缘清",
        r'边缘清(?!晰)': "边缘清晰",
        r'乳增生': "乳腺增生",
        r'双乳腺增生': "双乳乳腺增生",
        r'左乳腺增生': "左乳乳腺增生",
        r'右乳腺增生': "右乳乳腺增生",
    }

def apply_all_replacements(text):
    if not isinstance(text, str):
        return text
    
    for pattern, replacement in replacement_rules.items():
        text = re.sub(pattern, replacement, text)
    return text

def preprocess_text_fields(data_dict):
    if "Breast_assessment" in data_dict:
        for side in ["Left_breast", "Right_breast"]:
            if side in data_dict["Breast_assessment"]:
                if "BI-RADS" in data_dict["Breast_assessment"][side]:
                    data_dict["Breast_assessment"][side]["BI-RADS"] = apply_all_replacements(
                        data_dict["Breast_assessment"][side]["BI-RADS"]
                    )
                if data_dict["Breast_assessment"][side]["BI-RADS"] in ['0','1','2','3','4a','4b','4c','4A','4B','4C','5','6']:
                    data_dict["Breast_assessment"][side]["BI-RADS"]='BI-RADS '+data_dict["Breast_assessment"][side]["BI-RADS"]
                if "Density" in data_dict["Breast_assessment"][side]:
                    data_dict["Breast_assessment"][side]["Density"] = apply_all_replacements(
                        data_dict["Breast_assessment"][side]["Density"]
                    )
    if "Relations" in data_dict and data_dict['Relations'] is not None:

        new_relations = []
        for relation in data_dict["Relations"]:
            if isinstance(relation, (list, tuple)):
                new_relation = [apply_all_replacements(item) if isinstance(item, str) else item for item in relation]
                new_relations.append(new_relation)
            elif isinstance(relation, dict):
                new_relation = {k: apply_all_replacements(v) if isinstance(v, str) else v for k, v in relation.items()}
                new_relations.append(new_relation)
            else:
                new_relations.append(relation)
        
        data_dict["Relations"] = new_relations

    return data_dict

def process_samples(data):
    relation_dict={
        "Located_at": {
            "钙化": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "肿块": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "乳腺增生": [
            "左乳外上象限",
            "右乳",
            "左侧中央区",
            "右侧中央区",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "左乳内上象限",
            "双乳"
            ],
            "皮肤增厚": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "左侧乳晕区",
            "右侧乳晕区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "左乳内下象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "淋巴结肿大": [
            "右侧腋下",
            "双侧腋下",
            "左侧腋下"
            ],
            "乳头凹陷": [
            "双乳",
            "右乳",
            "左乳"
            ],
            "结构扭曲": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "悬韧带增粗": [
            "左乳外上象限",
            "双乳",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "结节": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限",
            "左乳内上象限"
            ],
            "结构不对称": [
            "左乳外上象限",
            "双乳",
            "左侧中央区",
            "右侧中央区",
            "右乳外下象限",
            "右乳",
            "左乳",
            "右乳外上象限",
            "右乳内上象限",
            "右乳内下象限",
            "左乳内下象限",
            "左乳外下象限"
            ]
        },
        "Suggestive_of": {
            "钙化": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "肿块": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "乳腺增生": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "皮肤增厚": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "淋巴结肿大": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "乳头凹陷": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "结构扭曲": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "结节": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ],
            "结构不对称": [
            "BI-RADS 2",
            "BI-RADS 6",
            "BI-RADS 4B",
            "BI-RADS 0",
            "BI-RADS 4C",
            "乳腺癌",
            "BI-RADS 1",
            "BI-RADS 3",
            "BI-RADS 5",
            "BI-RADS 4A"
            ]
        },
        'Modified_by':{
        '钙化': [
            '局灶性', 
            '边缘不规则', 
            '模糊不定形', 
            '分叶状', 
            '圆形', 
            '颗粒状', 
            '卵圆形', 
            '结节状', 
            '密度增高且不均匀', 
            '局部', 
            '点状', 
            '部分边缘模糊', 
            '模糊',
            '线虫样', 
            '簇状', 
            '不规则',
            '斑点状',
            '边缘模糊',
            '边缘清晰',
            '密度不均匀',
            '粗糙不均质',
            '壳样',
            '类圆形'
            ],
        '肿块': [
            '毛刺影',
            '局灶性',
            '密度增高',
            '边缘不规则',
            '分叶状',
            '模糊不定形',
            '圆形', 
            '颗粒状', 
            '卵圆形',
            '结节状',
            '密度增高且不均匀',
            '局部',
            '点状',
            '部分边缘模糊', 
            '模糊', 
            '密度均匀',
            '簇状', 
            '不规则', 
            '边缘模糊', 
            '局部不规则', 
            '边缘清晰', 
            '密度不均匀', 
            '部分边缘清晰',
            '类圆形'
            ], 
        '乳腺增生': [
            '结节状',
            '边缘模糊',
            '局灶性',
            '密度增高',
            '密度不均匀', 
            '模糊'
            ],
        '皮肤增厚': [
            '局部', 
            '密度增高',
            '模糊'
            ], 
        '淋巴结肿大': [
            '密度增高', 
            '边缘不规则',
            '卵圆形', 
            '密度增高且不均匀',
            '局部', 
            '密度均匀', 
            '不规则',
            '边缘模糊',
            '局部不规则',
            '边缘清晰',
            '密度不均匀', 
            '类圆形'
            ], 
        '乳头凹陷': [
            '局部', 
            '模糊'
            ], 
        '结构扭曲': [
            '不规则',
            '密度增高且不均匀',
            '边缘模糊',
            '局部不规则', 
            '局部', 
            '局灶性', 
            '密度增高',
            '密度不均匀',
            '边缘不规则',
            '模糊'
            ], 
        '悬韧带增粗': [
            '局部', 
            '密度增高'
            ], 
        '结节': [
            '毛刺影',
            '局灶性', 
            '密度增高',
            '边缘不规则', 
            '分叶状',
            '圆形', 
            '颗粒状',
            '卵圆形', 
            '结节状', 
            '密度增高且不均匀',
            '局部',
            '点状',
            '部分边缘模糊',
            '模糊', 
            '密度均匀',
            '簇状',
            '不规则',
            '边缘模糊',
            '局部不规则', 
            '边缘清晰', 
            '密度不均匀', 
            '部分边缘清晰',
            '类圆形'
            ], 
        '结构不对称': [
            '不规则', 
            '结节状', 
            '密度增高且不均匀',
            '边缘模糊', 
            '局灶性', 
            '局部',
            '密度增高', 
            '密度不均匀',
            '部分边缘模糊',
            '部分边缘清晰'
            ]
        }
        }
    standard_entities = {
        "密度": ["脂肪型", "纤维腺体型", "不均匀致密型", "致密型","BLA"],
        "观察": ["钙化", "肿块", "乳腺增生", "皮肤增厚", "淋巴结肿大", "乳头凹陷", 
                "结构扭曲", "悬韧带增粗", "结节", "结构不对称"],
        "诊断": ["BI-RADS 0", "BI-RADS 1", "BI-RADS 2", "BI-RADS 3",
               "BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C", "BI-RADS 5", "BI-RADS 6","乳腺癌", "BLA"],
    }
    
    exclude={}
    valid_status = ["POS", "NEG", "UNC", "BLA"]
    valid_relation_types = ["Located_at", "Suggestive_of", "Modified_by"]
    updated_data = []

    valid_num=0
    case_data = preprocess_text_fields(data)
    # print('2:',case_data.get("Relations"))
    text=case_data['Text']

    breast_assessment = case_data.get("Breast_assessment", {})
    
    for side in ["Left_breast", "Right_breast"]:
            
        breast_data = breast_assessment[side]
        density = breast_data.get("Density")
        if density and density not in standard_entities["密度"]:
            breast_data["Density"]="BLA"
        
        birads = breast_data.get("BI-RADS")
        if birads and birads not in standard_entities["诊断"]:
            breast_data["BI-RADS"]="BLA"
        
        entities = breast_data.get("Entities", {})
        for entity, status in list(entities.items()):
            if entity not in standard_entities["观察"]:
                del entities[entity] 
                continue 
            
            if status not in valid_status:
                entities[entity] = "BLA" 
    
    if case_data["Relations"] is None:
        case_data["Relations"]=[]
    relations = case_data.get("Relations")
    valid_relations = []
    seen_relations = set() 
    for relation in relations:
        if len(relation) != 3:
            continue
        
        entity1, rel_type, entity2 = relation
        
        if rel_type not in valid_relation_types:
            continue
        
        def find_and_fix_entity_position(entity):
            full_text = text
            
            matches = list(re.finditer(re.escape(entity), full_text))
            if not matches:
                return False
            else:
                return True
        new_start1 = find_and_fix_entity_position(entity1)
        if new_start1==False:
            continue 
        
        new_start2 = find_and_fix_entity_position(entity2)
        if new_start2==False:
            continue 
        
        if rel_type == 'Suggestive_of':
            if entity1 not in text.split('Impression:')[-1] if 'Impression:' in text else text:
                continue
        
        valid_relation = False
        
        if rel_type == "Located_at":
            if (entity1 in standard_entities["观察"] and 
                entity2 in relation_dict[rel_type][entity1]):
                valid_relation = True
            elif (entity1 in standard_entities["观察"] and 
                entity2 not in relation_dict[rel_type][entity1]):
                found_loc = False

                for loc in relation_dict[rel_type][entity1]:
                    if loc in entity2:
                        entity2 = loc
                        found_loc = True
                        break 
                
                if not found_loc:
                    if any(side in entity2 for side in ['左', '右', '双']):
                        if '左' in entity2:
                            correct_entity = '左乳'
                        elif '右' in entity2:
                            correct_entity = '右乳'
                        elif '双' in entity2:
                            correct_entity = '双乳'
                        else:
                            correct_entity = entity2 
                        entity2 = correct_entity
                
                if entity2 in relation_dict[rel_type][entity1]:
                    valid_relation = True
        
        elif rel_type == "Suggestive_of":
            if ((entity1 in standard_entities["观察"]) and 
                entity2 in relation_dict[rel_type][entity1]):
                valid_relation = True
        
        elif rel_type == "Modified_by":
            if (entity1 in standard_entities["观察"] and 
                (entity2 in relation_dict[rel_type][entity1])):
                valid_relation = True
        
        if not valid_relation:
            continue

        relation_key = (entity1, rel_type, entity2)
        if relation_key in seen_relations:
            continue
        
        if breast_assessment['Left_breast']['Entities'][entity1]!='POS' and breast_assessment['Right_breast']['Entities'][entity1]!='POS':
            continue
        
        seen_relations.add(relation_key)
        valid_relations.append(relation_key)
    case_data['Relations']=valid_relations
    case_data['Triples']=set(tuple(r) for r in valid_relations)
    return case_data
    
