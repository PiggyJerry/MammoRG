import os
import json
import re
from collections import defaultdict
from transformers import pipeline, AutoTokenizer
import torch
from tqdm import tqdm

# 初始化模型和tokenizer
model_name = "Qwen/Qwen3-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 使用pipeline初始化生成器
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype="auto",
    device_map="auto"
)
generator.tokenizer.padding_side = "left"  # 设置padding方向

print(f"模型已加载到设备: {generator.device}")

# 定义检查prompt模板
check_prompt_template = """钼靶实体词典：
---
**1. 密度**  
- 脂肪型  
- 纤维腺体型  
- 不均匀致密型  
- 致密型  

**2. 观察**  
- 钙化  
- 肿块  
- 乳腺增生  
- 皮肤增厚  
- 淋巴结肿大  
- 乳头凹陷  
- 乳头变形  
- 乳晕增厚  
- 结构扭曲  
- 悬韧带增粗  
- 结节  
- 导管增粗  
- 血管增粗   
- 结构不对称  

**3. 诊断**  
- Bi-Rads 0  
- Bi-Rads 1  
- Bi-Rads 2  
- Bi-Rads 3  
- Bi-Rads 4  
- Bi-Rads 4A  
- Bi-Rads 4B  
- Bi-Rads 4C  
- Bi-Rads 5  
- Bi-Rads 6   

**4. 解剖区域**  
- 双乳 / 双侧乳腺  
- 右乳 / 右侧乳腺  
- 左乳 / 左侧乳腺  
- 双侧腋下 / 右侧腋下 / 左侧腋下  
- 左乳外上象限 / 左乳外下象限 / 左乳内上象限 / 左乳内下象限  
- 右乳外上象限 / 右乳外下象限 / 右乳内上象限 / 右乳内下象限  
- 左乳上份 / 左乳下份 / 左乳外份 / 左乳内份  
- 右乳上份 / 右乳下份 / 右乳外份 / 右乳内份  
- 左乳中央区 / 右乳中央区  
- 右乳晕后外上方 / 右乳晕后区  
- 左乳晕后外上方 / 左乳晕后区  
- 左乳皮下脂肪层 / 右乳皮下脂肪层  

**5. 修饰词**  
- 良性 / 恶性  
- 对称 / 不对称  
- 均匀 / 不均  
- 点状 / 斑点状 / 片状 / 片絮状 / 团片状 / 斑片状  
- 簇状 / 团簇状 / 团状 / 颗粒状  
- 放射状 / 分叶状 / 线虫样 / 爆米花样 / 结节状  
- 圆形 / 卵圆形 / 类圆形  
- 点样 / 环形 / 毛刺 / 壳样  
- 区域性 / 隐匿性 / 局灶性 / 多形性  
- 模糊 / 清晰  
- 集中 / 少量 / 大量 / 多枚 / 弥漫 / 不规则 / 大 / 小
---

**替换规则**(格式: 原表述 → 标准术语, 分开的短语也应替换, 比如: "导管未见扩张"→"导管未见增粗"): 
    "a型"、"脂肪腺体型"、"脂肪为主型" → "脂肪型"  
    "b型"、"散在纤维体型"、"少量腺体型"、"均衡腺体型"、"散在腺体型"、"散在稀疏腺体型"、"少腺体型"、"疏松腺体型"、"散在纤维型" → "纤维腺体型"  
    "c型"、"多量腺体型"、"中量腺体型"、"多腺体型"、"脂肪腺体混合型"、"不均匀性致密型" → "不均匀致密型"  
    "d型"、"致密腺体型" → "致密型"  
    "淋巴结增大" → "淋巴结肿大"  
    "乳头内陷" → "乳头凹陷"  
    "结构紊乱" → "结构扭曲"  
    "Cooper's韧带增粗"、"Cooper韧带增粗"、"悬韧带增厚"、"Cooper's韧带增厚"、"Cooper韧带增厚" → "悬韧带增粗"  
    "导管扩张"、"导管增生" → "导管增粗"  
    "非对称性致密" → "结构不对称"  
    "双乳腺"、"双侧腺体" → "双侧乳腺"  
    "左乳腺"、"左侧腺体" → "左侧乳腺"  
    "右乳腺"、"右侧腺体" → "右侧乳腺"  
    "双侧腋窝"、"双腋区"、"双腋下" → "双侧腋下"  
    "左侧腋窝"、"左腋区"、"左腋下" → "左侧腋下"  
    "右侧腋窝"、"右腋区"、"右腋下" → "右侧腋下"  
    "小点状"、"圆点状" → "点状"  
    "细点状" → "斑点状"  
    "小簇状" → "簇状"  
    "局限性" → "局灶性"  
    "欠清" → "模糊"  
    "清楚" → "清晰"  
    "聚集"、"密集"、"群集" → "集中"  
    "少许"、"较少" → "少量"  
    "较多" → "大量"  
    "数枚" → "多枚"  
    "弥散" → "弥漫"  
    "凹凸不平"、"毛糙" → "不规则"  

请根据以下标准检查钼靶报告处理结果，并给出修正建议：

### 原始报告:
{origin_text}

### 处理后的报告:
{cleaned_text}

### 乳房评估:
{breast_assessment}

### 实体关系:
{relations}

### 检查标准:
1. **Cleaned_text检查**:
   - 是否所有词汇都替换为标准表达
   - 是否保留了原始含义
   - 是否有遗漏的替换

2. **乳房评估检查**:
   - 密度是否使用标准表达(脂肪型、纤维腺体型、不均匀致密型、致密型)
   - Bi-Rads分类是否正确
   - 所有观察实体的状态(POS/NEG/UNC/BLA)是否符合报告内容

3. **关系抽取检查**:
   - 关系类型是否正确(Located_at/Suggestive_of/Modify)
   - 实体组合是否符合规则
   - 实体是否来自标准词典

### 请按以下JSON格式提供修正建议:
{{
    "修正建议": {{
        "Cleaned_text": {{
            "Findings": ["需要修正的内容及建议", ...],
            "Impression": ["需要修正的内容及建议", ...]
        }},
        "Breast_assessment": {{
            "Left_breast": {{
                "Density": "修正建议",
                "Bi-Rads": "修正建议",
                "Entities": {{
                    "实体名称": "修正建议",
                    ...
                }}
            }},
            "Right_breast": {{...}}
        }},
        "Relations": [
            ["修正建议", ...],
            ...
        ]
    }},
    "检查通过": bool  // 是否所有检查项都通过
}}
"""

def load_json_file(json_file_path):
    """加载JSON文件"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, output_file_path):
    """保存JSON文件"""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_fix_suggestion(case_data):
    """使用Qwen3-32B生成修正建议"""
    origin_text = json.dumps(case_data.get("Origin_text", {}), ensure_ascii=False)
    cleaned_text = json.dumps(case_data.get("Cleaned_text", {}), ensure_ascii=False)
    breast_assessment = json.dumps(case_data.get("Breast_assessment", {}), ensure_ascii=False)
    relations = json.dumps(case_data.get("Relations", []), ensure_ascii=False)
    
    prompt = check_prompt_template.format(
        origin_text=origin_text,
        cleaned_text=cleaned_text,
        breast_assessment=breast_assessment,
        relations=relations
    )
    
    try:
        response = generator(
            [{"role": "user", "content": prompt}],
            max_new_tokens=4096,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 提取模型输出中的JSON部分
        content = response[0]["generated_text"]
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            print("未找到有效的JSON输出")
            return None
            
        json_str = json_match.group(0)
        return json.loads(json_str)
        
    except Exception as e:
        print(f"生成修正建议时出错: {str(e)}")
        return None

def apply_fix_suggestions(data, suggestions):
    """应用修正建议到数据中"""
    if not suggestions or "修正建议" not in suggestions:
        return data
    
    fixes = suggestions["修正建议"]
    
    # 应用Cleaned_text修正
    if "Cleaned_text" in fixes:
        cleaned_text = data.get("Cleaned_text", {})
        for field in ["Findings", "Impression"]:
            if field in fixes["Cleaned_text"]:
                # 这里简单实现，实际可能需要更复杂的文本替换
                if field in cleaned_text:
                    for fix in fixes["Cleaned_text"][field]:
                        # 简单的字符串替换示例
                        if "→" in fix:
                            old, new = fix.split("→")
                            cleaned_text[field] = cleaned_text[field].replace(old.strip(), new.strip())
        data["Cleaned_text"] = cleaned_text
    
    # 应用Breast_assessment修正
    if "Breast_assessment" in fixes:
        breast_assessment = data.get("Breast_assessment", {})
        for side in ["Left_breast", "Right_breast"]:
            if side in fixes["Breast_assessment"]:
                side_data = breast_assessment.get(side, {})
                side_fixes = fixes["Breast_assessment"][side]
                
                if "Density" in side_fixes:
                    side_data["Density"] = side_fixes["Density"]
                if "Bi-Rads" in side_fixes:
                    side_data["Bi-Rads"] = side_fixes["Bi-Rads"]
                
                if "Entities" in side_fixes:
                    entities = side_data.get("Entities", {})
                    for entity, fix in side_fixes["Entities"].items():
                        if entity in entities:
                            entities[entity] = fix
                    side_data["Entities"] = entities
                
                breast_assessment[side] = side_data
        data["Breast_assessment"] = breast_assessment
    
    # 应用Relations修正
    if "Relations" in fixes:
        # 这里简单替换整个Relations，实际可能需要更精细的处理
        data["Relations"] = fixes["Relations"]
    
    return data

def process_json_file(input_file, output_file, limit=10):
    """处理JSON文件并生成修正后的版本（只处理前limit个样例）"""
    data = load_json_file(input_file)
    updated_data = {}
    
    # 只处理前limit个样例
    for i, (case_id, case_data) in tqdm(enumerate(data.items()), desc="处理样例", total=min(limit, len(data))):
        if i >= limit:
            break
            
        print(f"\n正在处理样例 {case_id}...")
        
        # 生成修正建议
        suggestions = generate_fix_suggestion(case_data)
        
        # 应用修正建议
        if suggestions:
            print(f"样例 {case_id} 的修正建议: {json.dumps(suggestions, ensure_ascii=False, indent=2)}")
            updated_case = apply_fix_suggestions(case_data, suggestions)
            updated_case["检查结果"] = {
                "检查通过": suggestions.get("检查通过", False),
                "修正建议": suggestions.get("修正建议", {})
            }
            updated_data[case_id] = updated_case
        else:
            updated_data[case_id] = case_data
    
    # 保存更新后的JSON文件
    save_json_file(updated_data, output_file)
    print(f"\n处理完成，前{limit}个样例的结果已保存到: {output_file}")

if __name__ == "__main__":
    input_json = "/home/jiayi/MammoRG/report/process_output.json"
    output_json = "/home/jiayi/MammoRG/report/process_output_updated_top10.json"
    
    # 只检查前10个样例
    process_json_file(input_json, output_json, limit=10)