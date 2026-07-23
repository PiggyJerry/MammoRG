import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import pandas as pd
from transformers import pipeline, AutoTokenizer
import time
from tqdm import tqdm
import torch
import re
import ast

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

# 定义基础prompt模板（与之前相同）
base_prompt = """
---
**钼靶观察实体词典**  
- 钙化  
- 肿块  
- 乳腺增生  
- 皮肤增厚  
- 淋巴结肿大  
- 乳头凹陷  
- 结构扭曲  
- 悬韧带增粗  
- 结节   
- 结构不对称  
---

请根据钼靶观察实体词典并基于钼靶报告【{Text}】完成以下步骤:
1. **实体抽取和观察状态标注**:
   - 根据第一步得到的报告和钼靶实体词典分别输出左右乳房的观察实体的状态:
     - 乳腺密度(如"脂肪型",未提及则为"BLA")
     - Bi-Rads分类(以"Bi-Rads"开头,如"Bi-Rads 4A",未提及则为"BLA")
     - 钼靶观察实体词典中的所有观察实体及其状态:
       * POS(明确提及的)
       * NEG(否定形式排除的)
       * UNC(不确定的)
       * BLA(未提及的)

2. **输出要求**:
   - 严格按JSON格式输出:
   {{
       "Breast_assessment": {{
          "Left_breast": {{
            "Density": "",
            "Bi-Rads": "",
            "Entities": {{
              "钙化": "",
              "肿块": "",
              "乳腺增生": "",
              "皮肤增厚": "",
              "淋巴结肿大": "",
              "乳头凹陷": "",
              "结构扭曲": "",
              "悬韧带增粗": "",
              "结节": "",
              "结构不对称": ""
            }}
          }},
          "Right_breast": {{
            "Density": "",
            "Bi-Rads": "",
            "Entities": {{
              "钙化": "",
              "肿块": "",
              "乳腺增生": "",
              "皮肤增厚": "",
              "淋巴结肿大": "",
              "乳头凹陷": "",
              "结构扭曲": "",
              "悬韧带增粗": "",
              "结节": "",
              "结构不对称": ""
            }}
          }},
        }}
   }}
   /no_think
"""

def parse_model_output(content):
    """解析模型输出内容，提取Breast_assessment部分"""
    # 如果是列表,说明是完整的对话历史,需要找到assistant的content
    if isinstance(content, list):
        # 查找assistant的回复
        assistant_content = None
        for item in content:
            if item.get("role") == "assistant":
                assistant_content = item.get("content", "")
                break
        
        if assistant_content is None:
            # raise ValueError("模型输出中未找到assistant回复")
            return None
        
        content = assistant_content
    
    # 确保content是字符串
    if not isinstance(content, str):
        content = str(content)
    
    # 去除<think>标签部分
    if content.startswith("<think>"):
        parts = content.split("</think>")
        if len(parts) > 1:
            content = parts[1].strip()
    
    # 尝试直接提取JSON部分
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        # raise ValueError(f"未找到有效的JSON输出。模型输出内容:\n{content}")
        return None
    json_str = json_match.group(0)

    # # 预处理JSON字符串
    # def clean_json_string(s):
    #     if s.startswith('\ufeff'):
    #         s = s[1:]
    #     full_to_half = {'，': ',', '：': ':', '；': ';', '“': '"', '”': '"', '‘': "'", '’': "'", '。':'.'}
    #     for full, half in full_to_half.items():
    #         s = s.replace(full, half)
    #     import re
    #     s = re.sub(r'[\u200b-\u200d\ufeff]', '', s)
    #     return s
    # json_str = clean_json_string(json_str)

    # 尝试解析JSON
    try:
        model_output = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"DEBUG1: JSON解析失败，错误位置附近内容:\n{json_str[max(0, e.pos-20):e.pos+20]},\n确切的:{json_str[e.pos]}")
        try:
            json_str = json_str.replace("'", '"')
            model_output = ast.literal_eval(json_str)
        except Exception as e:
            return None
            raise ValueError(f"JSON和Python格式解析均失败: {str(e)}\n原始内容:\n{content}\模型输出json:{json_str}")
    
    # 只提取Breast_assessment部分
    if "Breast_assessment" not in model_output:
        return None
        raise ValueError("模型输出中未找到Breast_assessment字段")
    
    return model_output["Breast_assessment"]

def process_batch(records, batch_size=16):
    """批量处理记录"""
    batch_results = []
    messages = []
    record_infos = []
    
    # 准备批量数据
    for record in records:
        text = record["Text"]
        # impression = record["Cleaned_text"]["Impression"]
        patient_id = record["ID"]
        data_source = record["Data_source"]
        # image_paths = record["Image_paths"]
        # instruction = record["Instruction"]
        
        try:
            prompt = base_prompt.format(
                Text=text.replace('"', '\\"'),
            )
            messages.append([{"role": "user", "content": prompt}])
            record_infos.append((patient_id, data_source, text))
        except Exception as e:
            print(f"构造prompt时出错(记录ID {patient_id}): {str(e)}")
            batch_results.append({
                    "Data_source": data_source,
                    "ID": patient_id,
                    # "Image_paths": image_paths,
                    # "Instruction": instruction,
                    "Text": text,
                    "Breast_assessment": None
                })
    
    if not messages:
        return batch_results
    
    try:
        # 使用pipeline批量生成响应
        results = generator(
            messages,
            max_new_tokens=1024,
            do_sample=False,
            batch_size=batch_size,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 处理每个生成的响应
        for i, (result, (patient_id, data_source, text)) in enumerate(zip(results, record_infos)):
            if not result or not isinstance(result, list) or len(result) == 0:
                print(f"记录ID {patient_id} 的模型输出为空")
                continue
                
            content = result[0].get("generated_text", "")
            try:
                breast_assessment = parse_model_output(content)
                
                batch_results.append({
                    "Data_source": data_source,
                    "ID": patient_id,
                    # "Image_paths": image_paths,
                    # "Instruction": instruction,
                    "Text": text,
                    "Breast_assessment": breast_assessment
                })
            except Exception as e:
                print(f"解析模型输出时出错(记录ID {patient_id}): {str(e)}")
                # batch_results.append(None)
                
    except Exception as e:
        print(f"批量处理时出错: {str(e)}")
        batch_results.extend([None] * len(messages))
    
    return batch_results

def load_step1_json(file_path, limit=10):
    """加载step1.json文件，只读取前limit个样本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 转换为列表形式，保持原始顺序，并限制数量
    # return [value for key, value in sorted(data.items(), key=lambda x: int(x[0]))][:limit]
    return [value for key, value in sorted(data.items(), key=lambda x: int(x[0]))]
    

def process_all_records(input_file, output_file, batch_size=8):
    """处理所有记录"""
    print("开始加载step1.json...")
    records = load_step1_json(input_file)
    print(f"成功加载 {len(records)} 条记录")
    
    print("开始处理记录...")
    start_time = time.time()
    
    all_results = []
    current_batch = []
    
    for record in tqdm(records, desc="处理记录"):
        current_batch.append(record)
        
        if len(current_batch) >= batch_size:
            batch_results = process_batch(current_batch, batch_size)
            all_results.extend([r for r in batch_results if r is not None])
            current_batch = []
    
    # 处理剩余的记录
    if current_batch:
        batch_results = process_batch(current_batch, batch_size)
        all_results.extend([r for r in batch_results if r is not None])
    
    # 将结果数组转换为带编号的对象
    numbered_results = {}
    for idx, result in enumerate(all_results, start=1):
        if result is not None:  # 只保存有效结果
            numbered_results[str(idx)] = result
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(numbered_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成,共处理 {len(all_results)} 条记录")
    print(f"总耗时: {(time.time()-start_time)/60:.2f} 分钟")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "/home/jiayi/MammoRG/MammoRGTool/external_eval/external_norule.json"
    output_file = "/home/jiayi/MammoRG/Qwen3-32b-external/step2.json"
    
    process_all_records(input_file, output_file, batch_size=8)