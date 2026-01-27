import json
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm

# 1. 导入 MammoRGTool
from tool import MammoRGTool

INPUT_EXCEL = "/home/jiayi/MammoRG/MammoRGTool/gz2.xlsx"
OUTPUT_JSON = "/home/jiayi/MammoRG/MammoRGTool/gz2.json"

def parse_output_to_fields(output):
    """
    从 MammoRGTool 的 output 中解析需要的字段
    """
    text = output.get("Text", "")

    cleaned_findings = ""
    cleaned_impression = ""

    try:
        cleaned_findings = text.split("Findings:")[-1].split("; Impression")[0].strip()
    except Exception:
        cleaned_findings = ""

    try:
        cleaned_impression = text.split("; Impression:")[-1].strip()
    except Exception:
        cleaned_impression = ""

    cleaned_text = {
        "Findings": cleaned_findings,
        "Impression": cleaned_impression
    }

    relations = output.get("Relations", [])
    breast_assessment = output.get("Breast_assessment", {})

    return cleaned_text, relations, breast_assessment


def main():
    # 2. 初始化工具
    tool = MammoRGTool()

    # 3. 读取 Excel 文件
    print(f"读取 Excel 文件: {INPUT_EXCEL}")
    try:
        df = pd.read_excel(INPUT_EXCEL)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return
    
    # 检查必要的列是否存在
    required_columns = ["ID", "Findings", "Impression", "image_paths"]
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: Excel文件中缺少必要的列: {col}")
            return
    
    print(f"找到 {len(df)} 条记录")
    
    # 4. 转换为字典格式，key从1开始
    data = {}
    
    # 5. 逐条处理
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理 Excel 数据"):
        # 使用索引+1作为key
        key = str(idx + 1)
        
        # 获取ID
        item_id = str(row["ID"])
        
        # 处理image_paths
        # 假设image_paths列是字符串形式的字典
        image_paths_str = str(row["image_paths"])
        try:
            # 尝试解析JSON格式的字符串
            if image_paths_str.startswith("{") and image_paths_str.endswith("}"):
                image_paths = eval(image_paths_str)  # 使用eval解析字典字符串
            else:
                # 如果不是字典格式，创建空字典
                image_paths = {}
                print(f"警告: 第{idx+1}行的image_paths格式异常: {image_paths_str[:50]}...")
        except Exception as e:
            print(f"解析image_paths失败 (行{idx+1}): {e}")
            image_paths = {}
        
        # 创建基础数据结构
        item = {
            "Data_source": "gz2",  # 根据文件名设置数据源
            "ID": item_id,
            "Origin_text": {
                "Findings": str(row["Findings"]) if pd.notna(row["Findings"]) else "",
                "Impression": str(row["Impression"]) if pd.notna(row["Impression"]) else ""
            },
            "Image_paths": image_paths,
            "Instruction": {},
            "Cleaned_text": {
                "Findings": "",
                "Impression": ""
            },
            "Relations": [],
            "Breast_assessment": {}
        }
        
        # 获取原始文本
        findings = item["Origin_text"]["Findings"]
        impression = item["Origin_text"]["Impression"]
        
        if not findings and not impression:
            print(f"跳过第{idx+1}行: Findings和Impression都为空")
            data[key] = item
            continue
        
        # 6. 构造输入文本
        input_text = f"Findings:{findings}; Impression:{impression}"
        
        # 7. 调用模型
        try:
            output = tool.test(input_text)
        except Exception as e:
            print(f"[ERROR] Key={key}, ID={item_id}, error={e}")
            data[key] = item
            continue
        
        # 8. 解析输出
        cleaned_text, relations, breast_assessment = parse_output_to_fields(output)
        
        # 9. 更新字段
        item["Cleaned_text"] = cleaned_text
        item["Relations"] = relations
        item["Breast_assessment"] = breast_assessment
        
        # 添加到结果中
        data[key] = item
    
    # 10. 保存到 JSON 文件
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 处理完成，共处理 {len(data)} 条记录")
    print(f"✅ 已保存到: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()