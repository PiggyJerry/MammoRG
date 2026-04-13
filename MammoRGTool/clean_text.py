import re
import json
import os
import pandas as pd
from glob import glob
from tqdm import tqdm


def convert_roman_to_arabic(roman):
    roman_map = {
        'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6',
        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6',
        'ⅰ': '1', 'ⅱ': '2', 'ⅲ': '3', 'ⅳ': '4', 'ⅴ': '5', 'ⅵ': '6',
        'Ⅰ': '1', 'Ⅱ': '2', 'Ⅲ': '3', 'Ⅳ': '4', 'Ⅴ': '5', 'Ⅵ': '6'
    }
    return roman_map.get(roman.strip(), roman)


# ====== 新增：括号保护（支持嵌套，支持（）和()） ======
def protect_parentheses(text: str):
    """
    将文本中所有括号段（（）和()，支持嵌套）替换为占位符。
    返回：(替换后的文本, protected_segments列表)
    """
    if not isinstance(text, str) or not text:
        return text, []

    pairs = {'(' : ')', '（': '）'}
    opens = set(pairs.keys())
    closes = set(pairs.values())
    close_to_open = {v: k for k, v in pairs.items()}

    protected = []
    stack = []  # elements: (open_char, start_index)
    spans = []

    # 找出所有最外层括号span（支持嵌套）
    for i, ch in enumerate(text):
        if ch in opens:
            stack.append((ch, i))
        elif ch in closes:
            if stack and stack[-1][0] == close_to_open[ch]:
                open_ch, start = stack.pop()
                if not stack:
                    spans.append((start, i + 1))

    if not spans:
        return text, []

    # 从后往前替换，避免索引偏移
    new_text = text
    for idx, (s, e) in enumerate(reversed(spans)):
        protected.append(new_text[s:e])
        placeholder = f"__PAREN_{len(protected)-1}__"
        new_text = new_text[:s] + placeholder + new_text[e:]

    # 上面 protected 的顺序是从后往前收集的，保持占位符与列表一致即可
    return new_text, protected


def restore_parentheses(text: str, protected):
    if not isinstance(text, str) or not protected:
        return text
    # 依次替换回去
    for i, seg in enumerate(protected):
        text = text.replace(f"__PAREN_{i}__", seg)
    return text


def apply_outside_parentheses(text: str, func):
    """只对括号外内容应用 func；括号内原样保留"""
    if not isinstance(text, str):
        return text
    tmp, protected = protect_parentheses(text)
    tmp = func(tmp)
    tmp = restore_parentheses(tmp, protected)
    return tmp


# ====== 你的原始清洗逻辑：保持不变（只做轻微结构整理） ======
def process_text(text):
    if not isinstance(text, str):
        return text

    # 去掉开头的“2D显示: / 3D显示:”等
    text = re.sub(r'^\s*([23]D(?:\+[23]D)?\s*显示\s*[:：]\s*)', '', text, flags=re.IGNORECASE)

    # BI-RADS 里把“2 3.”这种误分隔改成“2。3.”
    text = re.sub(
        r'(?i)(BI[-/]RADS\s*[:：]\s*)([0-6][A-Ca-c]?)\s+([0-9])(\s*[.,、，])',
        lambda m: f"{m.group(1)}{m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )
    text = re.sub(
        r'(?i)(BI[-/]RADS\s*[:：]\s*)([0-6])\s{1,}([0-9])(\s*[.,、，])',
        lambda m: f"{m.group(1)}{m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )
    text = re.sub(
        r'(?i)(BI[-/]RADS\s*[:：]?\s*)([0-6])\s*\n\s*([0-9])(\s*[.,、，])',
        lambda m: f"{m.group(1)}{m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )
    text = re.sub(
        r'(?i)(BI[-/]RADS\s*[:：]\s*)([0-6])([0-9])(\s*[.,、，])',
        lambda m: f"{m.group(1)}{m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )
    text = re.sub(
        r'(?i)(Bi[-/]Rads\s+)([0-6])([0-9])(\s*[.,、，])',
        lambda m: f"{m.group(1)}{m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )

    # 你这里有一条重复的BI-RADS处理，我保持你的原意，但建议只留一种；先不改动逻辑
    text = re.sub(
        r'(?i)(BI[-/]RADS\s*[:：]\s*)([0-6])([0-9])(\s*[.,、，])',
        lambda m: f"BI-RADS {m.group(2)}。{m.group(3)}{m.group(4)}",
        text
    )

    # 统一BI-RADS写法
    text = re.sub(
        r'(?i)(BI[-/]RADS|Bi[-/]Rads)\s*[:：]?\s*',
        'BI-RADS ',
        text
    )

    # 压缩空白
    text = re.sub(r'[\s\u3000]+', ' ', text).strip()

    # 中央区/乳晕区补左/右/双（仅在括号外生效，因为外层会保护括号）
    def process_central_and_areolar(match):
        term = match.group()
        if re.search(r'(左[侧乳]|右[侧乳]|双[侧乳]?)\s*(中央区|乳晕区)', text[max(0, match.start()-5):match.end()], re.IGNORECASE):
            return term

        if re.search(r'左|左侧', text[:match.start()] + text[match.end():], re.IGNORECASE):
            return f"左侧{term}"
        elif re.search(r'右|右侧', text[:match.start()] + text[match.end():], re.IGNORECASE):
            return f"右侧{term}"
        elif re.search(r'双|双侧', text[:match.start()] + text[match.end():], re.IGNORECASE):
            return f"双侧{term}"
        else:
            return term

    text = re.sub(r'(?<![左|右|双])(中央区|乳晕区)', process_central_and_areolar, text)

    replacement_rules = {
        r'(?i)bi[-/ ]?rads\s*[:：]?\s*([ivxⅰ-ⅵⅠ-Ⅵ]+)\s*-\s*([ivxⅰ-ⅵⅠ-Ⅵ]+)':
            lambda m: f"BI-RADS {convert_roman_to_arabic(m.group(1))}-{convert_roman_to_arabic(m.group(2))}",

        r'(?i)(bi[-/ ]?rads)\s*[:：]?\s*([0-6ⅰⅱⅲⅳⅴⅵⅠⅡⅢⅣⅤⅥIiVv]+)([a-c])?(\d*)\s*类?':
            lambda m: (
                f"BI-RADS {convert_roman_to_arabic(m.group(2))}"
                + f"{m.group(3).upper() if m.group(3) else ''}"
                + (f"。{m.group(4)}" if m.group(4) else "")
            ),

        r'A腺体型|a腺体型|A型|a型|A类|a类|脂肪腺体型|脂肪为主型|ACR A|ACR a|acr A|acr a|脂肪腺体类': "脂肪型",
        r'B腺体型|b腺体型|B型|b型|B类|b类|散在纤维体型|散在纤维腺体型|少量腺体型|均衡腺体型|散在腺体型|散在稀疏腺体型|少腺体型|疏松腺体型|散在纤维型|均质纤维腺体型|ACR B|ACR b|acr B|acr b|散在纤维腺体类': "纤维腺体型",
        r'C腺体型|c腺体型|C型|c型|C类|c类|多量腺体型|中量腺体型|多腺体型|脂肪腺体混合型|不均匀性致密型|不均质致密型|不均匀纤维腺体型|不均匀腺体型|散在纤维不均匀致密型|散在不均匀致密型|不均匀致密线腺体型|混合腺体型|ACR C|ACR c|acr C|acr c|不均匀致密类': "不均匀致密型",
        r'D腺体型|d腺体型|D型|d型|D类|d类|致密腺体型|极度致密型|十分致密型|ACR D|ACR d|acr D|acr d|致密腺体类': "致密型",

        r'欠': "不",
        r'稍': "",
        r'尚': "",
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

    for pattern, replacement in replacement_rules.items():
        text = re.sub(pattern, replacement, text)

    return text


def clean_text(text):
    # 关键：括号内不改动
    return apply_outside_parentheses(text, process_text)
