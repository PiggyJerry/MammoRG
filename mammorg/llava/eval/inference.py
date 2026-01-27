"""
A model worker executes the model.
"""
import os
import json
import math

import torch
import fire
from tqdm import tqdm
from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils_clslabel import build_logger, disable_torch_init, data_loaders
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import re

def remove_spaces_except_birads(text):
    text = re.sub(r'(Bi-Rads)\s+', r'\1<<SPACE>>', text)
    text = text.replace(" ", "")
    text = text.replace("<<SPACE>>", " ")

    return text
def clean_report_text(text):
    text = re.sub(r'2D(\+3D)?显示[:：]', '', text)
    text = re.sub(r'\\{1,}n', '。', text)
    text = re.sub(r'\\{1,}[a-zA-Z]?', '', text)
    
    return text

def eval_model(
        conv_mode: str,
        model_path: str,
        model_base: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = "cuda",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        chunk_idx: int = 0,
        num_chunks: int = 1,
        batch_size: int = 8,
        loader: str = "default",
        group_by_length: bool = False,
    ):
    os.makedirs("logs", exist_ok=True)
    logger = build_logger("model_mimic_cxr", f"logs/model_mimic_cxr_{chunk_idx}.log")



    # load model
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    if not model_name.startswith("finetune-lora"):
        # "llava" needs to be in model_name to correctly load the model.
        raise ValueError(f"Model name {model_name} is not 'finetune-lora'.")
    logger.info(f"Loading the model {model_name} ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device)
    
    batch_prompts = []
    batch_input_ids = []
    batch_images = []
    query={}
    query['conversations']=[
        {
            "from": "human",
            "value": f"<image><image><image><image>\n生成一份乳腺钼靶检查报告，包含Findings和Impression两部分。"
        },
        {
            "from": "gpt",
            "value": ""
        }
    ]
    q = query["conversations"][0]["value"]

    num_images = q.count(DEFAULT_IMAGE_TOKEN)
    q = q.replace("<image>", "").strip()
    if model.config.mm_use_im_start_end:
        q = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)*num_images + '\n' + q
    else:
        q = DEFAULT_IMAGE_TOKEN*num_images + '\n' + q

    conv= conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images={}
    images_path = {
        "R_CC": "path to rcc image",
        "R_MLO": "path to rmlo image",
        "L_CC": "path to lcc image",
        "L_MLO": "path to lmlo image",
    }
    for view in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']:
        if view in images_path:
            image_path = images_path[view]
            image = Image.open(image_path).convert('RGB')
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images[view] = image.half()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    batch_prompts.append(prompt)
    batch_input_ids.append(input_ids)
    batch_images.append(images)
        
    if batch_images[0] is not None:
        images_dict = {
            'R_CC': torch.stack([x['R_CC'] for x in batch_images if x is not None and 'R_CC' in x]),
            'R_MLO': torch.stack([x['R_MLO'] for x in batch_images if x is not None and 'R_MLO' in x]),
            'L_CC': torch.stack([x['L_CC'] for x in batch_images if x is not None and 'L_CC' in x]),
            'L_MLO': torch.stack([x['L_MLO'] for x in batch_images if x is not None and 'L_MLO' in x])
        }
        images_dict = {k: v.cuda() for k, v in images_dict.items()}
    else:
        images_dict = None

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    with torch.inference_mode():
        batch_output_ids = model.generate(
            torch.stack(batch_input_ids).cuda(),
            images=images_dict,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=1024,
            use_cache=True).cpu()
        batch_outputs = tokenizer.batch_decode(
            batch_output_ids[:, len(batch_input_ids[0]):], skip_special_tokens=True
        )

    for outputs in batch_outputs:
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = remove_spaces_except_birads(outputs.strip())
        outputs = clean_report_text(outputs)
        print('Generated-report:', outputs)



if __name__ == "__main__":
    fire.Fire(eval_model)
