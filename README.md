# 🏥 MammoRG
## Introduction

Official implementation of MammoRG, introduced in "Cross-Modal Clinical Knowledge Integration for Mammography Report Generation".

MammoRG is capable of taking four-view mammogram images of patients (RCC, RMLO, LCC, LMLO) and generating corresponding reports.

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [MammoRGTool](#mammorgtool)


## Requirements
```shell
git clone https://github.com/PiggyJerry/MammoRG.git

cd MammoRG
conda create -n mammorg python==3.9
conda activate mammorg

pip install torch torchvision
python -m pip install -r requirements.txt

# Install FlashAttention
pip install flash-attn --no-build-isolation

# Install PyTorch Geometric
pip install torch_geometric
```
## Preparation
- A pre-trained LLaVA-Mammo, please download the weight from [A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram Visual Question Answering](https://drive.google.com/file/d/1uFCrOTbsvug8YZoHKR7wlvoTSwzB32EY/view?usp=sharing) and unzip it, then rename the forder to `LLaVA-Mammo-checkpoint`. Remember to change the related path in the code.
- A pre-trained vision backbone VersaMammo, please download the weight from [A Versatile Foundation Model for AI-enabled Mammogram Interpretation](https://drive.google.com/file/d/1HmEzoJDs99-t6_mUnrjnkcY8nTJ8WeVp/view?usp=sharing). Remember to change the related path in the code.
- Please prepare your own report database using the [code](https://github.com/PiggyJerry/MammoRG/blob/main/mammorg/llava/model/patient_rag/generate_report_database.py).

## Train
Before running the commands below, you need to have the data, and the above preparation ready. 

**Data**

Since our training set is a private dataset and cannot be made public, if you wish to use your own dataset.
First, you need to organize your dataset into an Excel file. It must include the following four columns: ID, Findings, Impression, and image_paths.
| ID | Findings | Impression | image_paths |
|--------------|--------------|------------|--------|
| 123456 | "..." | "..." | "R_CC": ""path to the rcc image", "R_MLO": "", "L_CC": "", "L_MLO": "" |

You can use the [processing code](https://github.com/PiggyJerry/MammoRG/blob/main/MammoRGTool/generate_data.py) to process the excel file into the following format:
````
{
   "1": {
      "Data_source": "Could be dataset's name",
      "ID": "Could be sample's name",
      "Original_text": {
         "Findings": "...",
         "Impression": "..."
      },
      "Cleaned_text": {
         "Findings": "...",
         "Impression": "..."
      },
      "Image_paths": {
         "R_CC": "path to the rcc image",
         "R_MLO": "path to the rmlo image",
         "L_CC": "path to the lcc image",
         "L_MLO": "path to the lmlo image",
      },
      "Breast_assessment": {
         "Left_breast": {
            "Density": "...",
            "Bi-Rads": "...",
            "Entities": {
               "钙化": "...",
               "肿块": "...",
               "乳腺增生": "...",
               "皮肤增厚": "...",
               "淋巴结肿大": "...",
               "乳头凹陷": "...",
               "结构扭曲": "...",
               "悬韧带增粗": "...",
               "结节": "...",
               "结构不对称": "..."
            }
         },
         "Right_breast": {
            "Density": "...",
            "Bi-Rads": "...",
            "Entities": {
               "钙化": "...",
               "肿块": "...",
               "乳腺增生": "...",
               "皮肤增厚": "...",
               "淋巴结肿大": "...",
               "乳头凹陷": "...",
               "结构扭曲": "...",
               "悬韧带增粗": "...",
               "结节": "...",
               "结构不对称": "..."
            }
         }
      },
      "Relations": [
         [
            Triplet 1
         ],
         [
            Triplet 2
         ],
         ...
      ]
   },
   "2": {
      ...
   },
   ...
}
````
**Notes before proceeding** 
- Change the paths in the scripts according to where you output the data.
- Change the `mm_vision_tower_checkpoint' and `mm_vision_tower_config' in `checkpoints/stage2_finetune/config.json' to the path of `VersaMammo.pth' and `versamammo.json' under `llava/model/multimodal_encoder'
- Change the path of the report database file in `llava/model/patient_rag/builder.py' to your file path
- Change the path of the KG file in `llava/model/graph_model/builder.py' to your file path or ours
- Change the path of `LLaVA-Mammo-checkpoint' in `llava/model/builder.py' (58 line) and `llava/train/train.py' (943 line)
- Change the path in `llava/eval/rrg_eval/run.py' (15 line)
- Change the path of the Test file in `llava/train/llava_trainer.py' (319 line) and the image folder path in `llava/train/llava_trainer.py' (344 line)

### Start training
```bash
cd mammorg
bash scripts/main.sh
```

## Evaluation
Before running the command below, you need to change the script accordingly.

```bash
cd mammorg
bash scripts/eval.sh
```

## Inference with only images
Please download the MammoRG [checkpoint](https://drive.google.com/drive/folders/14iz6pWb5FkGLvAZ0_iOo2IuYdva4T6iT?usp=sharing) first.

Before running the command below, you need to change the script and [inference](https://github.com/PiggyJerry/MammoRG/blob/main/mammorg/llava/eval/inference.py) accordingly.
```bash
cd mammorg
bash scripts/inference.sh
```
## MammoRGTool
Please download the MammoRGTool [checkpoint](https://drive.google.com/drive/folders/1KrXdk7jjvYXFL2K18i7hYnq7x8U8eItN?usp=sharing) first and put the downloaded folder `/checkpoint` under `/MammoRGTool`.

Here is an example for how to use MammoRGTool to evaluate:
```shell
cd MammoRGTool
python tool.py
```

## Mammography Lexicon and Relation Schema

MammoRG uses a predefined mammography lexicon to support structured report parsing, knowledge-graph construction, and term-aware tokenization. The current lexicon contains terms related to breast composition, BI-RADS assessment, abnormal findings, anatomical locations, imaging descriptors, and diagnostic suggestions.

### Lexicon Categories

| Category | Number of terms | Terms |
|---|---:|---|
| Breast composition | 4 | 脂肪型, 纤维腺体型, 不均匀致密型, 致密型 |
| Diagnostic suggestion | 9 | BI-RADS 0, BI-RADS 1, BI-RADS 2, BI-RADS 3, BI-RADS 4A, BI-RADS 4B, BI-RADS 4C, BI-RADS 5, BI-RADS 6, 乳腺癌 |
| Abnormal findings | 10 | 钙化, 肿块, 乳腺增生, 皮肤增厚, 淋巴结肿大, 乳头凹陷, 结构扭曲, 悬韧带增粗, 结节, 结构不对称 |
| Anatomical locations | 18 | 左乳外上象限, 双乳, 左侧中央区, 右侧中央区, 右乳外下象限, 右乳, 左乳, 右乳外上象限, 右乳内上象限, 右乳内下象限, 左乳内下象限, 左乳外下象限, 左乳内上象限, 左侧乳晕区, 右侧乳晕区, 右侧腋下, 双侧腋下, 左侧腋下 |
| Imaging descriptors | 28 | 圆形, 密度均匀, 分叶状, 局部不规则, 颗粒状, 壳样, 类圆形, 密度增高, 边缘模糊, 边缘清晰, 部分边缘模糊, 卵圆形, 模糊, 密度不均匀, 线虫样, 结节状, 点状, 不规则, 边缘不规则, 局部, 密度增高且不均匀, 粗糙不均质, 部分边缘清晰, 局灶性, 毛刺影, 斑点状, 簇状, 模糊不定形 |

### Relation Schema

MammoRG uses three predefined relation types to represent structured semantic associations among mammography concepts:

| Relation type | Description | Head entity type | Tail entity type |
|---|---|---|---|
| `Located_at` | Associates an abnormal finding with its anatomical location. | Abnormal finding | Anatomical location |
| `Modified_by` | Associates an abnormal finding with its imaging descriptor or morphological characteristic. | Abnormal finding | Imaging descriptor |
| `Suggestive_of` | Associates an abnormal finding with a diagnostic assessment or diagnostic suggestion. | Abnormal finding | BI-RADS category or diagnostic suggestion |

Each relation is represented as an exact triplet:

```text
[head entity, relation type, tail entity]
```

## Citation
If you use this code or models in your scientific work, please kindly cite our paper: 
```bibtex
@article{zhu2026cross,
  title={Cross-Modal Clinical Knowledge Integration for Mammography Report Generation},
  author={Zhu, Jiayi and Huang, Fuxiang and Xie, Yu and Wang, Xi and Chen, Zhixuan and Guo, Yuan and Kong, Qingcong and Li, Zhenhui and Luo, Qiong and Chen, Hao},
  journal={arXiv preprint arXiv:2605.31093},
  year={2026}
}
