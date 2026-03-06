# 🏥 MammoRG
## Introduction

Official implementation of MammoRG, introduced in "Cross-Modal Knowledge Retrieval for Mammography Report Generation".

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
- Change the `mm_vision_tower_checkpoint' and `mm_vision_tower_config' in `checkpoints/stage3/config.json' to the path of `VersaMammo.pth' and `versamammo.json' under `llava/model/multimodal_encoder'
- Change the path of the report database file in `llava/model/patient_rag/builder.py' to your file path
- Change the path of the KG file in `llava/model/graph_model/builder.py' to your file path or ours
- Change the path of `LLaVA-Mammo-checkpoint' in `llava/model/builder.py' (58 line) and `llava/train/train.py' (943 line) 

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
