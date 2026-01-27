# 🏥 MammoRG
## Introduction

Official implementation of MammoRG, introduced in "Cross-Modal Knowledge Retrieval for Mammography Report Generation".

MammoRG is capable of taking four-view mammogram images of patients (RCC, RMLO, LCC, LMLO) and generating corresponding reports.

## Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Train](#train)
  - [0. Preparation](#0-preparation)
  - [1. Pretrain (Alignment)](#1-pretrain-alignment)
  - [2. Fine-tuning (LoRA)](#2-fine-tuning-lora)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License and Usage Notices](#license-and-usage-notices)
- [Acknowledgements](#acknowledgements)

## Requirements
```shell
git clone https://github.com/PiggyJerry/MammoRG.git

cd MammoRG
conda create -n mammorg python==3.9
conda activate mammorg

python -m pip install -r requirements.txt
```

## Train
When starting from scratch, the following checkpoints are needed:
- A pre-trained LLaVA-Mammo, please download the weight from [A Benchmark for Breast Cancer Screening and Diagnosis in Mammogram Visual Question Answering](https://drive.google.com/file/d/1uFCrOTbsvug8YZoHKR7wlvoTSwzB32EY/view?usp=sharing)
- A pre-trained vision backbone VersaMammo, please download the weight from [A Versatile Foundation Model for AI-enabled Mammogram Interpretation](https://drive.google.com/file/d/1HmEzoJDs99-t6_mUnrjnkcY8nTJ8WeVp/view?usp=sharing)

### 0. Preparation
Before running the commands below, you need to have the data, image folder, and the above checkpoints ready. 

**0.1 Data**

Since our training set is a private dataset and cannot be made public, if you wish to use your own dataset, you can process the data into the following format:
````
{
   "1": {
      "Data_source": "Could be dataset's name",
      "ID": "Could be sample's name",
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

**0.2 Images**

You need to download the [MIMIC-CXR-JPG images from PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) by signing the data use agreement and following the instructions.

**0.3 Model weights**

You can find the pretrained model weights for BiomedCLIP-CXR and LLaVA-Rad at https://huggingface.co/microsoft/llava-rad.


**Notes before proceeding:** 
- Change the paths in the scripts below according to where you downloaded the data.
- Batch size is set for 4-GPU machines. If your machine has a difference number of GPUs, please change batch size. Training commands have been tested on a single 80GB A100 and 4x80GB H100, using torch 2.4.1 and cuda 11.8 with flash attention 2.7.2.post1.

### 1. Pretrain (Alignment)
At this stage, we only train the projection layer (which aligns the vision features with text features). The vision encoder and LLM are all frozen.

```bash
bash scripts/pretrain.sh
```

We get a pretrained projector `mm_projector.bin` after pretraining.

### 2. Fine-tuning (LoRA)
Once we have a pretrained projector, we can do fine-tuning. The command below fine-tunes the projector and LoRA of LLM:
```bash
bash scripts/finetune_lora.sh
```

## Inference

Before running the command below, you need to change the script accordingly.

```bash
bash scripts/eval.sh
```

**Note:** To reproduce the evaluation results from the manuscript on the MIMIC-CXR dataset, changing the script means uncommenting and updating the paths for `query_file` and `image_folder`.

In the manuscript, the Open-I and CheXpert chest X-ray images and reports are also used for evaluation. These datasets are available at their corresponding sources: [Open-I](https://openi.nlm.nih.gov/faq) | [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1).

## Evaluation

If you have run inference using multiple GPUs and have a resulting set of chunks with results, make sure you concatenate prediction chunks into a single file before running the following command:
```bash
cd llava/eval/rr_eval
python run.py ${YOUR_PREDICTION_FILE}
```

## Citation

```bibtex

@Article{ZambranoChaves2025,
author={Zambrano Chaves, Juan Manuel and Huang, Shih-Cheng and Xu, Yanbo and Xu, Hanwen and Usuyama, Naoto and Zhang, Sheng and Wang, Fei and Xie, Yujia and Khademi, Mahmoud and Yang, Ziyi and Awadalla, Hany and Gong, Julia and Hu, Houdong and Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng and Gu, Yu and Wong, Cliff and Wei, Mu and Naumann, Tristan and Chen, Muhao and Lungren, Matthew P. and Chaudhari, Akshay and Yeung-Levy, Serena and Langlotz, Curtis P. and Wang, Sheng and Poon, Hoifung},
title={A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings},
journal={Nature Communications},
year={2025},
month={Apr},
day={01},
volume={16},
number={1},
pages={3108},
issn={2041-1723},
doi={10.1038/s41467-025-58344-x},
url={https://doi.org/10.1038/s41467-025-58344-x}
}

```

## License and Usage Notices

The data, code, and model checkpoints are licensed and intended for research use only. The code and model checkpoints are subject to additional restrictions as determined by the Terms of Use of LLaMA, Vicuna, and GPT-4 respectively. Code and model checkpoints may be used for research purposes and should not be used in direct clinical care or for any clinical decision making purpose.

## Acknowledgements

Our codebase heavily relies on [LLaVA](https://github.com/haotian-liu/LLaVA) v1.5. Please check out their repo for more information, and consider citing them in addition to our manuscript if you use this codebase.

```bibtex

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

```
