import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class ChineseBERTProcessor:
    def __init__(self, model_name="bert-base-chinese", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        torch.manual_seed(42)

        additional_tokens = [
            "Bi-Rads 0","Bi-Rads 1","Bi-Rads 2","Bi-Rads 3",
            "Bi-Rads 4A","Bi-Rads 4B","Bi-Rads 4C","Bi-Rads 5","Bi-Rads 6"
        ]
        self.tokenizer.add_tokens(additional_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def encode_text(self, text: str, pooling_strategy="cls") -> np.ndarray:
        if not text or text.strip() == "":
            return np.zeros(768)

        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            if pooling_strategy == "mean":
                if last_hidden_state.size(1) > 2:
                    embedding = last_hidden_state[:, 1:-1, :].mean(dim=1)
                else:
                    embedding = last_hidden_state.mean(dim=1)
            else:  # cls pooling
                embedding = last_hidden_state[:, 0, :]

            return embedding.cpu().numpy().flatten()


def process_medical_texts(input_json_path, output_json_path):

    processor = ChineseBERTProcessor()

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    seen_reports = set()
    results = []

    for _, sample_data in tqdm(data.items()):

        try:
            cleaned = sample_data.get("Cleaned_text", {})
            findings = cleaned.get("Findings", "")
            impression = cleaned.get("Impression", "")

            full_report = (findings + " " + impression).strip()

            if full_report in seen_reports:
                continue
            seen_reports.add(full_report)

            embedding = processor.encode_text(full_report, pooling_strategy="cls")
            results.append(embedding.tolist())

        except Exception as e:
            results.append([0.0] * 768)


    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results



if __name__ == "__main__":
    input_json_path = "/home/jiayi/MammoRG-main/mammorg_data/split_data/Train.json"
    output_json_path = "/home/jiayi/MammoRG-main/mammorg/llava/model/patient_rag/Train_ChineseBERT_embedding_report.json"

    results = process_medical_texts(
        input_json_path, output_json_path
    )

