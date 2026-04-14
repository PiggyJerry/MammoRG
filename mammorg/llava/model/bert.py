from transformers import BertModel, BertTokenizer
import torch

class BertEmbeddingExtractor:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name,torch_dtype=torch.float32,
            device_map=None  
        )
        torch.manual_seed(42)
            
        # Add medical English terms to tokenizer
        additional_tokens = ["Bi-Rads 0", "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3",  
        "Bi-Rads 4A", "Bi-Rads 4B", "Bi-Rads 4C", "Bi-Rads 5", "Bi-Rads 6"]
        self.tokenizer.add_tokens(additional_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to('cpu')       
        self.model.eval()
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)
    
    def batch_get_embeddings(self, texts):
        return torch.stack([self.get_embedding(text) for text in texts])