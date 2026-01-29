import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json


class Patient_RAG(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
        self.adapter1=nn.Linear(embed_dim*4,768)
        self.norm=nn.LayerNorm(embed_dim)
        self.adapter2=nn.Linear(768,embed_dim)
        with open('/home/user/MammoRG-main/mammorg/llava/model/patient_rag/Train_ChineseBERT_embedding_report.json', 'r') as f:
            self.report_features = torch.from_numpy(np.array(json.load(f)))
        
        
    def get_report_features(self, image_features):
        if isinstance(self.report_features, np.ndarray):
            report_features_tensor = torch.from_numpy(self.report_features).float().to(image_features.device).to(image_features.dtype)
        else:
            report_features_tensor = self.report_features.to(image_features.device).to(image_features.dtype)
        
        image_features_norm = F.normalize(image_features, p=2, dim=-1)
        report_features_norm = F.normalize(report_features_tensor, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(image_features_norm, report_features_norm.T) / math.sqrt(self.embed_dim)
        similarity_matrix=F.softmax(similarity_matrix, dim=-1)
        report_features = torch.matmul(similarity_matrix, report_features_tensor)
        
        return report_features

    def forward(self, images):
        avg_images=[]
        for view in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']:
            avg_images.append(images[view])
        avg_images=torch.cat(avg_images,-1)
        avg_embeds=self.adapter1(avg_images.mean(1,keepdim=True))
        # avg_embeds=torch.stack(avg_embeds,0)
        report_features= self.adapter2(self.get_report_features(avg_embeds))
        return self.norm(report_features)
        
        
    
def build_patient_rag_module(config):
    return Patient_RAG(config.mm_hidden_size)
