from torch import nn
from transformers import *
import torch

class RelModel(nn.Module):
    def __init__(self, config):
        super(RelModel, self).__init__()
        self.config = config
        self.bert_dim = config.bert_dim

        self.bert_encoder = BertModel.from_pretrained("bert-base-chinese")
        
        new_pos_length = 1024
        original_config = self.bert_encoder.config
        original_config.max_position_embeddings = new_pos_length
        original_pos_weights = self.bert_encoder.embeddings.position_embeddings.weight.data
        new_pos_embeddings = torch.nn.Embedding(new_pos_length, original_config.hidden_size)
        original_pos = original_pos_weights.unsqueeze(0).permute(0, 2, 1)
        new_pos = torch.nn.functional.interpolate(original_pos, size=new_pos_length, mode='linear')
        new_pos_embeddings.weight.data = new_pos.squeeze(0).permute(1, 0)
        self.bert_encoder.embeddings.position_embeddings = new_pos_embeddings
        self.bert_encoder.embeddings.register_buffer("position_ids", torch.arange(new_pos_length).expand((1, -1)))
        if hasattr(self.bert_encoder.embeddings, 'token_type_ids'):
            del self.bert_encoder.embeddings.token_type_ids
        self.bert_encoder.embeddings.register_buffer("token_type_ids", torch.zeros([1, new_pos_length], dtype=torch.long), persistent=False)

        self.entity_embeddings = nn.Parameter(torch.randn((2+10)*2, self.bert_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.bert_dim, num_heads=8)

        self.left_state_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 4) 
        )
        
        self.left_density_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 5) 
        )
        
        self.left_birads_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 10) 
        )
        
        self.right_state_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 4) 
        )
        
        self.right_density_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 5)
        )
        
        self.right_birads_classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(self.bert_dim, 10) 
        )
        
        self.relation_matrix = nn.Linear(self.bert_dim * 3, self.config.rel_num * self.config.tag_size)
        self.projection_matrix = nn.Linear(self.bert_dim * 2, self.bert_dim * 3)
        self.dropout = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def get_encoded_text(self, token_ids, mask):
        return self.bert_encoder(token_ids, attention_mask=mask)[0]

    def predict_entity_states(self, encoded_text):
        batch_size = encoded_text.size(0)
        query = self.entity_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        key = value = encoded_text.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        attn_output, _ = self.cross_attn(
            query=query, 
            key=key,     
            value=value  
        ) 
        attn_output = attn_output.permute(1, 0, 2)

        left_attnout,right_attnout=attn_output.chunk(2,1)
        left_density,left_birads,left_state=left_attnout.split([1,1,10],1)
        right_density,right_birads,right_state=right_attnout.split([1,1,10],1)
        
        left_density_logits = self.left_density_classifier(left_density)
        left_birads_logits = self.left_birads_classifier(left_birads)
        left_state_logits = self.left_state_classifier(left_state) 
        
        right_density_logits = self.right_density_classifier(right_density)
        right_birads_logits = self.right_birads_classifier(right_birads)
        right_state_logits = self.right_state_classifier(right_state)
        return {'left_density_logits':left_density_logits,
                'left_birads_logits':left_birads_logits,
                'left_state_logits':left_state_logits,
                'right_density_logits':right_density_logits,
                'right_birads_logits':right_birads_logits,
                'right_state_logits':right_state_logits,}

    def triple_score_matrix(self, encoded_text, train=True):
        batch_size, seq_len, bert_dim = encoded_text.size()
        head_representation = encoded_text.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len*seq_len, bert_dim)
        tail_representation = encoded_text.repeat(1, seq_len, 1)
        entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)
        entity_pairs = self.projection_matrix(entity_pairs)
        entity_pairs = self.dropout_2(entity_pairs)
        entity_pairs = self.activation(entity_pairs)
        triple_scores = self.relation_matrix(entity_pairs).reshape(batch_size, seq_len, seq_len, self.config.rel_num, self.config.tag_size)
        
        if train:
            return triple_scores.permute(0,4,3,1,2)
        else:
            return triple_scores.argmax(dim=-1).permute(0,3,1,2)

    def forward(self, data, train=True):
        token_ids = data['token_ids']
        mask = data['mask']
        
        encoded_text = self.get_encoded_text(token_ids, mask)
        encoded_text = self.dropout(encoded_text)

        rel_output = self.triple_score_matrix(encoded_text, train)

        entity_output = self.predict_entity_states(encoded_text)
        
        return {
            'relation_output': rel_output, 
            'entity_output': entity_output    
        }