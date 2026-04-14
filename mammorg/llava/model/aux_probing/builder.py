import torch
import torch.nn as nn
import re
from llava.constants import DENSITY_CLASSES, BI_RADS_CLASSES, ENTITY_CLASSES, DIAGNOSIS, ANATOMY, ADJUNCTS, ENTITY_NAMES
class Probing1(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.left_status_classifier = nn.Linear(channels*2, len(ENTITY_NAMES))
        
        self.left_density_classifier = nn.Linear(channels*2, len(DENSITY_CLASSES)-1)
        
        self.left_birads_classifier = nn.Linear(channels*2, len(BI_RADS_CLASSES)-1)
        
        self.right_status_classifier = nn.Linear(channels*2, len(ENTITY_NAMES))
        
        self.right_density_classifier = nn.Linear(channels*2, len(DENSITY_CLASSES)-1)
        
        self.right_birads_classifier = nn.Linear(channels*2, len(BI_RADS_CLASSES)-1)

        self.located_at_classifier = nn.Linear(channels*4, len(ANATOMY)*len(ENTITY_NAMES))
        
        self.suggestive_of_classifier = nn.Linear(channels*4, len(DIAGNOSIS)*len(ENTITY_NAMES))
        
        self.modified_by_classifier = nn.Linear(channels*4, len(ADJUNCTS)*len(ENTITY_NAMES))
        
    def forward(self, x):
        right_breast = torch.cat([x['R_CC'], x['R_MLO']], -1).squeeze(1)
        left_breast = torch.cat([x['L_CC'], x['L_MLO']], -1).squeeze(1)
        whole_breast = torch.cat([x['R_CC'], x['R_MLO'],x['L_CC'], x['L_MLO']], -1).squeeze(1)
        
        left_density_logits = self.left_density_classifier(left_breast)
        left_birads_logits = self.left_birads_classifier(left_breast)
        left_status_logits = self.left_status_classifier(left_breast)
        right_density_logits = self.right_density_classifier(right_breast)
        right_birads_logits = self.right_birads_classifier(right_breast)
        right_status_logits = self.right_status_classifier(right_breast)
        located_at_logits = self.located_at_classifier(whole_breast).view(-1, len(ENTITY_NAMES), len(ANATOMY))
        suggestive_of_logits = self.suggestive_of_classifier(whole_breast).view(-1, len(ENTITY_NAMES), len(DIAGNOSIS))
        modified_by_logits = self.modified_by_classifier(whole_breast).view(-1, len(ENTITY_NAMES), len(ADJUNCTS))
        
        return {
            'left_density_logits': left_density_logits,
            'left_birads_logits': left_birads_logits,
            'left_status_logits': left_status_logits,
            'right_density_logits': right_density_logits,
            'right_birads_logits': right_birads_logits,
            'right_status_logits': right_status_logits,
            'located_at_logits': located_at_logits,
            'suggestive_of_logits': suggestive_of_logits,
            'modified_by_logits': modified_by_logits,
        }
    
class Probing2(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.left_status_classifier = nn.Linear(channels, len(ENTITY_NAMES))
        
        self.left_density_classifier = nn.Linear(channels, len(DENSITY_CLASSES)-1)
        
        self.left_birads_classifier = nn.Linear(channels, len(BI_RADS_CLASSES)-1)
        
        self.right_status_classifier = nn.Linear(channels, len(ENTITY_NAMES))
        
        self.right_density_classifier = nn.Linear(channels, len(DENSITY_CLASSES)-1)
        
        self.right_birads_classifier = nn.Linear(channels, len(BI_RADS_CLASSES)-1)

        self.located_at_classifier = nn.Linear(channels, len(ANATOMY)*len(ENTITY_NAMES))
        
        self.suggestive_of_classifier = nn.Linear(channels, len(DIAGNOSIS)*len(ENTITY_NAMES))
        
        self.modified_by_classifier = nn.Linear(channels, len(ADJUNCTS)*len(ENTITY_NAMES))
        
    def forward(self, x):
        x=x.squeeze(1)
        left_density_logits = self.left_density_classifier(x)
        left_birads_logits = self.left_birads_classifier(x)
        left_status_logits = self.left_status_classifier(x)
        right_density_logits = self.right_density_classifier(x)
        right_birads_logits = self.right_birads_classifier(x)
        right_status_logits = self.right_status_classifier(x)
        located_at_logits = self.located_at_classifier(x).view(-1, len(ENTITY_NAMES), len(ANATOMY))
        suggestive_of_logits = self.suggestive_of_classifier(x).view(-1, len(ENTITY_NAMES), len(DIAGNOSIS))
        modified_by_logits = self.modified_by_classifier(x).view(-1, len(ENTITY_NAMES), len(ADJUNCTS))
        
        return {
            'left_density_logits': left_density_logits,
            'left_birads_logits': left_birads_logits,
            'left_status_logits': left_status_logits,
            'right_density_logits': right_density_logits,
            'right_birads_logits': right_birads_logits,
            'right_status_logits': right_status_logits,
            'located_at_logits': located_at_logits,
            'suggestive_of_logits': suggestive_of_logits,
            'modified_by_logits': modified_by_logits,
        }
        
class DualProbing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.probing1 = Probing1(config.mm_hidden_size)
        self.probing2 = Probing2(config.mm_hidden_size)
    def forward(self,x1,x2):
        return self.probing1(x1), self.probing2(x2)
        
    
def build_probing(config):
    return DualProbing(config)