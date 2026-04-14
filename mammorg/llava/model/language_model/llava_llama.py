#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import json
import torch.nn.functional as F
def compute_aux_loss(preds, labels):
    """
    preds: {
        'left_density_logits': (batch_size, 5),
        'right_density_logits': (batch_size, 5),
        'left_birads_logits': (batch_size, 11),
        'right_birads_logits': (batch_size, 11),
        'left_status_logits': (batch_size, 14, 4),
        'right_status_logits': (batch_size, 14, 4)
    }
    labels: {
        'left_density': (batch_size,),
        'right_density': (batch_size,),
        'left_birads': (batch_size,),
        'right_birads': (batch_size,),
        'left_status': (batch_size, 14),
        'right_status': (batch_size, 14)
    }
    """
    MAX_DENSITY = 4 
    MAX_BIRADS = 9  
    MAX_STATUS = 2 
    
    left_density_mask = (labels['left_density_logits'] != MAX_DENSITY)
    if left_density_mask.any():
        left_density_loss = F.cross_entropy(
            preds['left_density_logits'][left_density_mask],
            labels['left_density_logits'][left_density_mask]
        )
    else:
        left_density_loss = torch.tensor(0.0, device=preds['left_density_logits'].device)
    
    right_density_mask = (labels['right_density_logits'] != MAX_DENSITY)
    if right_density_mask.any():
        right_density_loss = F.cross_entropy(
            preds['right_density_logits'][right_density_mask],
            labels['right_density_logits'][right_density_mask]
        )
    else:
        right_density_loss = torch.tensor(0.0, device=preds['right_density_logits'].device)
    
    density_loss = (left_density_loss + right_density_loss) / 2

    left_birads_mask = (labels['left_birads_logits'] != MAX_BIRADS)
    if left_birads_mask.any():
        left_birads_loss = F.cross_entropy(
            preds['left_birads_logits'][left_birads_mask],
            labels['left_birads_logits'][left_birads_mask]
        )
    else:
        left_birads_loss = torch.tensor(0.0, device=preds['left_birads_logits'].device)

    right_birads_mask = (labels['right_birads_logits'] != MAX_BIRADS)
    if right_birads_mask.any():
        right_birads_loss = F.cross_entropy(
            preds['right_birads_logits'][right_birads_mask],
            labels['right_birads_logits'][right_birads_mask]
        )
    else:
        right_birads_loss = torch.tensor(0.0, device=preds['right_birads_logits'].device)
    
    birads_loss = (left_birads_loss + right_birads_loss) / 2
    
    left_status_labels = labels['left_status_logits']  
    left_status_preds = preds['left_status_logits']  
    left_status_mask = (left_status_labels == 0) | (left_status_labels == 1)

    if left_status_mask.any():
        preds_valid = left_status_preds[left_status_mask]      
        labels_valid = left_status_labels[left_status_mask]     
        
        labels_valid = (labels_valid == 0).float()              

        # BCEWithLogitsLoss
        left_status_loss = F.binary_cross_entropy_with_logits(
            preds_valid,
            labels_valid,
            reduction='mean'
        )
    else:
        left_status_loss = torch.tensor(0.0, device=left_status_preds.device)

    right_status_labels = labels['right_status_logits']
    right_status_preds = preds['right_status_logits']
    right_status_mask = (right_status_labels == 0) | (right_status_labels == 1)

    if right_status_mask.any():
        preds_valid = right_status_preds[right_status_mask]       
        labels_valid = right_status_labels[right_status_mask]    
        
        labels_valid = (labels_valid == 0).float()               

        # BCEWithLogitsLoss
        right_status_loss = F.binary_cross_entropy_with_logits(
            preds_valid,
            labels_valid,
            reduction='mean'
        )
    else:
        right_status_loss = torch.tensor(0.0, device=right_status_preds.device)
    
    status_loss = (left_status_loss + right_status_loss) / 2
    located_at_labels = labels['located_at_logits']  
    located_at_preds = preds['located_at_logits'] 

    located_at_loss = F.binary_cross_entropy_with_logits(
        located_at_preds,  
        located_at_labels.float(),
        reduction='none'
    ).mean() 
    
    suggestive_of_labels = labels['suggestive_of_logits']  
    suggestive_of_preds = preds['suggestive_of_logits']  
    suggestive_of_loss = F.binary_cross_entropy_with_logits(
        suggestive_of_preds, 
        suggestive_of_labels.float(), 
        reduction='none'
    ).mean() 
    
    modified_by_labels = labels['modified_by_logits'] 
    modified_by_preds = preds['modified_by_logits']  
    

    modified_by_loss = F.binary_cross_entropy_with_logits(
        modified_by_preds, 
        modified_by_labels.float(), 
        reduction='none'
    ).mean()  
    num_loss_terms = 0
    total_loss = torch.tensor(0.0, device=preds['left_density_logits'].device)
    
    if density_loss != 0:
        total_loss += density_loss
        num_loss_terms += 1
    
    if birads_loss != 0:
        total_loss += birads_loss
        num_loss_terms += 1
    
    if status_loss != 0:
        total_loss += status_loss
        num_loss_terms += 1
        
    if located_at_loss != 0:
        total_loss += located_at_loss
        num_loss_terms += 1
    
    if suggestive_of_loss != 0:
        total_loss += suggestive_of_loss
        num_loss_terms += 1
        
    if modified_by_loss != 0:
        total_loss += modified_by_loss
        num_loss_terms += 1
    
    if num_loss_terms > 0:
        return total_loss / num_loss_terms
    else:
        return torch.tensor(0.0, device=preds['left_density_logits'].device)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
     
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aux_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.model.aux==True:
            aux_preds_graph, aux_preds_sentence = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            
            self._last_aux_graph_preds = aux_preds_graph
            self._last_aux_sentence_preds = aux_preds_sentence
            self._last_aux_labels = aux_labels
            
            aux_graph_loss = compute_aux_loss(aux_preds_graph,aux_labels)
            aux_sentence_loss = compute_aux_loss(aux_preds_sentence,aux_labels)
            print('graph_loss:',aux_graph_loss.item(), 'aux_sentence_loss:',aux_sentence_loss.item())
            return CausalLMOutputWithPast(
                loss=(aux_graph_loss+aux_sentence_loss)/2,
                logits=None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
