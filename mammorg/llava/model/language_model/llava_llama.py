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
import re
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from llava.mm_utils import tokenizer_image_token
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import json
import torch.nn.functional as F
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from .GRPO import *
from PIL import Image
import os
import sys
def remove_spaces_except_birads(text):
    text = re.sub(r'(Bi-Rads)\s+', r'\1<<SPACE>>', text)
    text = text.replace(" ", "")
    text = text.replace("<<SPACE>>", " ")
    return text
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
        self.config=config
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tool = None

        self.reference_model = None
        self.ref_lm_head = None
        self.post_init()
     
    def get_model(self):
        return self.model
    def _run_model_and_head(self, model, lm_head, input_ids, attention_mask, past_key_values, inputs_embeds, **kwargs):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        hidden_states = outputs[0]
        logits = lm_head(hidden_states)
        return outputs, hidden_states, logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        aux_labels: Optional[torch.LongTensor] = None,
        origin_infos: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        device = input_ids.device if input_ids is not None else next(self.parameters()).device
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.model.aux==True:
            aux_preds_graph, aux_preds_report = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            
            self._last_aux_graph_preds = aux_preds_graph
            self._last_aux_report_preds = aux_preds_report
            self._last_aux_labels = aux_labels
            
            aux_graph_loss = compute_aux_loss(aux_preds_graph,aux_labels)
            aux_report_loss = compute_aux_loss(aux_preds_report,aux_labels)
            print('graph_loss:',aux_graph_loss.item(), 'aux_report_loss:',aux_report_loss.item())
            return CausalLMOutputWithPast(
                loss=(aux_graph_loss+aux_report_loss)/2,
                logits=None,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

            logits=None
            

            if labels is None or self.reference_model is None:
                outputs, hidden_states, logits = self._run_model_and_head(
                    self.model, self.lm_head,
                    input_ids, attention_mask, past_key_values, inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )

                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
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

            if self.reference_model is not None and labels is not None:
                conv_mode = 'v1'
                image_folder = '/home/jiayi/MammoRG/mammorg_data'
                temperature, top_p, num_beams, num_generations = 0.9, 0.9, 1, 4
                
                prompts, report_labels, batch_images = [], [], []
                for origin_info in origin_infos:
                    q, lbl = origin_info["prompt"], origin_info["label"]
                    report_labels.append(lbl)

                    num_images = q.count(DEFAULT_IMAGE_TOKEN)
                    q = q.replace("<image>", "").strip()
                    if self.model.config.mm_use_im_start_end:
                        q = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) * num_images + '\n' + q
                    else:
                        q = DEFAULT_IMAGE_TOKEN * num_images + '\n' + q

                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], q)
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt())

                    if 'image_paths' in origin_info:
                        images = {
                            view: self.model.get_vision_tower().image_processor.preprocess(
                                Image.open(os.path.join(image_folder, origin_info['image_paths'][view])).convert('RGB'),
                                return_tensors='pt'
                            )['pixel_values'][0].to(inputs_embeds.dtype)
                            for view in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']
                            if view in origin_info['image_paths']
                        }
                    else:
                        images = None
                    batch_images.append(images)

                prompt_ids = torch.stack([tokenizer_image_token(p, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]).to(device)
                if batch_images[0] is not None:
                    images_dict = {view: torch.stack([x[view] for x in batch_images if x and view in x]).to(device)
                                for view in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']}
                else:
                    images_dict = None

                # ===== rollout =====
                batch_size = prompt_ids.size(0)
                prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
                prompt_mask = torch.ones_like(prompt_ids, device=device)
                if images_dict is not None:
                    images_dict = {k: v.repeat_interleave(num_generations, dim=0) for k, v in images_dict.items()}

                report_labels = sum([[lbl] * num_generations for lbl in report_labels], [])

                original_requires_grad = {p: p.requires_grad for p in self.parameters()}
                current_model = self.eval()
                for p in current_model.parameters():
                    p.requires_grad = False
                with torch.inference_mode():
                    batch_output_ids = current_model.generate(
                        prompt_ids,
                        images=images_dict,
                        do_sample=(temperature > 0),
                        temperature=temperature,
                        top_p=top_p,
                        num_beams=num_beams,
                        max_new_tokens=1024, 
                        use_cache=True  
                    )
                
                self.train()
                for p, req_grad in original_requires_grad.items():
                    p.requires_grad = req_grad
                prompt_len = prompt_ids.size(1)
                completion_ids = batch_output_ids[:, prompt_len:]
                completion_mask = create_completion_mask(completion_ids, self.tokenizer.eos_token_id)

                full_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                full_mask = torch.cat([prompt_mask, completion_mask], dim=1)

                prepared_input_ids, prepared_attention_mask, prepared_past_key_values, prepared_inputs_embeds, _ = \
                    self.prepare_inputs_labels_for_multimodal(full_input_ids, full_mask, None, None, images_dict)

                with torch.no_grad():
                    old_outputs, old_hidden, old_logits = self._run_model_and_head(
                        self.model, self.lm_head,
                        prepared_input_ids, prepared_attention_mask, prepared_past_key_values, prepared_inputs_embeds,
                        use_cache=use_cache, output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states, return_dict=return_dict
                    )

                    ref_outputs, ref_hidden, ref_logits = self._run_model_and_head(
                        self.reference_model, self.ref_lm_head,
                        prepared_input_ids, prepared_attention_mask, prepared_past_key_values, prepared_inputs_embeds,
                        use_cache=use_cache, output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states, return_dict=return_dict
                )
            
                # ===== log_probs & RL Loss =====
                logits_to_keep = completion_ids.size(1)
                old_log_probs = selective_log_softmax(old_logits[:, -logits_to_keep:, :], full_input_ids[:, -logits_to_keep:])
                ref_log_probs = selective_log_softmax(ref_logits[:, -logits_to_keep:, :], full_input_ids[:, -logits_to_keep:])
                
                decoded_texts = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                decoded_texts=[remove_spaces_except_birads(decoded_text) for decoded_text in decoded_texts]
                print('decoded_texts:',decoded_texts)
                report_labels=[remove_spaces_except_birads(report_label) for report_label in report_labels]

                metrics = self.tool.get_output(decoded_texts, report_labels)['Per_sample_f1']
                rewards = torch.tensor([
                    sum(x for x in [m['density_f1'], m['birads_f1'], m['entities_f1'], m['relations_f1']] if x is not None) / 
                    max(1, sum(1 for x in [m['density_f1'], m['birads_f1'], m['entities_f1'], m['relations_f1']] if x is not None))
                    for m in metrics
                ], dtype=torch.float32, device=device).view(batch_size, num_generations)
                mean_rewards = rewards.mean(dim=1, keepdim=True)
                std_rewards = rewards.std(dim=1, keepdim=True) + 1e-4
                advantages = ((rewards - mean_rewards) / std_rewards).view(-1, 1)

                rollout_cache = {
                    "full_input_ids": full_input_ids,   # 原始拼接的 ids（可选保留）
                    "full_mask": full_mask,
                    "images_dict": images_dict,
                    "completion_mask": completion_mask,
                    "old_log_probs": old_log_probs,
                    "ref_log_probs": ref_log_probs,
                    "advantages": advantages,
                    "rewards": rewards,
                }
                return rollout_cache
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
