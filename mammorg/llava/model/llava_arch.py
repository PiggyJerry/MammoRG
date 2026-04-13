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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from .aux_probing.builder import build_probing
from .graph_model.builder import build_graph_model
from .cross_module.builder import build_cross_module
from .patient_rag.builder import build_patient_rag_module
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.aux=False
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector1,self.mm_projector2 = build_vision_projector(config)
            self.probing = build_probing(config)
            self.graph_model = build_graph_model(config)
            self.cross_module = build_cross_module(config)
            self.patient_rag = build_patient_rag_module(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, tokenizer=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_config = model_args.vision_tower_config
        self.config.mm_vision_tower_checkpoint = model_args.vision_tower_checkpoint

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        print(self.config)
        self.mm_projector1, self.mm_projector2 = build_vision_projector(self.config)
        self.probing = build_probing(self.config)
        self.graph_model = build_graph_model(self.config)
        self.cross_module = build_cross_module(self.config)
        self.patient_rag = build_patient_rag_module(self.config)



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images,graph_features):
        dtype = next(self.get_model().parameters()).dtype
        images = images.to(dtype=dtype)
        image_features = self.get_model().get_vision_tower()(images)
        image_features, graph_features = self.get_model().cross_module(image_features,graph_features.to(image_features.dtype))
        
        project_image_features,project_graph_features = self.get_model().mm_projector1(image_features,graph_features)
        
        return image_features,graph_features.squeeze(1),project_image_features,project_graph_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        
        image_features = {}
        graph_features = {}
        backbone_graph_features = {}
        backbone_image_features = {}
        

        graph_feature = self.get_model().graph_model()
        for view_name in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']:
            if view_name not in images:
                raise ValueError(f"Missing required view: {view_name}")
            img_batch = images[view_name]
            if isinstance(img_batch, list) or img_batch.ndim == 5:
                concat_images = torch.cat([img for img in img_batch], dim=0)
                image_feature, aux_graph_feature, project_image_features,project_graph_features = self.encode_images(concat_images,graph_feature)
                split_sizes = [img.shape[0] for img in img_batch]
                backbone_image_features[view_name] = [f.flatten(0, 1) for f in torch.split(image_feature, split_sizes, dim=0)]
                backbone_graph_features[view_name] = [f.flatten(0, 1) for f in torch.split(aux_graph_feature, split_sizes, dim=0)]
                image_features[view_name] = [f.flatten(0, 1) for f in torch.split(project_image_features, split_sizes, dim=0)]
                graph_features[view_name] = [f.flatten(0, 1) for f in torch.split(project_graph_features, split_sizes, dim=0)]
            
            else:
                backbone_image_features[view_name], backbone_graph_features[view_name],  image_features[view_name], graph_features[view_name]= self.encode_images(img_batch,graph_feature)
        report_features = self.get_model().patient_rag(backbone_image_features)
        aux_preds_graph, aux_preds_report = self.get_model().probing(backbone_graph_features,report_features)
        if self.get_model().aux==True:
            return aux_preds_graph, aux_preds_report
        report_features=self.get_model().mm_projector2(report_features)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        batch_size = len(input_ids)  

        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                placeholder_features = image_features['R_CC'][cur_image_idx][0:0]
                cur_input_embeds = torch.cat([
                    self.get_model().embed_tokens(cur_input_ids[:half_len]),
                    placeholder_features,
                    self.get_model().embed_tokens(cur_input_ids[half_len:])
                ], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 4
                print('no images')
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            view_image_features = {
                    'R_CC': image_features['R_CC'][batch_idx],
                    'R_MLO': image_features['R_MLO'][batch_idx],
                    'L_CC': image_features['L_CC'][batch_idx],
                    'L_MLO': image_features['L_MLO'][batch_idx]
                }
            view_graph_features = {
                    'R_CC': graph_features['R_CC'][batch_idx],
                    'R_MLO': graph_features['R_MLO'][batch_idx],
                    'L_CC': graph_features['L_CC'][batch_idx],
                    'L_MLO': graph_features['L_MLO'][batch_idx]
                }
          
                
            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_indices[0]]))
            cur_new_input_embeds.append(view_graph_features['R_CC'])
            cur_new_input_embeds.append(view_image_features['R_CC'])
            cur_new_input_embeds.append(view_graph_features['R_MLO'])
            cur_new_input_embeds.append(view_image_features['R_MLO'])
            cur_new_input_embeds.append(view_graph_features['L_CC'])
            cur_new_input_embeds.append(view_image_features['L_CC'])
            cur_new_input_embeds.append(view_graph_features['L_MLO'])
            cur_new_input_embeds.append(view_image_features['L_MLO'])
            cur_new_input_embeds.append(report_features[batch_idx])
            if labels is not None:
                cur_new_labels.append(cur_labels[:image_token_indices[0]])
                cur_new_labels.append(torch.full((view_image_features['R_CC'].shape[0]+view_graph_features['R_CC'].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                cur_new_labels.append(torch.full((view_image_features['R_MLO'].shape[0]+view_graph_features['R_MLO'].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                cur_new_labels.append(torch.full((view_image_features['L_CC'].shape[0]+view_graph_features['L_CC'].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                cur_new_labels.append(torch.full((view_image_features['L_MLO'].shape[0]+view_graph_features['L_MLO'].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                cur_new_labels.append(torch.full((report_features[batch_idx].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                
                
                cur_labels = cur_labels[image_token_indices[-1]+1:]

            if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                cur_input_ids = cur_input_ids[image_token_indices[-1]+2:]
            else:
                cur_input_ids = cur_input_ids[image_token_indices[-1]+1:]
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
        self.tokenizer=tokenizer
        
