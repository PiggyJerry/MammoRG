import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional
import numpy as np

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone() 
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

def f1_score_binary(preds, targets, eps=1e-8):
    """
    preds, targets: 1D tensor, values in {0,1}
    """
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return 2 * precision * recall / (precision + recall + eps)


def f1_score_multiclass(preds, targets, num_classes, ignore_index=None):
    """
    macro-F1
    preds, targets: 1D tensor
    """
    f1s = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = (preds == c)
        targ_c = (targets == c)

        tp = (pred_c & targ_c).sum().float()
        fp = (pred_c & ~targ_c).sum().float()
        fn = (~pred_c & targ_c).sum().float()

        if tp + fp + fn == 0:
            continue

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1s.append(2 * precision * recall / (precision + recall + 1e-8))

    if len(f1s) == 0:
        return torch.tensor(0.0, device=preds.device)

    return torch.stack(f1s).mean()

def compute_aux_f1(preds, labels):
    MAX_DENSITY = 4
    MAX_BIRADS = 9
    MAX_STATUS = 2

    f1_list = []

    device = preds['left_density_logits'].device

    # ---------- density (multiclass) ----------
    for side in ['left', 'right']:
        mask = labels[f'{side}_density_logits'] != MAX_DENSITY
        if mask.any():
            pred_cls = preds[f'{side}_density_logits'][mask].argmax(dim=-1)
            gt_cls = labels[f'{side}_density_logits'][mask]
            f1 = f1_score_multiclass(pred_cls, gt_cls, num_classes=5)
            f1_list.append(f1)

    # ---------- birads (multiclass) ----------
    for side in ['left', 'right']:
        mask = labels[f'{side}_birads_logits'] != MAX_BIRADS
        if mask.any():
            pred_cls = preds[f'{side}_birads_logits'][mask].argmax(dim=-1)
            gt_cls = labels[f'{side}_birads_logits'][mask]
            f1 = f1_score_multiclass(pred_cls, gt_cls, num_classes=11)
            f1_list.append(f1)

    # ---------- status (binary, per-entity) ----------
    for side in ['left', 'right']:
        gt = labels[f'{side}_status_logits']          
        logit = preds[f'{side}_status_logits'] 

        mask = (gt == 0) | (gt == 1)
        if mask.any():
            prob = torch.sigmoid(logit)
            pred = (prob > 0.5).long()
            gt_bin = (gt == 0).long()

            f1 = f1_score_binary(
                pred[mask].view(-1),
                gt_bin[mask].view(-1)
            )
            f1_list.append(f1)

    # ---------- located_at (multi-label binary) ----------
    if 'located_at_logits' in preds:
        prob = torch.sigmoid(preds['located_at_logits'])
        pred = (prob > 0.5).long()
        gt = labels['located_at_logits'].long()

        f1 = f1_score_binary(pred.view(-1), gt.view(-1))
        f1_list.append(f1)

    # ---------- suggestive_of ----------
    if 'suggestive_of_logits' in preds:
        prob = torch.sigmoid(preds['suggestive_of_logits'])
        pred = (prob > 0.5).long()
        gt = labels['suggestive_of_logits'].long()

        f1 = f1_score_binary(pred.view(-1), gt.view(-1))
        f1_list.append(f1)

    # ---------- modified_by ----------
    if 'modified_by_logits' in preds:
        prob = torch.sigmoid(preds['modified_by_logits'])
        pred = (prob > 0.5).long()
        gt = labels['modified_by_logits'].long()

        f1 = f1_score_binary(pred.view(-1), gt.view(-1))
        f1_list.append(f1)

    if len(f1_list) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(f1_list).mean()

from llava.utils import build_logger, disable_torch_init, data_loaders
from llava.conversation import conv_templates, SeparatorStyle
from tqdm import tqdm
import math
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image, ImageFile
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
class LLaVATrainer(Trainer):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.reset_training_stats()
        self.tokenizer = tokenizer 
        self.reset_epoch_buffers()
    def reset_training_stats(self):
        self.total_loss = 0
        self.current_step = 0
        self.nan_detected = False
        self.gradient_norms = []
        self.current_step=1
        self.last_epoch_int = -1
        self.best_f1=0
    
    def reset_epoch_buffers(self):
        self.epoch_graph_preds = []
        self.epoch_sentence_preds = []
        self.epoch_aux_labels = []
    
    def on_epoch_end_logic(self, model):
        if not model.get_model().aux:
            return
        def concat_dict_list(dict_list):
            out = {}
            for k in dict_list[0]:
                out[k] = torch.cat([d[k] for d in dict_list], dim=0)
            return out
        def split_list(lst, n):
            """Split a list into n (roughly) equal-sized chunks"""
            chunk_size = math.ceil(len(lst) / n)  # integer division
            return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


        def get_chunk(lst, n, k):
            chunks = split_list(lst, n)
            return chunks[k]


        def create_batches(data, batch_size, group_by_length, tokenizer):
            if batch_size == 1 or not group_by_length:
                return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
            else:
                batches = []
                batch, batch_len = [], None
                for d in data:
                    d_len = len(tokenizer(d["conversations"][0]['value']).input_ids)
                    if batch_len is None or d_len == batch_len:
                        batch_len = d_len
                        batch.append(d)
                        if len(batch) == batch_size:
                            batches.append(batch)
                            batch, batch_len = [], None
                    else:
                        assert len(batch)
                        batches.append(batch)
                        batch, batch_len = [d], d_len
                if len(batch):
                    batches.append(batch)
                assert len(data) == sum(len(b) for b in batches)
                return batches
        
        image_processor=model.get_model().get_vision_tower().image_processor
        all_queries = data_loaders['MammoReport_test']('/home/user/MammoRG-main/mammorg_data/split_data/Test.json')
        queries = get_chunk(all_queries, 1, 0)

        batches = create_batches(queries, 64, False, self.tokenizer)
        for batch_queries in tqdm(batches):
            batch_prompts = []
            batch_input_ids = []
            batch_images = []
            batch_labels = []
            for query in batch_queries:
                q = query["conversations"][0]["value"]

                num_images = q.count(DEFAULT_IMAGE_TOKEN)
                q = q.replace("<image>", "").strip()
                q = DEFAULT_IMAGE_TOKEN*num_images + '\n' + q

                conv= conv_templates['v1'].copy()
                conv.append_message(conv.roles[0], q)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                if 'Image_paths' in query:
                    images = {}
                    for view in ['R_CC', 'R_MLO', 'L_CC', 'L_MLO']:
                        if view in query['Image_paths']:
                            image_path = os.path.join('/home/user/MammoRG-main/mammorg_data', query['Image_paths'][view])
                            image = Image.open(image_path).convert('RGB')
                            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                            images[view] = image.to(next(model.parameters()).dtype)
                else:
                    images = None
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                batch_prompts.append(prompt)
                batch_input_ids.append(input_ids)
                batch_images.append(images)
                batch_labels.append(query['aux_labels'])
            
            aux_labels = {
          
                'left_density_logits': torch.cat([torch.tensor(inst['left_density_logits']) for inst in batch_labels]), 
                'left_birads_logits': torch.cat([torch.tensor(inst['left_birads_logits']) for inst in batch_labels]),
                'left_status_logits': torch.stack([
                    torch.tensor(inst['left_status_logits']).view(-1)  
                    for inst in batch_labels
                ]),  
                'right_density_logits': torch.cat([torch.tensor(inst['right_density_logits']) for inst in batch_labels]),
                'right_birads_logits': torch.cat([torch.tensor(inst['right_birads_logits']) for inst in batch_labels]),
                'right_status_logits': torch.stack([
                    torch.tensor(inst['right_status_logits']).view(-1)
                    for inst in batch_labels
                ]),
                'located_at_logits': torch.stack([
                    torch.tensor(inst['located_at_logits'])  
                    for inst in batch_labels
                ]),
                'suggestive_of_logits': torch.stack([
                    torch.tensor(inst['suggestive_of_logits'])  
                    for inst in batch_labels
                ]),
                'modified_by_logits': torch.stack([
                    torch.tensor(inst['modified_by_logits']) 
                    for inst in batch_labels
                ]),
            }

            
            if batch_images[0] is not None:
                images_dict = {
                    'R_CC': torch.stack([x['R_CC'] for x in batch_images if x is not None and 'R_CC' in x]),
                    'R_MLO': torch.stack([x['R_MLO'] for x in batch_images if x is not None and 'R_MLO' in x]),
                    'L_CC': torch.stack([x['L_CC'] for x in batch_images if x is not None and 'L_CC' in x]),
                    'L_MLO': torch.stack([x['L_MLO'] for x in batch_images if x is not None and 'L_MLO' in x])
                }
                images_dict = {k: v.cuda() for k, v in images_dict.items()}
            else:
                images_dict = None
            
            with torch.no_grad():
                _ = model(
                    torch.stack(batch_input_ids).cuda(),
                    images=images_dict,
                    aux_labels={k: v.cuda() for k, v in aux_labels.items()})
            self.epoch_graph_preds.append(
                {k: v.detach().cpu() for k, v in model._last_aux_graph_preds.items()}
            )
            self.epoch_sentence_preds.append(
                {k: v.detach().cpu() for k, v in model._last_aux_sentence_preds.items()}
            )
            self.epoch_aux_labels.append(
                {k: v.detach().cpu() for k, v in model._last_aux_labels.items()}
            )

        graph_preds = concat_dict_list(self.epoch_graph_preds)
        sentence_preds = concat_dict_list(self.epoch_sentence_preds)
        aux_labels = concat_dict_list(self.epoch_aux_labels)

        graph_f1 = compute_aux_f1(graph_preds, aux_labels)
        sentence_f1 = compute_aux_f1(sentence_preds, aux_labels)

        total_f1=(graph_f1+sentence_f1)/2
        if total_f1>self.best_f1:
            def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
                to_return = {k: t for k, t in named_params if "lora_" not in k}
                if require_grad_only:
                    to_return = {k: t for k, t in to_return.items() if t.requires_grad}
                to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
                return to_return
            
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            non_lora_state_dict = {'.'.join(k.split('.')[1:]): v for k, v in non_lora_state_dict.items()}
            torch.save(non_lora_state_dict, os.path.join(self.args.output_dir, f'non_lora_trainables.bin'))
            self.best_f1=total_f1
        self.reset_epoch_buffers()
        
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.current_step!=self.state.global_step+1:
            current_epoch_int = int(self.state.epoch)
            if self.args.aux_only:
                if current_epoch_int > self.last_epoch_int:
                    if self.last_epoch_int >= 0:  
                        if (self.last_epoch_int) % 2 == 0: 
                            self.on_epoch_end_logic(model)
                    self.last_epoch_int = current_epoch_int
            self.total_loss+=loss.item()
            average_loss=self.total_loss/self.current_step
            print(f"\nStep {self.state.global_step} Loss: {average_loss}")
            self.current_step=self.state.global_step+1
        return loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
