from typing import List
import os
import json 
import random

import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU

import rrg_eval.rouge
from rrg_eval.factuality_utils import CONDITIONS
import sys
sys.path.append("/home/user/MammoRG")
from MammoRGTool.tool import MammoRGTool
try:
    import wandb
except ImportError:
    wandb = None


random.seed(3)
np.random.seed(3)


def bleu4(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        ret = BLEU().corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    else:
        return evaluate.load("bleu").compute(predictions=predictions, references=references)["bleu"]


def bleu1(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        ret = BLEU(max_ngram_order=1).corpus_score(hypotheses=predictions, references=[references], n_bootstrap=500)
        return {"median": ret.score, "ci_l": ret._mean - ret._ci, "ci_h": ret._mean + ret._ci}
    else:
        return evaluate.load("bleu").compute(predictions=predictions, references=references, max_order=1)["bleu"]


def rougel(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rougeL"])["rougeL"]
    else:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rougeL"]


def rouge2(predictions, references, bootstrap_ci: bool = False):
    if bootstrap_ci:
        return rrg_eval.rouge.compute(predictions, references, ["rouge2"])["rouge2"]
    else:
        return evaluate.load("rouge").compute(predictions=predictions, references=references)["rouge2"]


def bertscore(predictions, references):
    return evaluate.load("bertscore").compute(predictions=predictions, references=references)["f1"]

def mammorgscore(predictions, references, tool, bootstrap_ci: bool = False):
    return tool.get_output(preds=predictions, refs=references,calculate_ci=bootstrap_ci)


SCORER_NAME_TO_CLASS = {
    "ROUGE-L": rougel,
    "ROUGE-2": rouge2,
    "BLEU-4": bleu4,
    "BLEU-1": bleu1,
    "BERTScore": bertscore,
    'MammoRGScore': mammorgscore
}


class ReportGenerationEvaluator:
    def __init__(self, scorers=['CheXbert'], bootstrap_ci: bool = False, tool=None):
        self.bootstrap_ci = bootstrap_ci
        self.scorers = {}
        self.tool=tool
        for scorer_name in scorers:
            if scorer_name in SCORER_NAME_TO_CLASS:
                if scorer_name in SCORER_NAME_TO_CLASS: 
                    self.scorers[scorer_name] = SCORER_NAME_TO_CLASS[scorer_name]  
                else:
                    raise NotImplementedError(f'scorer of type {scorer_name} not implemented')

    def evaluate(self, predictions, references):
        assert len(predictions) == len(references), f'Length of predictions (i.e. generations) {len(predictions)} and references (i.e. ground truths) {len(references)} must match.'
        
        scores = {}
        
        for scorer_name, scorer in (pbar := tqdm(self.scorers.items())):
            pbar.set_description(scorer_name)
            if scorer_name=='MammoRGScore':
                scorer_scores = scorer(predictions, references, self.tool, self.bootstrap_ci)
            else:
                scorer_scores = scorer(predictions, references, self.bootstrap_ci)
            scores[scorer_name] = scorer_scores
            
        self.postprocess_eval(scores)
        return scores

    def postprocess_eval(self, scores):
        if self.bootstrap_ci:
            for name in list(scores.keys()):
                if name == "MammoRGScore":
                    metrics = scores.pop(name)
                    scores["composition_f1"] = {'median': metrics['Status_metrics']["composition_f1"],'ci_l': metrics['Status_metrics']["composition_f1_ci"][0], 'ci_h': metrics['Status_metrics']["composition_f1_ci"][1]}
                    scores["birads_f1"] = {'median': metrics['Status_metrics']["birads_f1"],'ci_l': metrics['Status_metrics']["birads_f1_ci"][0], 'ci_h': metrics['Status_metrics']["birads_f1_ci"][1]}
                    scores["finding_f1"] = {'median': metrics['Status_metrics']["finding_f1"],'ci_l': metrics['Status_metrics']["finding_f1_ci"][0], 'ci_h': metrics['Status_metrics']["finding_f1_ci"][1]}
                    scores["relation_f1"] = {'median': metrics['relation_metrics']["f1"],'ci_l': metrics['relation_metrics']["f1_ci"][0], 'ci_h': metrics['relation_metrics']["f1_ci"][1]}

        else:
            for name in list(scores.keys()):
                if name == "MammoRGScore":
                    metrics = scores.pop(name)
                    scores["composition_f1"] = metrics['Status_metrics']["composition_f1"]
                    scores["birads_f1"] = metrics['Status_metrics']["birads_f1"]
                    scores["finding_f1"] = metrics['Status_metrics']["finding_f1"]
                    scores["relation_f1"] = metrics['relation_metrics']["f1"]



def test_evaluator():
    generations = [
        "Totally unrelated.",
        'Lungs and pleural spaces are clear. Cardiomediastinal contour is normal.',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]

    ground_truths = [
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour',
        'The lungs are hyperexpanded with coarse bronchovascular markings in keeping with COPD. There is increased AP diameter and increased retrosternal airspace but the diaphragms have a near normal contour'
    ]
    
    evaluator = ReportGenerationEvaluator()
    print(evaluator.evaluate(generations, ground_truths))

    return


def main(
        filepath: str,
        scorers: List = None,
        report_chexbert_f1: bool = False,
        bootstrap_ci: bool = True,
        output_dir: str = "./",
        run_name: str = "mimic_cxr_eval",
    ):
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath) as f:
        preds, refs = [], []
        for l in f:
            d = json.loads(l)
            preds.append(d["prediction"])
            refs.append(d["reference"])

    if scorers is None:
        scorers = [
            'BLEU-1',
            'BLEU-4',
            'ROUGE-L',
            'MammoRGScore'
        ]
    tool=MammoRGTool(output_dir+'/tool_output.json')
    evaluator = ReportGenerationEvaluator(scorers=scorers, bootstrap_ci=bootstrap_ci,tool=tool)
    results = evaluator.evaluate(preds, refs)
    
    print("\n")
    print(f"Total reports: {len(preds)}\n")

    print("========== Main Results ==========")
    if bootstrap_ci:
        main_results = pd.DataFrame.from_dict({
            k:v for k,v in results.items() if k not in ("breakdown+", "breakdown-", "chexbert_metrics")   
        })
     
        print(main_results[[
            "BLEU-1", "BLEU-4", "ROUGE-L", 'composition_f1', 'birads_f1', 'finding_f1', 'relation_f1'
        ]])
    else:
        main_results = pd.DataFrame.from_dict({k:v for k,v in results.items() if type(v)!= dict}, 'index')
       
        print(main_results.T[[
            "BLEU-1", "BLEU-4", "ROUGE-L", 'composition_f1', 'birads_f1', 'finding_f1', 'relation_f1'
        ]])
    print("")

    main_results.to_csv(os.path.join(output_dir, "main.csv"))

    if wandb:
        wandb_results = {}
        for metric in main_results.columns:
            for index in main_results.index:
                key = metric
                if isinstance(index, str):
                    key += f"-{index}"
                wandb_results[key] = main_results[metric][index]

        wandb.init(name=run_name)
        wandb.log(wandb_results)
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)
