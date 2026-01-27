from .GCN import GraphModel
import json
import torch
from torch_geometric.data import Data
    
def build_graph_model(config):
    return GraphModel(config,'/home/jiayi/MammoRG-main/mammorg/KG.json')
   