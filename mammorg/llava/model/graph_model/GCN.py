import torch
import networkx as nx
from torch_geometric.nn import RGCNConv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from llava.constants import DENSITY_CLASSES, BI_RADS_CLASSES, ENTITY_CLASSES, DIAGNOSIS, ANATOMY, ADJUNCTS, ENTITY_NAMES
from llava.model.bert import BertEmbeddingExtractor
class MedicalKnowledgeGraph:
    def __init__(self, data):
        self.data = data
        self.G = nx.MultiDiGraph()
        self.entity_types = {
            'density': DENSITY_CLASSES[:-1],
            'observation': ENTITY_NAMES,
            'anatomy': ANATOMY,
            'diagnosis': DIAGNOSIS,
            'adjuncts': ADJUNCTS
        }
        self.relation_types = {
            'located_at': '位于',
            'suggestive_of': '提示',
            'modified_by': '被修饰'
        }
        
    def build_graph(self):
        for ent_type, entities in self.entity_types.items():
            for ent in entities:
                self.G.add_node(ent, type=ent_type)
        
        for obs, locs in self.data['Located_at'].items():
            for loc in locs:
                self.G.add_edge(obs, loc, relation='located_at')
        
        for obs, diags in self.data['Suggestive_of'].items():
            for diag in diags:
                self.G.add_edge(obs, diag, relation='suggestive_of')
        
        for obs, mods in self.data['Modified_by'].items():
            for mod in mods:
                self.G.add_edge(obs, mod, relation='modified_by')
        
        return self.G

class FeatureExtractor(nn.Module):
    def __init__(self, bert_dim, hidden_dim, num_relations, 
                 num_layers=2, dropout=0., norm_type='layernorm', 
                 use_residual=True):
        """
        Enhanced Feature Extractor with Configurable RGCN Layers
        
        Args:
            bert_dim: Input feature dimension (e.g., BERT hidden size)
            hidden_dim: Hidden layer dimension
            num_relations: Number of relation types for RGCN
            num_layers: Number of RGCN layers (default: 2)
            dropout: Dropout rate (default: 0.2)
            norm_type: Normalization type ('layernorm', 'batchnorm', or None)
            use_residual: Whether to use residual connections (default: True)
        """
        super(FeatureExtractor, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type
        self.use_residual = use_residual
        
        self.in_conv = RGCNConv(bert_dim, hidden_dim, num_relations)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        # RGCN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm_type else None
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        
        # Normalization layers
        if norm_type:
            for _ in range(num_layers):
                if norm_type == 'layernorm':
                    self.norms.append(nn.LayerNorm(hidden_dim))
                elif norm_type == 'batchnorm':
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Residual projection
        if use_residual and bert_dim != hidden_dim:
            self.res_proj = nn.Linear(bert_dim, hidden_dim)
        else:
            self.res_proj = None

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        h = x
        h = self.in_conv(h, edge_index, edge_type)
        h = self.in_norm(h)
        h = F.relu(h)

        for i in range(self.num_layers):
            # Residual connection (except first layer)
            residual = h
            
            # RGCN layer
            h = self.norms[i](h)
            h = self.convs[i](h, edge_index, edge_type)

            # Activation & Dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            h = h + residual
        h = self.out_norm(h)
        return h
    
class GraphModel(torch.nn.Module):
    def __init__(self, config, json_path):
        super(GraphModel, self).__init__()
        
        self.bert_extractor = None 
        self.data = None
        self.json_path=json_path
        self.model = FeatureExtractor(
            bert_dim=768,
            hidden_dim=config.mm_hidden_size,
            num_relations=3,
            num_layers=4,
            norm_type='layernorm'        
        )
    def build_graph_data(self):
        if self.data is None:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            kg = MedicalKnowledgeGraph(data)
            G = kg.build_graph()

            bert_extractor = BertEmbeddingExtractor()
            nodes = list(G.nodes())
            node_features = bert_extractor.batch_get_embeddings(nodes)

            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edges, edge_types = [], []
            relation_to_idx = {'located_at': 0, 'suggestive_of': 1, 'modified_by': 2}
            
            for src, dst, attr in G.edges(data=True):
                edges.append([node_to_idx[src], node_to_idx[dst]])
                edge_types.append(relation_to_idx[attr['relation']])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            
            self.data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_type,
                num_relations=len(relation_to_idx)
            )
    def forward(self):
        if self.data is None:
            self.build_graph_data()
        self.model = self.model.float()
        device = next(self.model.parameters()).device
        target_dtype = next(self.model.parameters()).dtype

        self.data = self.data.to(device)

        if self.data.x.dtype != target_dtype:
            self.data.x = self.data.x.to(target_dtype)
        
        return self.model(self.data)
    