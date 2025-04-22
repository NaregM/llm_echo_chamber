import numpy as np
import pandas as pd

import torch
from typing import List, Dict
import numpy.typing as npt

from collections import Counter
from itertools import combinations

from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

import re

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Load the tokenizer and model (BERT-base-uncased)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
embedding_layer = model.get_input_embeddings()

# -----------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """
    """
    return tokenizer.tokenize(text)

def get_token_embedding(token: str) -> npt.NDArray:
    
    """
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    token_tensor = torch.tensor([token_id])
    
    with torch.no_grad():
        
        emb = embedding_layer(token_tensor)
    
    return emb.squeeze(0).numpy()