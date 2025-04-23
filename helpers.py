import numpy as np
import pandas as pd

import torch
from typing import List, Dict
import numpy.typing as npt

from collections import Counter, defaultdict
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

def average_distributions(dists):
    """
    Takes a list of distributions (dicts of token probabilities) and averages them.
    Assumes all distributions have the same keys (tokens).
    """
    averaged = defaultdict(float)
    num = len(dists)

    for dist in dists:
        for k, v in dist.items():
            averaged[k] += v / num
    
    return dict(averaged)

def prepare_tokens(responses_by_llm, questions):

    llm_names = list(responses_by_llm.keys())
    unique_tokens = set()

    for q_idx, question in enumerate(questions):
        question_tokens = set(tokenize(question))

        # Gather all N answers for this question (responses from each LLM)
        question_answers = [
            responses_by_llm[llm][q_idx]['llm_answer']
            for llm in llm_names
        ]

        for answer in question_answers:
            answer_tokens = tokenize(answer)
            filtered_tokens = [token for token in answer_tokens if token not in question_tokens\
                                  and len(token) > 2]
            unique_tokens.update(filtered_tokens)

    # Even when random_state is fixed KMeans implementations still depend on the order 
    # of the input points 
    unique_tokens = sorted(list(unique_tokens), key=lambda x: (len(x), x))
    
    return unique_tokens
