###################################################################
# For computing cluster probability distributions for each response
# and calculate pairwise Jensen Shannon Divergence
###################################################################
import numpy as np
from typing import List, Dict

from collections import Counter
from itertools import combinations

from scipy.stats import entropy
from sklearn.cluster import KMeans

from helpers import tokenize, get_token_embedding, prepare_tokens

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

def cluster_tokens(token_list, n_clusters=15) -> Dict[str, int]:
    """
    Cluster a list of unique tokens using their embeddings.
    Returns a dictionary mapping token -> cluster label.
    """
    embeddings = []
    valid_tokens = []
    for token in token_list:
        try:
            emb = get_token_embedding(token)
            embeddings.append(emb)
            valid_tokens.append(token)
        except Exception as e:
            # If for some reason we cannot get an embedding, skip token.
            pass
    embeddings = np.stack(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=420)
    clusters = kmeans.fit_predict(embeddings)
    token_to_cluster = dict(zip(valid_tokens, clusters))
    return token_to_cluster

def get_cluster_prob_distribution(text, token_to_cluster):
    """
    Given a text, compute the probability distribution over token clusters.
    Tokens not in the mapping are ignored.
    Returns a dict: {cluster_label: probability}.
    """
    tokens = tokenize(text)
    clusters = [token_to_cluster[token] for token in tokens if token in token_to_cluster]
    if len(clusters) == 0:
        return {}
    count = Counter(clusters)
    total = sum(count.values())
    return {cluster: count[cluster] / total for cluster in count}

def jensen_shannon_divergence_dict(dist_p, dist_q, base=2):
    """
    Compute the Jensen–Shannon divergence between two discrete probability distributions.
    The union of keys is taken to ensure both distributions have the same support.
    """
    keys = set(dist_p.keys()).union(dist_q.keys())
    p = np.array([dist_p.get(k, 0) for k in keys])
    q = np.array([dist_q.get(k, 0) for k in keys])
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base=base) + 0.5 * entropy(q, m, base=base)

def compute_tau_q(
    responses_by_llm,
    questions):
    """
    Compute average pairwise JSD (tau) and std deviation per question.
    Returns two lists: mean JSD (tau) and std JSD per question.
    """
    tau_values = []
    std_values = []
    
    unique_tokens = prepare_tokens(responses_by_llm, questions)
    n_clusters = len(unique_tokens)//20
    token_to_cluster = cluster_tokens(unique_tokens,
                                  n_clusters=n_clusters)
    llm_names = list(responses_by_llm.keys())
    num_questions = len(questions)
    
    for q_idx in range(num_questions):
        question_responses = [responses_by_llm[llm][q_idx]['llm_answer'] for llm in llm_names]
        distributions = [
            get_cluster_prob_distribution(response, token_to_cluster)
            for response in question_responses
        ]
        
        jsd_pairs = [
            jensen_shannon_divergence_dict(distributions[i], distributions[j])
            for i, j in combinations(range(len(distributions)), 2)
        ]
        tau_q = np.mean(jsd_pairs)
        std_q = np.std(jsd_pairs)
        tau_values.append(tau_q)
        std_values.append(std_q)
    
    return tau_values, std_values
