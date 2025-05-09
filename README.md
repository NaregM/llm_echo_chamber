We define the mean pairwise Jensen–Shannon divergence (JSD) among $N$ large language models (LLMs) across $Q$ questions. 
For each question $q$, we compare the token cluster distributions generated by each model’s response, 
measuring divergence between every pair of models. The final metric, $\tau$, averages these pairwise JSD scores across all questions:

$$
\tau = \frac{1}{Q} \sum_{q=1}^{Q} D_{\text{pairwise}}^{(q)},
$$

where

$$
\tau_q = D_{\text{pairwise}}^{(q)} = \frac{1}{\binom{N}{2}} 
\sum_{1 \leq i < k \leq N} 
\text{JSD}\left( P_i^{(q)} \mid \mid P_k^{(q)} \right).
$$

Here, $P_i^{(q)}$ denotes the token cluster probability distribution for model $i$ responding to question $q$.
