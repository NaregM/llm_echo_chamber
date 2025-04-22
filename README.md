This paper investigates whether a collection of $N$ widely-used large language models (LLMs)
exhibit an echo chamber effect in their generated responses. To quantify response diversity, we
introduce a novel metric: the mean pairwise Jensen–Shannon divergence, $\tau$, computed over the models’
outputs.

$$
\tau \equiv \frac{1}{Q \binom{N}{2}}\sum_{q=1}^{Q} \sum_{1 \leq i < k \leq N} \text{JSD}\left( \mathrm{P}_i^{(q)} \|\| \mathrm{P}_k^{(q)} \right).
$$