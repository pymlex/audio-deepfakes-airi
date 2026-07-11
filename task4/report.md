# Task 4: Uncertainty Estimation для CM

## Методы

### MC Dropout

$T$ forward pass с включённым dropout при inference. Predictive mean $\bar{p} = \frac{1}{T}\sum_t p_t$, uncertainty $u = \mathrm{std}(p_t)$.

### Deep Ensemble

$M$ моделей с perturbed weights. Disagreement $\mathrm{std}_m p_m$ как epistemic uncertainty.

### Temperature Scaling

Калибровка logits: $\tilde{z} = z / T$. $T$ подбирается на dev по NLL.

### Evidential Deep Learning

Dirichlet параметры $\alpha = \mathrm{softplus}(f(x)) + 1$. Uncertainty $u = K / \sum_k \alpha_k$.

### Entropy

Predictive entropy $H = -\sum_k p_k \log p_k$.

## Результаты

Метрики сохранены в `outputs/`:
- `mc_dropout.json`
- `deep_ensemble.json`
- `temperature_scaling_metrics.json`
- `entropy.json`
- `all_uncertainty.json`

MC Dropout и ensemble позволяют abstain на trials с высокой uncertainty. Temperature scaling улучшает калибровку без изменения ranking.

## Abstention

Reject prediction если $u > \tau_u$. На eval это снижает false accept за счёт отложенных решений.
