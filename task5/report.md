# Task 5: Explainability для CM

## Методы

### Saliency maps

$|\partial s / \partial x|$ по waveform. Показывает локально чувствительные отсчёты.

### Integrated Gradients

$$\mathrm{IG}_i = (x_i - x'_i) \int_0^1 \frac{\partial s}{\partial x_i}\Big|_{x' + \alpha(x-x')} d\alpha$$

### Grad-CAM

Веса каналов последнего conv слоя по градиентам, heatmap на mel-оси.

### Occlusion sensitivity

Последовательное зануление окон waveform, $\Delta s = s_{full} - s_{occluded}$.

### Layer probing

Linear probe на features layer1–layer4. Показывает, на каком уровне separability bona fide/spoof максимальна.

## Визуализации

Примеры в `outputs/`:
- `saliency_*.png`
- `ig_*.png`
- `gradcam_*.png`
- `occlusion_*.png`
- `layer_probing.json`

## Probing

| Слой | Linear probe accuracy |
|------|----------------------|
| layer1 | 0.750 |
| layer2 | 0.875 |
| layer3 | 0.875 |
| layer4 | 1.000 |

Separability растёт от layer1 к layer4.

## SpeechEval и OOD

Для out-of-domain интерпретации рекомендуется SpeechEval и собственные записи с XTTS-v2 для сравнения attribution maps между benchmark и wild deepfakes.
