# audio-deepfakes-airi

Countermeasure, SASV, uncertainty estimation, and interpretability for audio deepfake detection on ASVspoof-style data.

## Overview

Binary countermeasure model discriminates bona fide vs spoof speech. SASV extends the setting to target, nontarget, and spoof trials. Uncertainty methods include MC Dropout, deep ensemble, temperature scaling, evidential learning, and entropy. Interpretability covers saliency, Integrated Gradients, Grad-CAM, occlusion, and layer probing.

Dataset: [pymlex/audio-deepfakes-airi](https://huggingface.co/datasets/pymlex/audio-deepfakes-airi) with `flac_T.zip`, `last_eval.zip`, `metadata.zip`.

## Project tree

```
audio-deepfakes-airi/
├── install.py
├── main.py
├── config.py
├── schemas.py
├── requirements.txt
├── .env.example
├── models/                  # shared CM / SASV architectures
├── data/                    # dataset loaders and augmentations
├── utils/                   # metrics, training, plotting helpers
├── task1/                   # Task 1: binary countermeasure (CM)
│   ├── configs/
│   ├── 1_1/                 # data distribution
│   ├── 1_2/                 # audio samples
│   ├── 1_2_1/ … 1_2_4/     # model, loss, dataset, augmentations
│   ├── 1_3_1/               # train/test smoke
│   ├── 1_4_2/               # main training pipeline
│   ├── 1_5_1/ … 1_5_2/     # SOTA and tricks
│   ├── 1_6_1/ … 1_6_2/     # error analysis and cross-eval
│   ├── report.md
│   └── outputs/
├── task2/                   # Task 2: SASV
│   ├── main.py
│   ├── report.md
│   └── outputs/
├── task4/                   # Task 4: uncertainty estimation
│   ├── main.py
│   ├── plot.py
│   ├── report.md
│   └── outputs/
├── task5/                   # Task 5: explainability
│   ├── main.py
│   ├── plot.py
│   ├── report.md
│   └── outputs/
└── scripts/
    ├── setup_data.py
    ├── run_task1.py
    ├── run_task2.py
    ├── run_task4.py
    ├── run_task5.py
    ├── publish.py
    ├── upload_hf.py
    ├── deploy_server.py
    └── server/              # server-side deploy and pull helpers
```

Подпапки в `task1` нумеруются с префиксом `1_`, чтобы не пересекаться с `task2`, `task4` и `task5` в корне репозитория.

## Pipeline

```mermaid
flowchart TB
  subgraph data [Data]
    D[Download HF archives, extract flac and metadata]
  end
  subgraph cm [Task 1 CM]
    C1[Distribution and samples]
    C2[WavResNet / AASIST-lite train]
    C3[Eval EER, t-DCF, analysis]
  end
  subgraph sasv [Task 2 SASV]
    S[Fusion CM + ECAPA embeddings]
  end
  subgraph unc [Task 4 Uncertainty]
    U[MC Dropout, ensemble, calibration]
  end
  subgraph xai [Task 5 Explainability]
    X[Saliency, IG, Grad-CAM, probing]
  end
  D --> C1 --> C2 --> C3
  C2 --> S
  C2 --> U
  C2 --> X
```

## Setup

One-command install after clone on Ubuntu Jupyter with RTX GPU:

```bash
git clone https://github.com/pymlex/audio-deepfakes-airi.git
cd audio-deepfakes-airi
cp .env.example .env
python install.py
```

Set `HF_TOKEN` in `.env` for dataset and model upload.

## Run

Full pipeline:

```bash
python main.py --task all
```

Individual tasks:

```bash
python main.py --task 1
python main.py --task 2
python main.py --task 4
python main.py --task 5
```

Task 1 only:

```bash
python scripts/run_task1.py --step train --config task1/configs/cm_baseline.yaml
python scripts/run_task1.py --step sota
```

## Models

**WavResNet** — mel-spectrogram, dB scale, ResNet-18 backbone, 2-class CE loss.

**AASISTLite** — CNN encoder, temporal and frequency graph attention, fusion classifier.

**SASVFusionModel** — ECAPA-style embeddings, CM scores, MLP fusion for accept/reject.

## Metrics

CM: EER, min t-DCF, balanced accuracy, ROC-AUC, per-class accuracy.

SASV: a-DCF, SASV-EER, target accept rate, spoof reject rate.

## Publish

| Target | Repository |
|--------|------------|
| Code, metrics, plots, reports | [github.com/pymlex/audio-deepfakes-airi](https://github.com/pymlex/audio-deepfakes-airi) |
| Dataset archives | [pymlex/audio-deepfakes-airi](https://huggingface.co/datasets/pymlex/audio-deepfakes-airi) |
| Model checkpoints | [pymlex/audio-deepfakes-airi](https://huggingface.co/pymlex/audio-deepfakes-airi) |

```bash
python scripts/publish.py
python scripts/upload_hf.py
```

## Citation

```bibtex
@misc{zyukov2026_audio_deepfakes_airi,
  author       = {Alex Zyukov},
  title        = {audio-deepfakes-airi: Countermeasure, SASV, Uncertainty, and Explainability},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/pymlex/audio-deepfakes-airi}}
}
```

```bibtex
@article{wang2024asvspoof5,
  title   = {ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale},
  author  = {Wang, Xin and others},
  journal = {arXiv preprint arXiv:2408.08739},
  year    = {2024}
}
```

```bibtex
@article{jung2022sasv,
  title   = {SASV 2022: The First Spoofing-Aware Speaker Verification Challenge},
  author  = {Jung, Jee-weon and others},
  journal = {arXiv preprint arXiv:2203.14732},
  year    = {2022}
}
```

```bibtex
@article{sensoyan2018evidential,
  title   = {Evidential Deep Learning to Quantify Classification Uncertainty},
  author  = {Sensoyan, Murat and others},
  journal = {arXiv preprint arXiv:1806.01768},
  year    = {2018}
}
```

The project is under GPL-3.0 license.
