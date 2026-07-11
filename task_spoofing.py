# Speech Deepfake Detection, Countermeasures, SASV, Uncertainty, and Interpretability (200 + 50 points)

## Context and preserved resources

Several attack families are relevant for voice biometrics:

- replay attacks: a recording of a target speaker is played back;
- impersonation attacks: a person imitates the target speaker;
- synthetic or converted speech attacks: text-to-speech, voice conversion, voice cloning, and other generative systems.

We focus on bona fide vs spoof speech detection and then extend the setting to SASV, where a system should accept target bona fide speech and reject both impostors and spoofed speech.

Useful ASVspoof and related resources from the previous version of the notebook are preserved here:

- ASVspoof website: https://www.asvspoof.org/
- ASVspoof 2019 LA data: https://datashare.ed.ac.uk/handle/10283/3336
- ASVspoof 2021 LA: https://zenodo.org/records/4837263#.YnDIinYzZhE
- ASVspoof 2021 DF: https://zenodo.org/records/4835108#.YnDIb3YzZhE
- ASVspoof 5 / 2024 resources: https://zenodo.org/records/14498691
- Old smaller Google Drive subsample used by the previous notebook: https://drive.google.com/drive/folders/1-CyCFA3komqrtyoYj21y5gzfh_vcKBYx?usp=share_link
- ASVspoof 5 evaluation plan: https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase1.pdf
- ASVspoof 5 result summary paper: https://arxiv.org/pdf/2408.08739

You may use other datasets such as SingFake, MLAAD, In-the-Wild, or your own audio, but your report must state exactly what data was used.

One important thing to notice: main application of this algorithms is in voice-biometry, when we want to stop illegal intruder. Thus, there are two (or 3) common ways of developing VAS algorithms: speaker-aware, we we train verification model which is sensitive to spoofing and differs bona-fide and impostor or we combinde verification score and score of VAS model to decide, whetherr the person is the same, and finally simple and general countermeasures setup, when given audio and model should predict whether it is a spoof or bona-fide. We will work on the last setup.

**Important: feel free not to use pre-defined functions, you can solve the task as you wish or change functions/pipelines significantly.**

## Download / Data Preparation

https://huggingface.co/datasets/pymlex/audio-deepfakes-airi

flac_T.zip
6.76 GB
last_eval.zip
5.52 GB
metadata.zip
481 kb

#### Preparations

import copy
import os
import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython.display as ipd
from typing import Optional
import torch
import torch.nn as nn
import torchaudio
# import librosa
import sklearn.metrics as metrics
from torchvision.models import resnet50, resnet18
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import sigmoid
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
!ls

"""# Task 1. Audio Deepfake Detection (ADD, SDD) in binary (countermeasure, CM) setting (60 pts)

test_track_1.csv results are expected
"""

def make_CM_df(...):
    # ToDO
    df = df.reset_index(drop=True)
    return df

df_train =
df_dev =
df_eval =

"""**Task 1.1 (2 points)** Check data distribution. Is it imbalanced? How are you going to deal with it?

Hints: upsampling, downsampling, add new data, e.g. from this [source](https://commonvoice.mozilla.org/) (but will new audios have the same distribution?), or adjust loss functions.

**Answer**

...
"""

def balanced_split(df):
    # Optional
    return ...

# for audio manipulations we advise you to use torchaudio or librosa
x, sr = torchaudio.load(df_eval["audio_path"].iloc[0])

ipd.display(ipd.Audio(x, rate=sr))

"""**Task 1.2. (1 point)** Display several spoof and bona-fide audios. Can you hear the difference?"""

### Your code is here

"""### Custom dataset and Model

You can find inspiration and hints in
- https://www.asvspoof.org/workshop2024
- https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://pypi.org/project/audiomentations/
- https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
- and any other blogposts about spectrograms

**Task 2.1 (5 points)** We can work with audios as with images, transforming into spectrograms. Here your task is to implement simple model, which receives raw wav (amplitudes, but probably already preprocced in dataset), transforms it into mel-spectrogram, changes amplitude to DB scale and then procceses through the layers. You are free to use ready pre-trained backbones, e.g. from `torchvision.models` and fine-tune them.
"""

class WavResNet(nn.Module):
    def __init__(self, classes=None, resample=16000, n_mels=80, melspec_config=None):
        super().__init__()
        self.fbank = ...
        self.to_db = ...

        model = ...
        model.conv1 = ...
        num_ftrs = ...
        model.fc = nn.Linear(num_ftrs, classes)
        self.network = model
        # or create your own layers and use them in forward pass


    def forward(self, x, wav_lens: Optional[torch.Tensor]=None):  # can check the length if you want. this is helpful for inference
        mels = ...
        mels_db = ...
        mels_db = ...
        out = ...
        return out

"""**Task 2.2 (1 point)**
In your opinion, which approach is better for binary classification:
- Model's last layer output has shape 1, train with BCE-like loss.
- Model's last layer output has shape 2, train with cross-entropy like loss.

Make experiments.

**Task 2.3 (3 points)** Create custom dataset, which recieves ```df``` and returns preprocessed audio.

**Task 2.4 (1 point)** Should we use augmentaions? If yes, which ones?
"""

class DatasetWav(Dataset):

    def __init__(self, data_frame, padding_sec=4, default_sr=16000, transform=None):
        self.df = data_frame
        self.padding_sec = padding_sec
        self.default_sr = default_sr
        self.labels = ...
        self.paths = ...
        self.vad = ...   # in this task you are free to ommit it in order to speed up calculations,
                         # also the provided data should be rather clean
    def __len__(self):
        return

    def __getitem__(self, idx):

        label = ...
        path = ...
        waveform , sr = ... # normalize=True
        # transform waveform from stereo to mono channel
        waveform = ...
        resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.default_sr) # should we use it for our data or can ommit?
        waveform = ...
        # waveform = self.vad(waveform)

        # came up with idea, what to do if audio is longer or shorter than reuqired
        if (len(waveform) < self.padding_sec * self.default_sr):
          pass
        else:
          pass

        return x, label

# check that works
batch_size = None
train_dataset_wavs = DatasetWav(df_train) #.iloc[0:200]
train_dataloader_wavs = DataLoader(train_dataset_wavs, batch_size=batch_size, shuffle=True, num_workers=8)
x, y = next(iter(train_dataloader_wavs))
model = ...
model(x)

"""### Train and Test functions
**Task 3.0 (1 point)** What is the difference between `model.train()` and `model.eval()`? Does `model.eval()` mode take gradient statisitcs into account?

**Task 3.1 (5 points)** Implement train and test functions, which iterate over all batches. Do logging of loss, accuracy on each batch and after every epoch. Check equal error rate EER, explain, what is it. Evaluation metrics should include EER, per-class accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, t-DCF (see ASVspoof evaluation plan for priors).
"""

def calculate_eer(y, y_score):
  fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
  eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
  thresh = interp1d(fpr, thresholds)(eer)
  return eer, thresh

#Other metrics

def train(model, dataloaders, criterion, optimizer,
          num_epochs=3, scheduler=None, savename=None,
          print_counter=10, decay_factor=10,
          device=torch.device("cuda")):
    pass

def test(model, test_dataloader, criterion=None, device=None, savename=None):

    pass

"""### Main loop

**Task 4.1 (3 points)** What loss will you choose and why? What it should receive for input (logits, probs)? Explain your answer.
Consider also:
- https://medium.com/swlh/focal-loss-what-why-and-how-df6735f26616
- https://libauc.org/


**Task 4.2 (5 points)** Implement main function, which receives yaml or json config (or path to it) and train and test the model, save model checkpoints, and model's test predictions. At least, you should obain results better than random. It is good, if accuracy on test dataset is >= 0.75.
"""

def main(config):
    lr =
    epochs =
    batch_size =
    нужно логирование, как с wandb


    # debug firstly on small subpart of dataset
    train_dataset_wavs = DatasetWav() #.iloc[0:200]
    train_dataloader_wavs = DataLoader()
    val_dataset_wavs =
    val_dataloader_wavs =
    test_dataset_wavs =
    test_dataloader_wavs =
    dataloaders = {"train": train_dataloader_wavs, "val": val_dataloader_wavs}


    device =
    classes =
    model =
    model = model.to(device)
    # freeze layers except the last one if you want
    for param in model.parameters():
      param.requires_grad = ...
    model.network.fc.requires_grad_(...)

    optimizer =
    scheduler =
    sc = torch.tensor([alpha, 1 - alpha]).to(device) # weights for loss
    criterion = nn.CrossEntropyLoss(weight=sc)

    for_ckpts = "."
    os.makedirs(for_ckpts, exist_ok=True)

    model, val_acc_history = train(...)

    y, probs = test(...)

    return y, probs

y, probs = main(...)

"""Training might require a long time. So, use a subpart of the dataset to receive rather good results. Also, you don't have to train for many epochs, 1 might be enough. Also consider to make some speedups in the model and dataset.

### SOTA solutions

**Task 5.1** Consider SOTA solution since Interspeech 2024 and ASVspoof 2024, but preferably since 2026. Study approaches, consider more strong model that improves results (5 pts).


**Task 5.2** Experiment with techniques and tricks to improve the results. (3 points for every sucessful trick, up to 15 points).

### Evaluation

**Task 6.1 (5 points)** Analyze the results. Play some audios with wrong predictions. Plot prediction distributions vs class. Ananlyze the intersection intersection.

**Task 6.2 (40 points)**

    - (5 points) Evaluate the results on test data samples from 21-LA, 21-DF, 24-dev, 24-eval, 24-test. Is there a degradtion of results?
    - (30 points max, 10 point for every correctly tested technique (not neccessery correctly, although preferably :) )) Consider several Uncertainty Estimation (Quantification) or classfification with abstain (rejection)  or score callibration. Describe the methods briefly , prepare the code (or use ope-sourced), and conduct experiments.
    - (1 point) Obtain better results than with the previous method.
    - (2 points) Discuss how the considered techniques help to improve the results (for example, it abstain from oredictions).
    - (2 points) Check your approaches on out-of-domain bona-fide and spoof audio. For example, consider you voice, YouTuve, xtts-v2.



**Task 6.2 (5 points)** Discuss the results and your approaches + hyperparameters. If you can, run some experiments with different parameters.

 Write your ideas, what else you can try to improve the results in the future.


**Task 6.3 (3 points)** Provide a link to github. Upload best model's weights to hugging face.

## Task 2: SASV Setting (50 points)

test_4k-track_2.csv results are expected


SASV means spoofing-aware speaker verification. The system receives a trial and should:

- accept `target` / `accept`: bona fide speech from the claimed speaker;
- reject `nontarget` / `impostor`: bona fide speech from another speaker;
- reject `spoof`: synthetic or converted speech.

Required trial file: `test_4k-track_2.csv` under `METADATA_DIR` by default.

Inspiration and protocol references:

- SASV Challenge results: https://sasv-challenge.github.io/challenge_results/
- Wildspoof overview of results: https://drive.google.com/file/d/1fcWpUi39Glq4onEAgeQZ13BJPRBEjbyH/view
- SASV 2022 paper: https://arxiv.org/abs/2203.14732
- https://sasv-challenge.github.io/pdfs/2022_descriptions/IDVoice.pdf


These links might be helpful:
- https://github.com/archinetai/surgeon-pytorch
- https://www.kaggle.com/code/peter0749/additive-margin-softmax-loss-with-visualization
- https://www.asvspoof.org/workshop2024_program

Provide overview of state-of-the-art approache and common baselines.
Expected results are end-to-end and/or fusion model. Metrics (EER, accuracy for accept/imposter/spoof categories, A-DCF, balanced accuracy).
"""

# Your solution is here

"""## Task 4: Uncertainty Estimation for CM / SASV (40 points)

Use 3-5 uncertainty estimation methods for CM and/or SASV. At least one model should use an evidential deep learning or classification uncertainty approach.

Reference materials:

- LabML evidential uncertainty tutorial: https://nn.labml.ai/uncertainty/evidence/index.html
- Evidential Deep Learning paper: https://arxiv.org/pdf/1806.01768
- PyTorch classification uncertainty examples: https://github.com/dougbrion/pytorch-classification-uncertainty
- Recent audio uncertainty reference: https://arxiv.org/pdf/2407.01143

Suggested methods:

- MC Dropout;
- deep ensembles;
- temperature scaling and calibration;
- evidential deep learning;
- entropy / predictive confidence;
- test-time augmentation;
- dropouts, probing.

"""



"""## Task 5: Explainability / Interpretability for Audio (50 points)

Apply 4-5 interpretability methods to the spectrogram CM model or the raw-audio model. You do not need to fully implement every research idea below; provide several working starter examples and a clear research discussion.

Starter methods:

- saliency maps;
- Integrated Gradients;
- Grad-CAM or Grad-CAM-like methods for spectrogram CNNs;
- occlusion sensitivity;
- temporal masking;
- frequency masking;
- probing intermediate layers;
- representation analysis.

Links and ideas for inspiration:

- Speech interpretability tutorial: https://interpretingdl.github.io/speech-interpretability-tutorial/interspeech2025/intro.html
- https://arxiv.org/abs/2606.10912
- https://arxiv.org/pdf/2602.05027
- https://openaccess.thecvf.com/content/CVPR2026W/APAI/papers/Subramani_Explainability_Analysis_of_Mel-Based_Synthesis_Models_for_Anti-Spoofing_CVPRW_2026_paper.pdf
- Probing tutorial: https://interpretingdl.github.io/speech-interpretability-tutorial/interspeech2025/representational-analyses/probing.html
- https://www.isca-archive.org/interspeech_2024/li24oa_interspeech.pdf
- https://arxiv.org/pdf/2407.18517v1
- https://github.com/davidcombei/ai4t?tab=readme-ov-file
- https://arxiv.org/abs/2510.14664
- SpeechEval dataset: https://huggingface.co/datasets/Hui519/SpeechEval
- https://www.isca-archive.org/interspeech_2025/bolanos25_interspeech.pdf
- https://github.com/menglu-lml/Interpretable-Detection-IS24
- https://github.com/MuSAELab/AUDDT

Research menu:

- sparse autoencoders for audio representations;
- token-level, phoneme-level, or word-level SLIM-like analysis;
- suspicious temporal segment search: pauses, transitions, intonation anomalies;
- probing layers on correct and incorrect spoof / bona fide predictions;
- SpeechLLM-as-a-judge for reasoning and explanations;
- comparison of real-world audio deepfakes and benchmark deepfakes.

"""