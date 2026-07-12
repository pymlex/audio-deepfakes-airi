# Task 2. Spoofing-aware speaker verification

## Постановка задачи

В протоколе SASV для пары сигналов $(x_{\mathrm{ref}}, x_{\mathrm{query}})$ система должна принять trial, если query — bona fide речь заявленного спикера, и отклонить trial, если речь принадлежит другому спикеру или является spoof. Формально для трёх классов target, nontarget и spoof минимизируют weighted detection cost function, а на практике отслеживают accuracy по классам, EER и a-DCF.

Датасет `test_4k-track_2.csv` содержит 5 840 trials с примерно равным числом target, nontarget и spoof. После фильтрации битых FLAC осталось 798 валидных trials.

## Архитектура

Модель `SASVFusionModel` состоит из ECAPA-TDNN-подобного encoder для эмбеддингов $e_{\mathrm{ref}}$ и $e_{\mathrm{query}}$, countermeasure-модуля, выдающего spoof score для каждого сигнала, и fusion MLP над признаками
$$
\mathbf{z} = \big[\cos(e_{\mathrm{ref}}, e_{\mathrm{query}}),\; s_{\mathrm{cm}}^{\mathrm{ref}},\; s_{\mathrm{cm}}^{\mathrm{query}}\big].
$$
Решение принимается как $\hat{y} = \sigma(\mathrm{MLP}(\mathbf{z}))$.

Такой каскад позволяет переиспользовать pretrained CM и ASV encoder. End-to-end SASV потенциально лучше согласует границу решения, но требует больше paired trials и более тяжёлого обучения.

## Результаты

На 798 валидных trials получены следующие метрики:

| Метрика | Значение |
|---------|----------|
| accuracy | 0.650 |
| balanced accuracy | 0.634 |
| EER | 0.398 |
| ROC-AUC | 0.629 |
| a-DCF | 0.903 |
| target accept rate | 0.472 |
| nontarget reject rate | 0.771 |
| spoof reject rate | 0.889 |

Spoof отклоняется в 89% случаев, что выше, чем target accept rate 47%. Система консервативна к spoof, но теряет часть настоящих target trials. a-DCF 0.903 отражает компромисс между miss target и false accept spoof при выбранном пороге.

![ROC-кривая SASV](outputs/sasv_roc.png)

## Сравнение с baseline

Классический baseline — последовательное применение ASV и CM с fusion скоров на уровне правила. Наша fusion MLP обучается end-to-end на trial-level метках и показывает EER 0.398 на доступной подвыборке. Для production-уровня нужна полная выборка trials без битых файлов и joint fine-tune CM с SASV.

## Артефакты

Код: https://github.com/pymlex/audio-deepfakes-airi  
Веса: https://huggingface.co/pymlex/audio-deepfakes-airi
