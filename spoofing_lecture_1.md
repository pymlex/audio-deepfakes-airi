# Голосовой антиспуфинг: Countermeasures, SASV, трюки и Audio-LLM

КРАТКИЙ ОБЗОР СОРЕВНОВАНИЙ, ИССЛЕДОВАНИЙ И БЕНЧМАРКОВ

SAFE AI

СЕМИНАР

НЕДЕЛЯ 9

### Главная идея

Развитие генеративных (аудио) технологий привело к новым векторам атак и способам мошенничества.

- Современные TTS/VC/voice cloning модели уменьшают стоимость атаки: достаточно короткого фрагмента голоса, публичных данных и API.
- Детекторы часто выглядят сильными на академическом train/test, но падают на новых генераторах, кодеках, языках, доменах и частичных подделках.
- Поэтому важнее всего обсуждать **угрозу**, **протокол оценки**, **обобщение** и **ошибки в реальном мире**.

Источники: [ASVspoof 2019](https://arxiv.org/abs/1904.05441); [ASVspoof 5;](https://arxiv.org/abs/2408.08739) [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857)

### Карта семинара

- Проблематика, соревнования, бенчмарки и метрики.
- Угрозы: replay/PA, LA, deepfake, adversarial, partial spoof, singing voice.
- Countermeasure systems: front-end, encoder, back-end, loss, calibration.
- RawNet2, AASIST, AASIST2, AASIST3, one-class learning, SAMO.
- Spoofng-aware speaker verifcation: от CM+ASV к SASV.
- SSL encoders и audio LLM/reasoning: что дают и где ломаются.
- Практические выводы: как строить robust benchmark и production pipeline.

### Термины: bona fde, spoof, deepfake

| Bona<br>fde                                                                                                                                        | Spoof                                                                                                                              | Deepfake                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Настоящая<br>речь<br>человека,<br>записанная<br>или<br>переданная<br>без<br>синтетической<br>подмены<br>релевантного<br>голосового<br>содержимого. | Сигнал,<br>цель<br>которого<br>-<br>обмануть<br>ASV<br>или<br>слушателя:<br>replay,<br>TTS,<br>VC,<br>impersonation,<br>tampering. | Обычно<br>синтетическая<br>или<br>измененная<br>речь,<br>созданная<br>современными<br>генеративными<br>моделями.<br>В<br>ASVspoof<br>2021/5<br>выделена<br>как<br>отдельный<br>сценарий. |

Источник: [Yamagishi et al., "ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection", 2021](https://arxiv.org/abs/2109.00537)

# Почему голос особенно уязвим

- Голос одновременно **биометрический идентификатор**, **медиа-сигнал** и **социальное доказательство**.
- В отличие от пароля, голос часто публичен: интервью, подкасты, видео, конференции, звонки.
- Атака не обязана быть идеальной для человека: достаточно пересечь порог ASV или вызвать сомнение у аудитории.
- Детектору нужно решать OOD-задачу: новые TTS/VC появляются быстрее, чем пересобираются датасеты.

Источник: [Zhang, Jiang, Duan, "One-class Learning Towards Synthetic Voice Spoofng Detection", 2020](https://arxiv.org/abs/2010.13995)

# Угрозы: от replay к deepfake

- **PA/replay**: физическое проигрывание записи в микрофон.
- **LA**: подача синтетической или converted речи в канал доступа без акустического replay.
- **Deepfake in the wild**: социальные сети, звонки, компрессия, шум, multilingual, unknown generators.
- **Partial spoof**: подменяются короткие участки, а большая часть записи bona fde.

Источники: [ASVspoof 2019](https://arxiv.org/abs/1904.05441); [PartialSpoof](https://arxiv.org/abs/2204.05177)

# ASV и CM: где стоит антиспуфинг

- Классический CM отвечает: **bona fde или spoof?**
- ASV отвечает: **это целевой спикер или нет?**
- SASV объединяет оба вопроса: **это настоящий голос целевого спикера?**

Источник: [Jung et al., "SASV 2022: The First Spoofng-Aware Speaker Verifcation Challenge", 2022](https://arxiv.org/abs/2203.14732)

### Метрики: EER, min t-DCF, a-DCF, SASV-EER

- **EER**: точка, где false accept rate равен false reject rate. Удобна для сравнения, но не задает стоимость ошибок.
- **min t-DCF**: минимальная tandem detection cost function для каскада ASV+CM; учитывает prior spoof/non-target/target и стоимости ошибок.
- **a-DCF / actDCF**: ASVspoof 5 вводит метрики для stand-alone CM и SASV, включая calibration-sensitive варианты.
- **SASV-EER**: единая ошибка для target bona fde, non-target bona fde и spoofed non-target trials.

#### Смысл

Одна и та же EER может быть приемлемой в датасете и плохой в банке, если spoof prior или стоимость false accept намного выше.

Источники: [ASVspoof 2019 Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf); [SASV Evaluation Plan;](https://arxiv.org/abs/2201.10283) [ASVspoof 5](https://arxiv.org/abs/2408.08739)

# ASVspoof: зачем серия соревнований

- Цель серии дать общие данные, протоколы и метрики для сравнения countermeasures против spoofng attacks.
- Главный акцент ASVspoof: не просто найти артефакт известного генератора, а оценить устойчивость к неизвестным атакам.
- С 2019 года важен t-DCF: CM оценивается как часть ASV-системы, а не как изолированный классификатор.
- С 2021/5 появляется более явный deepfake/in-the-wild фокус.

Источник: [Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection", 2019](https://arxiv.org/abs/1904.05441)

# ASVspoof 2019: LA и PA в одном benchmark

#### Logical Access

TTS и voice conversion attacks, включая современные neural acoustic и waveform models. Атака как цифровая подача в систему.

#### Physical Access

Replay attacks: запись bona fde речи проигрывается в физической среде и заново записывается микрофоном.

- Важный вклад: единая база для synthesized, converted и replayed speech.
- Метрика min normalized t-DCF стала основной, EER вторичной.
- Сильные системы часто были ансамблями и fusion разных frontends/backends.

Источник: [Todisco et al., ASVspoof 2019 summary paper](https://arxiv.org/abs/1904.05441)

### ASVspoof 2019: что показали результаты

- LA оказалось легче для сильных систем, но отдельные атаки вроде A17 были устойчивыми к обычным спектральным признакам.
- PA/replay сложнее из-за акустической среды, устройств воспроизведения и записи.
- Fusion работал хорошо для LA, но переносить fusion-стратегии на PA было сложнее.
- Наличие "unknown attacks" сформировало мотивацию для raw waveform моделей, graph attention и one-class losses.

Источник: [Wang et al., "ASVspoof 2019: spoofng countermeasures for synthesized, converted and replayed speech", 2021](https://arxiv.org/abs/2102.05889)

# ASVspoof 2021: LA, PA и DF

| Track          | Что<br>проверяет                                                                                                          | Почему<br>стало<br>сложнее                                                                                                                         |
|----------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| LA<br>PA<br>DF | TTS/VC<br>в<br>logical<br>access<br>replay<br>в<br>физическом<br>доступе<br>speech<br>deepfake<br>detection<br>без<br>ASV | channel<br>variability<br>и<br>transmission<br>efects<br>реальные<br>помещения<br>и<br>replay<br>setup<br>standalone<br>fake<br>audio<br>detection |

- Участникам не дали новые matched train/dev данные: нужно было готовиться на ASVspoof 2019 и augmentation.
- Это ближе к реальности: будущий spoofng generator неизвестен заранее.

Источник: [Delgado et al., "ASVspoof 2021 Evaluation Plan", 2021](https://arxiv.org/abs/2109.00535)

# ASVspoof 2021: главный урок

#### Generalization over leaderboard

Сильный результат на known protocol не гарантирует устойчивость к channel, codec, compression и доменным сдвигам.

- LA и DF результаты оказались близки к предыдущим benchmark, несмотря на channel/compression variability.
- PA остался проблемным: реальная физическая среда сильно меняет сигнал.
- Challenge показал, что отсутствие matched train/dev не уничтожает прогресс, но увеличивает роль augmentation и robust features.

Источник: [Yamagishi et al., ASVspoof 2021 summary paper](https://arxiv.org/abs/2109.00537)

# ASVspoof 5 / 2024: что изменилось

- База построена на crowdsourced speech с гораздо большим числом спикеров и разнообразными акустическими условиями.
- Атаки тоже crowdsourced и проверяются surrogate detection models.
- Впервые включены adversarial attacks.
- Две задачи: stand-alone countermeasure и spoofng-robust ASV / SASV.
- Метрики поддерживают как CM без ASV, так и integrated SASV.

Источники: [ASVspoof5 Evaluation Plan Phase 2, v0.6, 2024](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf); [Wang et al., ASVspoof 5, 2024](https://arxiv.org/abs/2408.08739)

# ASVspoof 5: почему это benchmark нового поколения

- **Scale**: crowdsourced speakers и атаки расширяют вариативность.
- **Attacker-aware design**: атаки генерируются и фильтруются с учетом surrogate detectors.
- **Adversarial setting**: проверяется не только генерация речи, но и попытка обойти детектор.
- **SASV**: оценка ближе к реальному biometric product: нужно отвергнуть spoof и non-target.

#### Вывод

После ASVspoof 5 уже недостаточно "хорошего LA detector": нужен pipeline, устойчивый к crowdsourcing, codecs, adversarial optimization и speaker-verifcation constraints.

Источник: [Wang et al., "ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale", 2024](https://arxiv.org/abs/2408.08739)

# Бенчмарки вне ASVspoof

| Dataset<br>/<br>benchmark | Что<br>добавляет                                                                                | Ограничение                                                 |
|---------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| SpeechFake                | 3M+<br>deepfake<br>samples,<br>3000+<br>часов,<br>40<br>synthesis<br>tools,<br>46<br>языков     | синтетическая<br>конструкция<br>датасе<br>та                |
| ShiftySpeech              | 3000+<br>часов,<br>7<br>distribution<br>shifts,<br>6<br>TTS,<br>12<br>vocoders,<br>3<br>языка   | benchmark-controlled<br>shifts,<br>не<br>вся<br>wild-среда  |
| Deepfake-Eval-2024        | in-the-wild<br>media<br>from<br>2024:<br>56.5h<br>audio,<br>52<br>языка,<br>social<br>platforms | gated,<br>evaluation-only<br>terms                          |
| SingFake<br>/<br>CtrSVDD  | singing<br>voice<br>deepfake<br>detection                                                       | музыка<br>и<br>vocals<br>требуют<br>отдельной<br>постановки |

Источники: [SpeechFake, ACL 2025](https://aclanthology.org/2025.acl-long.493/); [ShiftySpeech;](https://huggingface.co/datasets/ash56/ShiftySpeech) [Deepfake-Eval-2024;](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) [SingFake](https://arxiv.org/abs/2309.07525); [CtrSVDD](https://arxiv.org/abs/2406.02438)

# Как читать benchmark правильно

- <sup>1</sup> Какие генераторы в train/dev/eval? Есть ли unseen generators?
- <sup>2</sup> Спикеры пересекаются? Языки пересекаются? Тексты пересекаются?
- <sup>3</sup> Есть ли channel, codec, replay, social-media laundering?
- <sup>4</sup> Метрика threshold-free или зависит от calibration?
- <sup>5</sup> Система trained closed-set или open-track с внешними данными?
- <sup>6</sup> Ошибки по attack-id, codec-id, speaker-id, language-id опубликованы?

#### Правило

Benchmark полезен только тогда, когда понятно, **какой именно сдвиг распределения** он измеряет.

### Базовая постановка CM

$$f_{\theta}(x) \rightarrow s_{\text{cm}}, \qquad \hat{y} = \begin{cases} \text{bona fide}, & s_{\text{cm}} \geq \tau, \\ \text{spoof}, & s_{\text{cm}} < \tau. \end{cases}$$

- Вход: waveform или time-frequency representation.
- Выход: score, не обязательно calibrated probability.
- Главная проблема: spoof class открыт. В train есть только некоторые генераторы и каналы.
- Поэтому обычная binary CE может учить shortcut по генератору, а не "признак синтетичности".

Источник: [Zhang et al., OC-Softmax motivation: unknown attacks and distribution mismatch](https://arxiv.org/abs/2010.13995)

# Где живут spoofng artifacts

#### Спектральные

- overly smooth formants
- vocoder high-frequency noise
- phase inconsistency
- codec/resampling traces

#### Временные и речевые

- unnatural prosody
- boundary artifacts
- phoneme duration shifts
- local splice discontinuities
- Хорошая CM-система должна видеть и local micro-artifacts, и long-context inconsistencies.

Источник: [Jung et al., AASIST motivation: artifacts can reside in spectral or temporal domains](https://arxiv.org/abs/2110.01200)

# Frontends: от cepstra к raw waveform

| Frontend                        | Что<br>кодирует                                     | Комментарий                                                                          |
|---------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------|
| MFCC                            | mel-scaled<br>cepstra                               | speech<br>ASR<br>heritage;<br>может<br>сглажи<br>вать<br>high-frequency<br>artifacts |
| LFCC                            | linear<br>flterbanks<br>+<br>cepstra                | сохраняет<br>больше<br>high-frequency<br>detail;<br>сильный<br>ASVspoof<br>baseline  |
| CQCC/CQT                        | constant-Q<br>frequency<br>resolution               | удобен<br>для<br>широкого<br>частотного<br>диапазона<br>и<br>replay/LA<br>artifacts  |
| Mel<br>spectrogram<br>/<br>LFBE | 2D<br>time-frequency<br>map                         | хорошо<br>совместим<br>с<br>CNN/ResNet/ViT                                           |
| Raw<br>waveform                 | learned<br>flters,<br>Sinc,<br>CNN                  | может<br>найти<br>признаки,<br>которые<br>handcrafted<br>features<br>теряют          |
| SSL<br>embeddings               | wav2vec2,<br>HuBERT,<br>WavLM,<br>XLS-R,<br>Whisper | сильные<br>general-purpose<br>speech<br>representations                              |

Источник: [Gu et al., ALLM4ADD related work: handcrafted features, deep features, LCNN, RawNet2, SSL encoders](https://ar5iv.labs.arxiv.org/html/2505.11079v2)

# Mel/MFCC/LFCC: почему "спектрограмма" не одна

- **Mel**: полезна для speech perception и ASR, но компрессирует высокие частоты, где могут жить vocoder artifacts.
- **MFCC**: decorrelated cepstral representation; сильна в классическом speech processing, но может терять тонкие детали.
- **LFCC**: linear scale дает более равномерное покрытие частот, поэтому часто сильнее для LA anti-spoofng.
- **LFBE/melspec + CNN**: сохраняют spatial time-frequency structure, но требуют careful augmentation.

#### Практический вывод

Если сравнивать melspec-based подходы, нужно фиксировать FFT, hop, window, frequency range, normalization и augmentation. Иначе сравнение архитектур нечестное.

Источник: [ASVspoof 5 baseline context and common CM frontends](https://arxiv.org/abs/2408.08739)

### CQT, CQCC и GQT-подходы

- CQT дает логарифмически меняющееся частотное разрешение: узкие bins на низких частотах и широкие на высоких.
- CQCC применяет cepstral transform к CQT-like spectrum и исторически был сильной spoofng countermeasure feature.
- GQT/GF-style features обобщают идею на более гибкие time-frequency tilings.
- Ограничение: feature engineering может быть уязвим к новым генераторам и codecs, если backend не учит robust representation.

Источник: [ASVspoof 2019 analyses mention CQCC baseline lineage and anti-spoofng frontends](https://www.asvspoof.org/asvspoof2019/ASRU-2019-special-session.html)

# Backends: что классифицирует признаки

| Backend                       | Типичная<br>роль                                                     | Риск                                            |
|-------------------------------|----------------------------------------------------------------------|-------------------------------------------------|
| GMM<br>/<br>SVM               | классический<br>baseline<br>поверх<br>cepstra                        | слабая<br>выразительность                       |
| LCNN                          | популярный<br>CNN<br>baseline<br>для<br>spectrogram-like<br>features | overft<br>к<br>протоколу                        |
| ResNet<br>/<br>SE-ResNet      | устойчивые<br>2D<br>CNN<br>backbones                                 | требуется<br>хороший<br>frontend                |
| RawNet2                       | raw<br>waveform<br>end-to-end                                        | чувствителен<br>к<br>длине<br>и<br>augmentation |
| AASIST<br>/<br>RawGAT-ST      | graph<br>attention<br>over<br>spectral/temporal<br>nodes             | сложнее<br>интерпретировать                     |
| Transformer<br>/<br>Conformer | long-context<br>modeling                                             | data<br>hungry                                  |
| SSL<br>+<br>head              | pretrained<br>representation<br>+<br>small<br>classifer              | leakage<br>of<br>pretraining<br>biases          |

Источник: [media-sec-lab Audio-Deepfake-Detection curated repository: methods, datasets, audio large models](https://github.com/media-sec-lab/Audio-Deepfake-Detection)

# Data augmentation: не украшение, а часть threat model

- Additive noise, RIR, reverberation: имитируют реальную среду.
- Codec / bandwidth / FIR fltering: имитируют телефонию, social platforms, compression.
- RawBoost: нелинейная convolutive noise, impulsive noise, stationary noise.
- Speed perturbation и chunking: помогают speaker/ASV и short-utterance robustness.
- Neural codec augmentation становится важной после Encodec/codec laundering.

Источник: [Kondratev, Aliyev, Intema ASVspoof5 system description: FIR, RawBoost, codecs, WavLM, fusion](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf)

# RawNet2: мотивация

#### Проблема

ASVspoof 2019 показал, что некоторые LA attacks избегают handcrafted countermeasures. Нужно дать модели доступ к waveform-level cues.

- RawNet2 принимает raw audio и учит фильтры напрямую.
- Идея: не фиксировать заранее, какие spectral/phase/time cues важны.
- В ASVspoof 2019 LA RawNet2 особенно помогал на сложных A17 attacks; fusion с baseline CM давал один из лучших результатов.

Источник: [Tak et al., "End-to-end anti-spoofng with RawNet2", 2020](https://arxiv.org/abs/2011.01108)

# RawNet2: схема системы

- Первый слой работает как learnable flterbank.
- Residual blocks извлекают local temporal patterns.
- GRU агрегирует sequence-level evidence.
- Loss обычно CE/weighted CE; результат часто улучшается fusion с LFCC/CQCC systems.

Источник: [Ofcial RawNet2 anti-spoofng implementation](https://github.com/eurecom-asp/rawnet2-antispoofing)

# RawNet2: что взять в практику

- Raw waveform models не отменяют feature-based systems: они часто комплементарны.
- Нужно контролировать duration mismatch: training chunks и evaluation chunks влияют на результат.
- Нужна сильная augmentation, иначе raw model легко учит channel shortcuts.
- Полезно анализировать per-attack performance: общий EER может скрыть провал на одной атаке.

Источник: [Tak et al., RawNet2 for ASVspoof 2019 LA](https://arxiv.org/abs/2011.01108)

# AASIST: мотивация

#### Зачем graph attention

Spoofng artifacts могут быть и спектральными, и временными. Ensemble "один детектор на один тип артефакта" работает, но дорог и плохо переносится.

- AASIST строит integrated spectro-temporal graph attention.
- Использует RawNet2-like encoder для представлений.
- Heterogeneous stacking graph attention layer моделирует temporal и spectral nodes.
- Stack node и max graph operation помогают агрегировать evidence.

Источник: [Jung et al., "AASIST: Audio Anti-Spoofng using Integrated Spectro-Temporal Graph Attention Networks", 2021](https://arxiv.org/abs/2110.01200)

# AASIST: схема

- Graph view позволяет явно моделировать связи между time frames и spectral bands.
- AASIST-L показывает, что архитектурный bias может быть важнее размера: lightweight variant с 85K параметров тоже конкурентен.

Источник: [EURECOM AASIST publication page: AASIST-L and relative improvement summary](https://www.eurecom.org/en/publication/6696)

# AASIST2: short utterance robustness

- Проблема: многие CM train/eval используют фиксированную длительность, но реальные входы короткие и переменные.
- AASIST2 заменяет residual blocks на **Res2Net blocks**, чтобы извлекать multi-scale features.
- Training strategies: Dynamic Chunk Size и Adaptive Large Margin Fine-Tuning.
- Цель: сохранить качество на обычных utterances и улучшить короткие.

Источник: [Zhang et al., "Improving Short Utterance Anti-Spoofng with AASIST2", 2023/2024](https://arxiv.org/abs/2309.08279)

### AASIST3: KAN, SSL, regularization

- AASIST3 усиливает AASIST с Kolmogorov-Arnold Networks, дополнительными слоями, encoders и pre-emphasis.
- Мотивация ASVspoof 2024: deepfake attacks, TTS/VC vulnerability, need for stronger ASV security.
- В статье указаны minDCF 0.5357 closed и 0.1414 open для ASVspoof 2024 setting.
- Практический takeaway: современный strong system это уже не один блок, а аккуратная смесь SSL features, архитектурного bias и regularization.

Источник: [Borodin et al., "AASIST3: KAN-Enhanced AASIST ...", 2024, revised 2026](https://arxiv.org/abs/2408.17352)

### Почему CE не всегда достаточно

$$\mathcal{L}_{CE} = -\sum_{i} y_{i} \log p_{\theta}(y_{i} \mid x_{i})$$

- CE хорошо разделяет known bona fde и known spoof в train.
- Но spoof class открыт: новый генератор может не быть похож на train spoof.
- CE может запомнить generator/channel artifacts, а не общие признаки synthetic speech.
- Это мотивирует margin losses, one-class learning, contrastive learning и domain generalization.

Источник: [Zhang et al., OC-Softmax: unknown attack generalization motivation](https://arxiv.org/abs/2010.13995)

# OC-Softmax: компактная bona fde область

#### Идея

Не пытаться описать все spoof attacks. Вместо этого сделать bona fde embeddings компактными, а spoof отталкивать от bona fde center с angular margin.

$$\cos(\theta_i) = \hat{w}^\top \hat{x}_i$$

- Bona fde samples притягиваются к center.
- Spoof samples pushed away с margin.
- На ASVspoof 2019 LA авторы сообщают EER 2.19% без augmentation и без ensemble.

Источник: [Zhang, Jiang, Duan, "One-class Learning Towards Synthetic Voice Spoofng Detection", 2020](https://arxiv.org/abs/2010.13995)

# SAMO: speaker-aware one-class learning

#### Ограничение OC-Softmax

Bona fde речь не один кластер: спикеры, языки, акценты и каналы создают естественную multi-modal структуру.

- SAMO = Speaker Attractor Multi-Center One-Class learning.
- Bona fde embeddings кластеризуются вокруг speaker attractors.
- Spoof attacks отталкиваются от всех attractors.
- Inference может использовать enrollment speaker embedding или nearest training attractor.

Источник: [Ding, Zhang, Duan, "SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofng", 2022](https://arxiv.org/abs/2211.02718)

# Margin losses в ASV и CM

| Loss                       | Где<br>полезен                           | Комментарий                                          |
|----------------------------|------------------------------------------|------------------------------------------------------|
| BCE/CE                     | базовая<br>CM<br>classifcation           | простая,<br>но<br>может<br>плохо<br>OOD<br>обобщать  |
| Focal<br>loss              | class<br>imbalance,<br>hard<br>examples  | чувствителен<br>к<br>calibration                     |
| AM/AAM-Softmax             | speaker<br>verifcation<br>embeddings     | увеличивает<br>angular<br>separation                 |
| OC-Softmax                 | unknown<br>spoof<br>generalization       | моделирует<br>bona<br>fde<br>как<br>compact<br>class |
| SAMO                       | speaker-diverse<br>bona<br>fde           | multi-center<br>one-class<br>formulation             |
| Contrastive<br>/<br>SupCon | representation<br>robustness             | требует<br>грамотных<br>positive/negative<br>pairs   |
| Multi-task                 | CM<br>+<br>ASV<br>+<br>codec/lang/domain | риск<br>negative<br>transfer                         |

Источник: [Kondratev, Aliyev: CE + OC-Softmax for CM, AAM-Softmax for ASV in ASVspoof5 system](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf)

# SSL encoders: wav2vec2, HuBERT, WavLM, XLS-R, Whisper

- SSL speech encoders дают сильные frame-level representations без обучения с нуля.
- Часто используется схема: **frozen/partially fne-tuned encoder + pooling + classifer**.
- WavLM-base в ASVspoof5 system description: CNN feature encoder + 12 transformer blocks, mean pooling, dropout, linear head.
- Whisper encoder в audio LLM используется через log-Mel и transformer encoder.

Источник: [Kondratev, Aliyev, ASVspoof5 system: WavLM-base CM model and training details](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf)

### Почему SSL помогает и почему может ломаться

#### Плюсы

- robust acoustic representation
- multilingual pretraining
- better sample efciency
- transferable low-level cues

#### Риски

- ASR objectives не равны spoof objectives
- high-level semantics может скрыть artifacts
- pretraining domain bias
- fne-tuning overft к benchmark
- Поэтому в сильных системах часто сравнивают layer aggregation, freeze policy, pooling и augmentation.

Источник: [ALLM4ADD related work: SSL features such as wav2vec2 and HuBERT for audio deepfake detection](https://ar5iv.labs.arxiv.org/html/2505.11079v2)

# Calibration: score не равен вероятность

- Softmax score удобен, но часто плохо calibrated.
- minDCF может быть хорошим, а actDCF/Cllr плохими.
- Для production нужны calibration set, prior assumptions, threshold policy и мониторинг drift.
- ASVspoof5 Intema system прямо показывает разницу softmax vs LLR scores для actDCF/Cllr.

#### Практический вывод

Не отдавать "probability of fake" пользователю без calibration и confdence policy.

Источник: [Kondratev, Aliyev, ASVspoof5 system: softmax scores vs LLR discussion](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf)

# Spoofng-aware speaker verifcation

#### Обычный ASV-вопрос

"Этот голос принадлежит заявленному спикеру?"

#### SASV-вопрос

"Это **bona fde** речь **заявленного** спикера?"

- В обычном ASV spoof может выглядеть как target.
- В обычном CM target/non-target identity игнорируется.
- SASV вводит trial set: target bona fde, non-target bona fde, spoofed non-target.

Источник: [Jung et al., SASV 2022](https://arxiv.org/abs/2203.14732)

# SASV 2022: две baseline-стратегии

#### Score fusion

Предобученные ASV и CM независимы. Итоговый score - сумма или calibrated fusion speaker similarity и CM score.

#### DNN fusion

Использует speaker embeddings и spoofng embeddings, затем обучает backend для SASV decision.

- Challenge дал open-source ASV/CM subsystems.
- 23 команды submitted solutions; top system снизила EER conventional ASV с 23.83% до 0.13% в SASV setup.

Источник: [Jung et al., SASV 2022 results](https://arxiv.org/abs/2203.14732)

# SASV: fusion или end-to-end?

| Strategy                     | Плюсы                                                          | Риски                                                        |
|------------------------------|----------------------------------------------------------------|--------------------------------------------------------------|
| CM<br>then<br>ASV<br>cascade | просто,<br>интерпретируемо,<br>можно<br>переиспользовать<br>CM | hard<br>threshold<br>может<br>убить<br>bona<br>fde<br>target |
| Score-level<br>fusion        | гибко,<br>хорошо<br>работает<br>в<br>challenge                 | calibration<br>и<br>overft<br>к<br>dev/progress              |
| Embedding-level<br>fusion    | учит<br>interaction<br>между<br>speaker<br>и<br>spoof<br>cues  | нужны<br>trial<br>labels<br>и<br>careful<br>sampling         |
| End-to-end<br>SASV           | единая<br>цель,<br>потенциально<br>луч<br>ший<br>optimum       | сложно<br>собрать<br>данные,<br>риск<br>shortcut<br>learning |

Источник: [Teng et al., "SA-SASV: An End-to-End Spoof-Aggregated Spoofng-Aware Speaker Verifcation System", 2022](https://arxiv.org/abs/2203.06517)

### ASVspoof5 SASV solution: Intema

- Track 1: stand-alone CM, Track 2: combined SASV.
- CM: raw audio, WavLM-base, mean pooling, dropout + linear, CE + OC-Softmax, FIR и RawBoost augmentation.
- ASV: ResNet modifcation + CAM++, 80-dim flterbanks, VoxCeleb2 + Russian Common Voice subset, AAM-Softmax, large-margin fne-tuning.
- Fusion: cascade, score-sum, и Power Weighted Score Fusion:

$$s_{\mathsf{sasv}} = s_{\mathsf{asv}} \cdot s_{\mathsf{cm}}^q.$$

Reported fnal: Track 1 open minDCF 0.0936; Track 2 open min a-DCF 0.1203.

Источник: [Kondratev, Aliyev, "Intema System Description for the ASVspoof5 Challenge", ASVspoof 2024](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf)

# SASV: что важно для своей системы

- <sup>1</sup> Не смешивать score scales: ASV cosine/logit и CM probability имеют разную статистику.
- <sup>2</sup> Делать dev split, похожий на deployment: target/non-target/spoof priors.
- <sup>3</sup> Анализировать три ошибки отдельно: target reject, non-target accept, spoof accept.
- <sup>4</sup> Проверять spoofed target и spoofed non-target отдельно, если протокол позволяет.
- <sup>5</sup> Писать policy: что делать с low-confdence и OOD samples.

Источник: [SASV Challenge 2022 Evaluation Plan](https://arxiv.org/abs/2201.10283)

# Partial spoof: почему это отдельная задача

#### Атака

Подменяется не вся запись, а короткий участок: одно слово, сумма платежа, имя, команда, политическое утверждение.

- Utterance-level detector видит в основном bona fde сигнал.
- Спуфовый участок может быть 20-640 ms или несколько слов.
- Нужна не только классификация, но и **локализация** manipulated regions.
- Для forensic и journalism важно объяснить "где именно подделано".

Источник: [Zhang et al., "The PartialSpoof Database and Countermeasures ...", 2022](https://arxiv.org/abs/2204.05177)

# PartialSpoof database

- Сценарий: synthesized/transformed speech segments embedded into bona fde utterance.
- Расширенная версия добавляет segment labels на разных temporal resolutions.
- Рассмотрены шесть temporal resolutions от 20 ms до 640 ms.
- Предлагается CM, который одновременно использует utterance-level и segment-level labels.
- Reported utterance-level EER: 0.77% на PartialSpoof и 0.90% на ASVspoof 2019 LA.

Источник: [Zhang et al., PartialSpoof, IEEE/ACM TASLP version](https://arxiv.org/abs/2204.05177)

# Half-Truth и LlamaPartialSpoof

| Dataset                | Мотивация                                                                                                        | Что<br>проверяет                                                                                           |
|------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| Half-Truth<br>/<br>HAD | attacker<br>inserts<br>small<br>fabricated<br>clips<br>into<br>real<br>speech                                    | utterance<br>detection<br>+<br>manipulated<br>region<br>localization                                       |
| LlamaPartialSpoof      | fake<br>speech<br>generation<br>from<br>attacker's<br>perspective,<br>LLM-driven<br>disinformation<br>simulation | fully<br>and<br>partially<br>fake<br>speech;<br>CM<br>robustness<br>to<br>text/TTS/concatenation<br>biases |

- LlamaPartialSpoof: 130 часов, fully + partially fake speech; best reported EER current CM 24.49% in unseen scenarios.
- Урок: partial spoof ломает assumption "один label на весь utterance".

Источники: [Half-Truth, 2021;](https://arxiv.org/abs/2104.03617) [HAD dataset;](https://zenodo.org/records/10377492) [LlamaPartialSpoof, 2024/2025](https://arxiv.org/abs/2409.14743)

### Архитектура для partial spoof

- Multi-task loss: utterance CE + frame/segment CE + boundary consistency.
- Multiple resolutions: short windows ловят splices, long windows ловят prosody/context.
- Evaluation: segment-level F1/IoU плюс utterance-level EER.

Источник: [PartialSpoof: segment labels at multiple temporal resolutions](https://arxiv.org/abs/2204.05177)

### Singing voice deepfake detection

#### Почему speech detector не переносится напрямую

Синтетическое пение часто публикуется как песня с инструментальным сопровождением; музыка маскирует артефакты, а singing voice имеет другую фонетику, pitch dynamics и timbre control.

- В пении важны pitch contour, vibrato, formants, accompaniment leakage, vocal separation.
- Attack surface: unauthorized artist voice usage, AI covers, synthetic singers, voice conversion for music.
- Speech CM может сильно просесть на singing data.

Источник: [Zang et al., "SingFake: Singing Voice Deepfake Detection", 2023/2024](https://arxiv.org/abs/2309.07525)

# SingFake

- First curated in-the-wild dataset for Singing Voice Deepfake Detection.
- 28.93 часов bona fde и 29.40 часов deepfake song clips.
- 5 языков, 40 singers, train/validation/test split with multiple scenarios.
- Speech countermeasure systems trained on speech lag behind; training on SingFake improves results.
- Remaining challenges: unseen singers, codecs, languages, musical contexts.

Источник: [Zang et al., SingFake, ICASSP 2024](https://arxiv.org/abs/2309.07525)

# CtrSVDD и SVDD Challenge

- CtrSVDD: controlled benchmark for singing voice deepfake detection.
- 47.64 часов bona fde и 260.34 часов deepfake singing vocals.
- 14 deepfake methods и 164 singer identities.
- Baseline system supports fexible front-end features; experiments highlight feature selection and generalization to unseen deepfake methods.
- SVDD 2024 challenge разделяет controlled и wild tracks, расширяя постановку SingFake/CtrSVDD.

Источники: [CtrSVDD, Interspeech 2024](https://arxiv.org/abs/2406.02438); [SVDD Challenge 2024 Evaluation Plan](https://challenge.singfake.org/SVDD_Challenge_2024_Eval_Plan_v0.1.pdf)

# Что менять для SVDD

- Добавить vocal separation или train/test на mixtures отдельно от isolated vocals.
- Учитывать pitch и long-range musical structure, а не только speech phonetics.
- Делать splits по singer, language, genre, accompaniment и generator.
- Проверять robustness к compression, remixing, mastering и platform processing.
- Не считать speech anti-spoofng SOTA baseline достаточным для music-domain deployment.

Источник: [Recent SVDD direction: fullband-subband modeling and high-frequency cues for SingFake detection, 2026](https://arxiv.org/abs/2604.04841)

### Зачем audio LLM в deepfake detection

- Классические CM дают score, но плохо объясняют решение.
- In-the-wild audio требует reasoning over context: noise, language, speaker style, recording device, prompt, semantics.
- Audio LLM потенциально дает:
  - natural-language rationale,
  - few-shot/in-context adaptation,
  - routing for OOD samples,
  - multimodal and forensic workfow integration.
- Но zero-shot audio LLM не обучены на spoof artifacts и часто угадывают.

Источник: [Gu et al., "ALLM4ADD", 2025](https://arxiv.org/abs/2505.11079)

### ALLM4ADD: zero-shot провал как мотивация

- Авторы задают вопрос: могут ли Audio Large Language Models решать audio deepfake detection?
- Проверяют Qwen-Audio и Qwen2-Audio series с prompt templates вроде "Is this audio fake or real?"
- Zero-shot performance низкая: модели не предназначены для fake audio artifacts.
- Вывод: нужно reformulate ADD как audio question answering и делать supervised fne-tuning.

Источник: [ALLM4ADD HTML: zero-shot evaluation and task formulation](https://ar5iv.labs.arxiv.org/html/2505.11079v2)

### ALLM4ADD: система

- Формат: user prompt + audio feature tokens, output constrained to Fake/Real.
- Supervised fne-tuning with language modeling loss.
- LoRA for LLM component; audio encoder can be trainable or frozen depending on variant.
- Особенно полезно в data-scarce scenarios.

Источник: [ALLM4ADD method section](https://ar5iv.labs.arxiv.org/html/2505.11079v2)

### HIR-SDD: human-inspired reasoning

- Проблема: SDD methods плохо обобщают на новые audio domains/generators и почти не дают human-perceptible explanations.
- HIR-SDD сочетает Large Audio Language Models с chain-of-thought reasoning, полученным из human-annotated dataset.
- Цель: не только binary score, но и разумные объяснения: какие perceptual cues указывают на bona fde/spoof.
- Это важно для forensic workfows, где решение нужно аргументировать.

Источник: [Dvirniak et al., "Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning", 2026](https://arxiv.org/abs/2603.10725)

# ICLAD: comparison-guided in-context detection

- ICLAD использует audio language model для training-free generalization к unseen in-the-wild deepfakes.
- Core idea: Pairwise Comparative Reasoning генерировать и сопоставлять evidence за real и fake гипотезы.
- Dynamic routing: in-distribution samples идут к specialized detector, OOD samples к ALM.
- RAG/in-context examples помогают ALM принимать решение и давать textual rationale.
- На in-the-wild datasets авторы сообщают до 2x relative macro-F1 improvement over specialized detector.

Источник: [Chou, Zhu, Koppisetti, "ICLAD", 2026](https://arxiv.org/abs/2604.16749)

### Reasoning: польза и опасность

#### Польза

- объяснение для человека
- OOD triage
- few-shot adaptation
- аудиторский след

#### Опасность

- hallucinated cues
- persuasive wrong rationale
- leakage через prompt
- weak calibration

#### Практический принцип

LLM-rationale не должен заменять calibrated detector score. Он полезен как объясняющий и routing слой, но требует human evaluation и hallucination checks.

Источники: [HIR-SDD](https://arxiv.org/abs/2603.10725); [ICLAD](https://arxiv.org/abs/2604.16749); [ALLM4ADD](https://arxiv.org/abs/2505.11079)

# SpeechFake: масштаб и разнообразие

- ACL 2025 dataset for speech deepfake detection.
- 3M+ deepfake samples, 3000+ hours audio.
- 40 speech synthesis tools: TTS, voice conversion, neural vocoder.
- 46 languages.
- Авторы также исследуют, как generation methods, language diversity и speaker variation влияют на detection performance.

Источник: [Huang et al., "SpeechFake", ACL 2025](https://aclanthology.org/2025.acl-long.493/)

# ShiftySpeech: controlled distribution shifts

- 3000+ hours synthetic speech.
- 7 distribution shifts: reading style, podcast, YouTube, languages, demographics and more.
- 6 TTS systems, 12 vocoders, 3 languages.
- Мотивация: ASVspoof-like benchmarks дают низкие ошибки, но не проверяют широкий набор real-world variability.
- Хороший benchmark для controlled robustness analysis.

Источник: [Garg et al., "ShiftySpeech", Hugging Face dataset card and arXiv:2502.05674](https://huggingface.co/datasets/ash56/ShiftySpeech)

# Deepfake-Eval-2024: in-the-wild benchmark

- 2024-circulated deepfakes from social media and detection platform users.
- 45h video, 56.5h audio, 1975 images; 88 websites; 52 languages.
- Open-source SOTA detectors drop sharply: arXiv abstract reports AUC drop 48% for audio compared to previous benchmarks.
- Hugging Face card: gated access; dataset is public but requires accepting conditions; evaluation-only use, not training.

Источники: [Chandra et al., Deepfake-Eval-2024;](https://arxiv.org/abs/2503.02857) [HF dataset card](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024)

# 6KSFX и dataset repos

- 6KSFX Synth Dataset: 2025 dataset/code repository for trimming, down-mixing and labeling synthetic sound samples; citation arXiv:2501.17198.
- media-sec-lab Audio-Deepfake-Detection: curated repository with surveys, datasets, preprocessing, feature extraction, end-to-end methods, network training, audio large models.
- Такие repos полезны не как ground truth, а как map of the feld: быстро видеть новые datasets and baselines.

Источники: [6KSFX GitHub](https://github.com/nellyngz95/6KSFX); [Audio-Deepfake-Detection GitHub](https://github.com/media-sec-lab/Audio-Deepfake-Detection)

### Что должна решать хорошая CM-система

- Detect known and unknown TTS/VC/deepfake attacks.
- Survive codec, bandwidth, noise, reverberation, replay and platform laundering.
- Work across languages, speakers, accents and speaking styles.
- Handle short utterances and partial spoof.
- Provide calibrated scores and uncertainty.
- Integrate with ASV for target-speaker verifcation.
- Produce interpretable evidence when used for forensic or moderation.

### Сравнение подходов

| Подход                                                   | Когда<br>хорош                                                                        | Где<br>слаб                                                                        |
|----------------------------------------------------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| LFCC/CQCC<br>+<br>LCNN/GMM<br>Melspec<br>+<br>CNN/ResNet | быстрый<br>baseline,<br>fair<br>comparison<br>удобно<br>и<br>сильно<br>при<br>хороших | плохо<br>переносит<br>новые<br>domains<br>high-frequency/phase<br>artifacts<br>мо  |
| RawNet/AASIST                                            | данных<br>end-to-end<br>artifacts,<br>spectro<br>temporal<br>graph<br>bias            | гут<br>потеряться<br>needs<br>careful<br>augmentation<br>and<br>duration<br>policy |
| SSL<br>+<br>classifer                                    | strong<br>representation,<br>sample<br>efciency                                       | pretraining<br>mismatch,<br>calibration                                            |
| One-class/SAMO                                           | unknown<br>spoof<br>generalization                                                    | bona<br>fde<br>diversity<br>and<br>enrollment<br>complexity                        |
| Audio<br>LLM/reasoning                                   | explanations,<br>OOD<br>routing,<br>in<br>context<br>adaptation                       | hallucinations,<br>compute,<br>weak<br>zero<br>shot                                |

# Adversarial и laundering attacks

- ASVspoof 5 впервые включает adversarial attacks в challenge design.
- Replay/laundering может сделать deepfake более похожим на bona fde по признакам детектора.
- Neural codecs могут скрывать или менять генераторные артефакты.
- Атакующий может оптимизировать синтез под surrogate detector.
- Поэтому robust evaluation должен включать adaptive attacks, codecs, replay, compression and post-processing.

Источники: [ASVspoof 5 adversarial attacks;](https://arxiv.org/abs/2408.08739) [Replay Attacks Against Audio Deepfake Detection, 2025](https://huggingface.co/papers/2505.14862)

# Production pipeline: не один детектор

- Quality checks: clipping, duration, SNR, sample rate, codec, speech/music separation.
- CM ensemble: independent frontends and encoders reduce shared failure modes.
- OOD router: send suspicious domain shifts to conservative policy or human/LLM-assisted review.

# Как спроектировать сильный эксперимент

- Train/dev/eval split by generator, speaker, language, text and channel.
- Report pooled and per-condition EER/minDCF/a-DCF.
- Include calibration metrics, not only min metrics.
- Evaluate short utterances and partial-spoof localization.
- Test cross-dataset: ASVspoof, in-the-wild, SpeechFake/ShiftySpeech.
- Run ablations: frontend, encoder layer, loss, augmentation, fusion.
- Publish protocols and score fles, not only fnal table.

# Open problems

- **Open-world generalization**: new generators, new languages, new domains.
- **Calibration under drift**: thresholds age quickly.
- **Partial and semantic spoof**: small edits with high consequence.
- **Adversarial robustness**: attackers optimize against detectors.
- **Interpretability**: forensic explanations without hallucination.
- **Dataset governance**: consent, licensing, misuse restrictions, privacy.
- **Unifed SASV**: target identity and authenticity in one reliable score.

# Рекомендуемый порядок чтения

- ASVspoof 2019 summary + evaluation plan: базовые сценарии, t-DCF.
- ASVspoof 2021 evaluation plan + summary: LA/PA/DF and no matched training.
- RawNet2, AASIST, AASIST2, AASIST3: эволюция архитектур.
- OC-Softmax and SAMO: losses for unknown attacks.
- SASV 2022 and ASVspoof5: integrated speaker verifcation.
- PartialSpoof, LlamaPartialSpoof, SingFake, CtrSVDD: новые threat models.
- ALLM4ADD, HIR-SDD, ICLAD: reasoning and audio LLM direction.

### Takeaways

- Deepfake detection security/OOD задача, а не обычный supervised classifcation benchmark.
- Сильная система сочетает frontends, raw/SSL encoders, losses, augmentation, calibration and fusion.
- ASVspoof 5 с adversarial attacks и SASV поднимает планку: нужно думать как атакующий.
- Partial spoof и singing voice требуют специальных протоколов, а не переиспользования speech CM вслепую.
- Audio LLM полезны для explanations и routing, но zero-shot detection слаб и требует fne-tuning/human evaluation.

### Источники: соревнования и бенчмарки I

Todisco et al. [ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.](https://arxiv.org/abs/1904.05441) 2019.

ASVspoof. [ASVspoof 2019 Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf). 2019.

Delgado et al. [ASVspoof 2021 Evaluation Plan](https://arxiv.org/abs/2109.00535). 2021.

Yamagishi et al. [ASVspoof 2021: accelerating progress in spoofed and deepfake speech detection.](https://arxiv.org/abs/2109.00537) 2021.

ASVspoof. [ASVspoof5 Evaluation Plan Phase 2](https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf). 2024.

Wang et al. [ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale](https://arxiv.org/abs/2408.08739). 2024.

Jung et al. [SASV Challenge 2022 Evaluation Plan.](https://arxiv.org/abs/2201.10283) 2022.

Jung et al. [SASV 2022: The First Spoofng-Aware Speaker Verifcation Challenge.](https://arxiv.org/abs/2203.14732) 2022.

Huang et al. [SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](https://aclanthology.org/2025.acl-long.493/). ACL 2025.

Garg et al. [ShiftySpeech dataset card.](https://huggingface.co/datasets/ash56/ShiftySpeech) 2025.

Chandra et al. [Deepfake-Eval-2024.](https://arxiv.org/abs/2503.02857) 2025/2026.

Deepfake-Eval-2024. [Hugging Face dataset card](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024).

media-sec-lab. [Audio-Deepfake-Detection curated repository](https://github.com/media-sec-lab/Audio-Deepfake-Detection).

# Источники: соревнования и бенчмарки II

Garcia-Sihuay, Reiss. [6KSFX Synth Dataset repository](https://github.com/nellyngz95/6KSFX). 2025.

### Источники: модели, losses, SASV, partial, singing, LLM I

Tak et al. [End-to-end anti-spoofng with RawNet2.](https://arxiv.org/abs/2011.01108) 2020.

EURECOM. [rawnet2-antispoofng code](https://github.com/eurecom-asp/rawnet2-antispoofing).

Jung et al. [AASIST: Audio Anti-Spoofng using Integrated Spectro-Temporal Graph Attention Networks.](https://arxiv.org/abs/2110.01200) 2021.

Zhang et al. [Improving Short Utterance Anti-Spoofng with AASIST2](https://arxiv.org/abs/2309.08279). 2023/2024.

Borodin et al. [AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization.](https://arxiv.org/abs/2408.17352) 2024/2026.

Zhang, Jiang, Duan. [One-class Learning Towards Synthetic Voice Spoofng Detection.](https://arxiv.org/abs/2010.13995) 2020.

Ding, Zhang, Duan. [SAMO: Speaker Attractor Multi-Center One-Class Learning for Voice Anti-Spoofng.](https://arxiv.org/abs/2211.02718) 2022.

Kondratev, Aliyev. [Intema System Description for the ASVspoof5 Challenge: Power Weighted Score Fusion](https://www.isca-archive.org/asvspoof_2024/aliyev24_asvspoof.pdf). ASVspoof 2024.

Zhang et al. [The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance.](https://arxiv.org/abs/2204.05177) 2022.

Yi et al. [Half-Truth: A Partially Fake Audio Detection Dataset.](https://arxiv.org/abs/2104.03617) 2021.

Luong et al. [LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation](https://arxiv.org/abs/2409.14743). 2024/2025.

Zang et al. [SingFake: Singing Voice Deepfake Detection](https://arxiv.org/abs/2309.07525). 2023/2024.

Zang et al. [CtrSVDD: A Benchmark Dataset and Baseline Analysis for Controlled Singing Voice Deepfake Detection](https://arxiv.org/abs/2406.02438). 2024.

### Источники: модели, losses, SASV, partial, singing, LLM II

SVDD Challenge. [SVDD 2024 Evaluation Plan](https://challenge.singfake.org/SVDD_Challenge_2024_Eval_Plan_v0.1.pdf). 2024.

Gu et al. [ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection](https://arxiv.org/abs/2505.11079). 2025.

Dvirniak et al. [Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning](https://arxiv.org/abs/2603.10725). 2026.

Chou, Zhu, Koppisetti. [ICLAD: In-Context Learning with Comparison-Guidance for Audio Deepfake Detection.](https://arxiv.org/abs/2604.16749) 2026.