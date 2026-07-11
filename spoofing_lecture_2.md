# Voice Biometric Anti-Spoofing Spoofing-Aware Speaker Verification

## Where we are going

#### I — The detection problem

- *•* ASV vs. CM vs. SASV
- *•* Front-end / back-end / score
- *•* Spoof detection *is* hypothesis testing
- *•* EER, ROC, AUC, DCF / t-DCF

#### II — Replay (recorded speech)

- *•* Threat geometry; room impulse response
- *• T*60, reverberation, energy decay
- *•* Vocoder-based channel estimation
- *•* One-class back-ends: GMM, VAE

#### III — Synthesis (TTS / VC)

- *•* Sub-band artefacts; band-limited EER
- *•* Fusion / ensembles

#### IV — Architectures

- *•* LightCNN (MFM), RawNet2, sinc-layer, FMS
- *•* GNN → GCN → GAT → AASIST

#### V — SSL front-ends

- *•* Wav2Vec 2.0; SSL + AASIST
- *•* Why replay and synthesis live in di!erent places

A running thread: the attacker adds a channel — a loudspeaker, a vocoder, a neural synthesizer — and the countermeasure's whole job is to detect the statistics that channel leaves behind.

<span id="page-2-0"></span>The Detection Problem and Its Metrics

# Two decisions, two systems

# Automatic Speaker Verification (ASV)

Is this speaker the *enrolled owner*, or another *real* person? Guards against impostor attacks (a di!erent human voice).

#### Countermeasure (CM)

Is this speech *real* (bona-fide, genuine) or *not*? Guards against synthesized and replayed speech.

Spoofing-Aware Speaker Verification (SASV) is the complete authentication method: ASV and CM combined. Throughout, we focus on the *CM* — but its metrics and its fusion with ASV are the same detection-theory story.

# The anatomy of a countermeasure

#### Front-end (input features)

- *•* Raw waveform *x*[*n*]
- *•* Spectrogram / STFT magnitude
- *•* Cepstral coe!cients (LFCC, MFCC, CQCC)
- *•* Self-supervised embeddings (Wav2Vec, HuBERT)

The score is, ideally, monotone in a *likelihood ratio*. The next slide makes that precise.

#### Back-end (architecture)

- *•* Gaussian Mixture Model (generative)
- *•* Deep neural network (discriminative)
- *•* One-class / anomaly model (VAE, OC-softmax)

Frame the trial as two competing hypotheses on the observation  $\boldsymbol{x}$ :

$$H_0: x \sim p(x \mid \text{spoof}), \qquad H_1: x \sim p(x \mid \text{bona-fide}).$$

The Neyman–Pearson lemma says the most powerful test — maximal detection rate at a fixed false-alarm rate — thresholds the likelihood ratio:

$$\Lambda(x) = \frac{p(x \mid \text{bona-fide})}{p(x \mid \text{spoof})} \underset{H_0}{\overset{H_1}{\geqslant}} \tau.$$

Every CM you will see is an *estimator of*  $\Lambda$  (GMM ratio, network logit, reconstruction probability), read out against a threshold  $\tau$ .

#### Two ways to be wrong

- False acceptance: a spoof scores  $\geq \tau$  the spoofer gets in.
- False rejection: a genuine owner scores  $< \tau$  the owner is locked out.

Sliding  $\tau$  trades one error for the other. There is no  $\tau$  that zeroes both — the distributions overlap. The whole game is the shape of that trade-off curve.

With score *s* and threshold ω (accept bona-fide if *s* ↑ ω ):

$$\text{FAR}(\tau) = \text{Pr}(s \geq \tau \mid \text{spoof}) \quad \text{(false accept)} \ \text{FRR}(\tau) = \text{Pr}(s < \tau \mid \text{bona-fide}) \quad \text{(false reject)}$$

- *•* FAR *decreases*, FRR *increases* in ω.
- *•* Equal Error Rate: the ω <sup>ω</sup> where they cross,

$$FAR(\tau^*) = FRR(\tau^*) \equiv EER.$$

*•* One scalar, threshold-free, lower is better ↔ — the field's default headline number.

EER is where the two error curves intersect.

Sweep ω and plot TPR against FPR: the *Receiver Operating Characteristic*. Two facts make it indispensable.

*•* AUC has a probabilistic meaning (Mann–Whitney):

$$AUC = \Pr(s^+ > s^-),$$

the chance a random bona-fide outscores a random spoof. 0*.*5 = chance, 1 = perfect.

*•* EER is the ROC's crossing with the anti-diagonal TPR = 1 ↗ FPR.

A confusion matrix is one *point* on this curve; the ROC is the whole family of operating points. <sup>0</sup> <sup>0</sup>*.*<sup>2</sup> <sup>0</sup>*.*<sup>4</sup> <sup>0</sup>*.*<sup>6</sup> <sup>0</sup>*.*<sup>8</sup> <sup>1</sup> <sup>0</sup>

EER weights both errors equally. Real deployments do not. The Detection Cost Function is the *Bayes risk* of the test:

$$\mathrm{DCF}(\tau) = \mathit{C}_{\mathsf{miss}} \, \pi_{\mathsf{tar}} \, \mathit{P}_{\mathsf{miss}}(\tau) + \mathit{C}_{\mathsf{fa}} \, (1 - \pi_{\mathsf{tar}}) \, \mathit{P}_{\mathsf{fa}}(\tau).$$

- $C_{miss}$ ,  $C_{fa}$ : costs of a miss / false alarm.
- $\pi_{tar}$ : prior probability of a target trial.
- Report min-DCF=  $\min_{\tau} \mathrm{DCF}(\tau)$ , normalized by the best "always decide" baseline.

#### The spoofing twist: t-DCF

A CM never runs alone; it sits in front of an ASV. The tandem DCF scores the *combined* system, propagating CM errors through the ASV's own miss/false-alarm behaviour:

t-DCF = 
$$f(P_{miss}^{cm}, P_{fa}^{cm}; ASV \text{ operating point}).$$

This is why ASVspoof challenges rank on min t-DCF, not EER alone.

# The detector's view in five lines

- 1. A CM is a *likelihood-ratio test*; Neyman–Pearson is the gold standard it estimates.
- 2. Two errors false accept (spoofer in) and false reject (owner out) trade o! in ω .
- 3. EER is the threshold-free balance point; AUC= Pr(*s*<sup>+</sup> *> s*→).
- 4. DCF is Bayes risk; t-DCF scores the CM *tandem* with the ASV.
- 5. Everything that follows is a better *estimator* of the same likelihood ratio.

Next: the first attack family — replay — and the physics that gives it away.

<span id="page-10-0"></span>[Replay Attacks: Recorded Speech and Room Acoustics](#page-10-0)

Part 2

## The replay threat model

A replay attack records a genuine utterance, then plays it back at the sensor. Two acoustic events stack up:

- *• Da acquisition* distance: attacker → talker, when the recording is captured (zone C).
- *• Ds presentation* distance: loudspeaker → ASV microphone, when it is replayed (zone A).

The signal therefore passes through *two* rooms and *two* transducers — and that extra channel is the fingerprint we exploit.

Schematic of replay geometry (zones A/B/C).

To a good approximation a room is linear and time-invariant. The microphone hears the source convolved with the *room impulse response h*, plus noise:

$$x(n) = s(n) * h(n) + v(n)$$

- *• s*(*n*): the clean speech the talker produced.
- *• h*(*n*): room + loudspeaker + microphone everything between mouth and ADC.
- *• v*(*n*): additive ambient noise.

A replay attack *multiplies* the impulse responses: genuine *h*room becomes *h*room ↔ *h*spk ↔ *h*room<sup>↑</sup> . More convolution = more spectral colouring to detect.

Convolution in time = multiplication in frequency: *X*(*f* ) = *S*(*f* )*H*(*f* ) + *V* (*f* ). This linearity is the lever for the whole

# Where the impulse response comes from: reflections

The microphone receives the same utterance many times over: the *direct* sound, then a discrete burst of early reflections (distinct geometric paths), decaying into a dense di!use tail of late reflections.

- *•* Direct: shortest path, sets the "dry" timbre.
- *•* Early (↭ 80 ms): few strong echoes; encode room geometry.
- *•* Late: statistically dense reverberation; encode room *size* and absorption.

Energy–time profile of a room impulse response.

**Echo**: a few *discrete, resolvable* repetitions (>50 ms apart) — you hear "hello ... hello".

**Reverberation**: reflections so dense they *merge* into a continuous decay — "hellooooo".

The standard summary of that decay is the reverberation time  $T_{60}$ : the time for sound energy to fall by 60 dB after the source stops. From Sabine's equation,

$$T_{60} = 0.161 \frac{V}{A}, \qquad A = \sum_{i} S_{i} \alpha_{i},$$

with volume V (m<sup>3</sup>) and absorption A summed over surfaces  $S_i$  with coefficients  $\alpha_i$ .

Energy decay curve (Schroeder):  $EDC(t)=\int_{t}^{\infty}h^{2}(\tau)\ d\tau;\ T_{60}$  is read from its -60 dB slope.

## A categorical parameterization of replay conditions

Replay corpora discretize the continuous acoustic space into labelled triplets and duples — so that "hard" and "easy" conditions can be sampled systematically.

| c}3<br>Environment<br>(S,<br>R,<br>Ds<br>)<br>{a,<br>b,<br>↓ |               |                 |                   | Attack<br>(Da,                                            | C}2<br>Q)<br>{A,<br>B,<br>↓ |                |                 |  |
|--------------------------------------------------------------|---------------|-----------------|-------------------|-----------------------------------------------------------|-----------------------------|----------------|-----------------|--|
|                                                              | a             | b               | c                 |                                                           | A                           | B              | C               |  |
| (m2)<br>S:<br>room<br>size<br>R:<br>T60<br>(ms)              | 2–5<br>50–200 | 5–10<br>200–600 | 10–20<br>600–1000 | Da:<br>attacker–talker<br>(cm)<br>Q:<br>device<br>quality | 10–50<br>perfect            | 50–100<br>high | ><br>100<br>low |  |
| Ds<br>:<br>talker–ASV<br>(cm)                                | 10–50         | 50–100          | 100–150           |                                                           |                             |                |                 |  |

Replay device quality *Q* — OB = occupied bandwidth, minF = its lower edge, LNLR = linear-to-non-linear power ratio

| Q<br>OB<br>(kHz)<br>minF<br>(Hz)<br>LNLR                                       | (dB)                      |
|--------------------------------------------------------------------------------|---------------------------|
| perfect<br>0<br>↔<br>high<br>><br>10<br><<br>600<br>low<br><<br>10<br>><br>600 | ↔<br>><br>100<br><<br>100 |

A "low-quality" device is band-limited and nonlinear — exactly the artefacts a CM hunts.

A recorded utterance, silent environment *v*(*n*)=0, in the STFT domain (channel *H* is roughly time-invariant over the utterance, hence no *l*):

$$X(k,l) = S(k,l) H(k).$$

Take logs — convolution becomes addition:

$$\log |X(k,l)| = \log |S(k,l)| + \log |H(k)|.$$

That additivity is the entire trick: the *channel* log *|H*(*k*)*|* sits in a *separate additive term* we can try to isolate.

# Why a vocoder?

Pass *x* through a channel-independent vocoder: it re-synthesizes the speech *content S* while imposing its *own* fixed response *H*vocoder and a residual *H*res, but it does *not* carry the original channel *H*. Comparing "before" and "after" the vocoder therefore *exposes H*.

After the channel-independent vocoder,

$$X_{\text{voc}}(k, l) = S(k, l) H_{\text{voc}}(k) H_{\text{res}}(k),$$

and in logs,

$$\log |X_{\text{voc}}| = \log |S| + \log |H_{\text{voc}}| + \log |H_{\text{res}}|.$$

Frame-level channel estimate (subtract the two logs — *S* cancels):

$$\log H_{vr}(k, l) = \log |H_{voc}(k)| + \log |H_{res}(k)|$$
$$- \log |H(k)|.$$

Utterance-level estimate (average over *L* frames):

$$\log H_{vr} = \frac{1}{L} \sum_{l=1}^{L} \log |X_{voc}(k, l)|$$
$$- \frac{1}{L} \sum_{l=1}^{L} \log |X(k, l)|.$$

# Read-out

*H*voc*, H*res are *fixed* by the vocoder. So log *Hvr* is, up to a constant, ↗ log *|H*(*k*)*|*: a direct estimate of the *trial's channel*. Genuine vs. replayed utterances have di!erent *H* — and now it is measurable.

Frame-level keeps the full *F* ↗ *T* map; utterance-level averages to *F* ↗ 1. Either feeds a *one-class* model trained only on bona-fide.

Model the bona-fide channel-feature density as a mixture of *K* Gaussians:

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad \sum_{k} \pi_k = 1.$$

- *•* Fit *{*ε*<sup>k</sup> , µ<sup>k</sup> ,* !*k}* by EM on bona-fide data only.
- *•* Score a trial by its log-likelihood log *p*(*x*) *low* likelihood ↘ o!-manifold ↘ likely spoof.

This is a generative estimate of *p*(*x |* bona-fide); with a spoof model it becomes the likelihood ratio ! from Part I.

*K* = 2 mixture (solid) over its components (dashed).

#### One-class back-end (2): Variational Auto-Encoder

A VAE learns a latent generative model  $p_{\theta}(x \mid z)p(z)$  and an encoder  $q_{\phi}(z \mid x)$ , trained to maximize the ELBO:

$$\log p_{\theta}(x) \geq \underbrace{\mathbb{E}_{q_{\phi}} \big[\log p_{\theta}(x \mid z)\big]}_{\text{reconstruction}} - \underbrace{\mathrm{KL} \big(q_{\phi}(z \mid x) \parallel p(z)\big)}_{\text{regularizer}}.$$

Sampling uses the reparameterization  $z = \mu_x + \sigma_x \odot \epsilon$ ,  $\epsilon \sim \mathcal{N}(0, I)$ . Trained on bona-fide only, a spoof reconstructs poorly.

#### Reconstruction probability

Draw L latents, average:

$$\frac{\mathbf{1}}{L} \sum_{l} p_{\theta} \! \left( \! \mathbf{x} \mid \boldsymbol{\mu}_{\hat{\mathbf{X}}}^{(l)}, \boldsymbol{\sigma}_{\hat{\mathbf{X}}}^{(l)} \right) < \alpha \; \Rightarrow \; \text{anomaly (spoof)}.$$

#### Part II takeaways

#### Replay, end to end

- 1. Replay = extra acoustic channel: *x* = *s* ↔ *h* + *v*, two rooms, two transducers.
- 2. Room IR = direct + early + late reflections; summarized by *T*<sup>60</sup> (Sabine).
- 3. Logs turn the convolutive channel *additive* the lever for estimation.
- 4. A channel-independent vocoder exposes the trial channel log *Hvr* .
- 5. Score it with a *one-class* model (GMM likelihood, VAE reconstruction prob).

Next: the other attack family — fully synthetic speech — which hides in di!erent places.

Part 3

<span id="page-21-0"></span>[Synthesized Speech and Fusion](#page-21-0)

# Synthesized speech: TTS, voice conversion, or both

Synthesized speech is generated by *Text-To-Speech* (text ≃ waveform) or *Voice Conversion* (one speaker ≃ another), or a pipeline of both.

Crucially — unlike replay, whose fingerprint is a *whole-band* channel — synthesis artefacts are localized in sub-bands:

- *•* Vocoders and neural decoders mis-model specific frequency regions (often very high or very low bands, phase, harmonic fine structure).
- *•* Di!erent front-ends "see" di!erent bands, so the *optimal* band is attack-dependent.

Band-limited EER vs. (*f* min*, f*max): full/high/low/band-pass.

Practical consequence: *learn* the band (sinc-layer, Part IV) instead of fixing it.

# Fusion / ensembles: combining complementary detectors

No single front-end covers every attack. Fuse several, at one of two levels.

Score-level (late fusion):

$$s_{\mathsf{fused}} = \sum_{m=1}^{M} w_m \, s_m,$$

with weights *w<sup>m</sup>* from logistic-regression calibration (e.g. BOSARIS).

Embedding-level (early fusion):

$$e = \left[ \ e_1; \ e_2; \ldots; e_M \ \right] \ \to \ \mathsf{back}\text{-end classifier}.$$

Late fusion combines *scores*; early fusion concatenates *embeddings* and learns a joint back-end.

Part 4

<span id="page-24-0"></span>[Countermeasure Architectures](#page-24-0)

# LightCNN: competitive activations via Max-Feature-Map

A classic CNN whose nonlinearity is Max-Feature-Map (MFM) — a learnable *feature selector* rather than a fixed gate. Given 2*N* channels split in half,

$$MFM 2/1 : h(x) = max(x^1, x^2)$$
 (element-wise),

reducing 2*N* ≃*N*. The 3*/*2 variant from 3*N* channels keeps

$$h^1 = \max(x^1, x^2, x^3), \qquad h^2 = \mathrm{median}(x^1, x^2, x^3).$$

Optional *Network-in-Network* (1 ⇐ 1 linear) layers add capacity cheaply.

Compatible front-ends: LFCC, MFCC, CQCC, STFT,

#### Why MFM helps spoof detection

ReLU zeroes negatives; MFM instead *selects* the more informative of two learned maps. The result is sparse, gradient-friendly, and tends to isolate the *discriminative sub-band* energy that separates genuine from synthetic.

# RawNet2: end-to-end from the raw waveform

- *•* Sinc layer: interpretable, parametric band-pass front-end (next slide).
- *•* ResBlock + FMS: residual conv blocks with feature-map scaling.
- *•* GRU + FC: temporal aggregation → utterance score.

RawNet2 throws away hand-crafted cepstra entirely — the network *learns* its filterbank. The sinc layer is what makes that learnable front-end physically meaningful, and AASIST (later) keeps exactly this encoder.

# The Sinc-layer: a learnable, interpretable band-pass front-end

A conventional 1-D conv learns all *L* taps of each filter:

$$y[n] = x[n] * h[n] = \sum_{l=0}^{L-1} x[l] h[n-l].$$

A sinc filter instead fixes the *shape* to an ideal band-pass and learns only its two cut-o!s (*f*min*, f*max):

$$g[n, f_{\min}, f_{\max}] = 2f_{\max} \operatorname{sinc}(2\pi f_{\max} n) - 2f_{\min} \operatorname{sinc}(2\pi f_{\min} n),$$

smoothed by a Hamming window, *g*[*n*] ⇒ *g*[*n*] *· w*[*n*], to tame truncation ripple.

## Why this is a band-pass

An ideal rectangular pass-band in frequency inverse-transforms to a sinc in time. A band [*f*min*, f*max] is the *di*!*erence* of two low-pass rectangles ↘ the di!erence of two sincs above.

A single sinc kernel *g*[*n*] (windowed).

Three learned sinc filters: sharp, rectangular, interpretable pass-bands.

#### Advantages over a free conv filter

- *•* 2 parameters per filter (*f*min*, f*max) instead of *L* taps far fewer weights, less overfitting.
- *•* Physical interpretation: each filter *is* a named frequency band; learned bands cluster around pitch and formants.
- *•* Faster convergence and more meaningful features than an unconstrained CNN front-end.

#### Feature-Map Scaling (FMS) $\equiv$ Squeeze-and-Excitation

Re-weight channels by a learned, input-dependent gate. Squeeze each filter map  $c_f \in \mathbb{R}^T$  to a scalar by global average pooling, then excite:

$$s_f = \sigma(W[\operatorname{GAP}(c_1), \ldots, \operatorname{GAP}(c_F)]) \in (0, 1).$$

Apply the gate, with any of the published variants:

$$c_f' \in \left\{ \ c_f + s_f, \quad c_f \cdot s_f, \quad \left(c_f + s_f\right) s_f, \quad c_f s_f + s_f \ \right\}.$$

Squeeze (pool)  $\to$  excite (FC+ $\sigma$ )  $\to$  scale. The gate tells the net which frequency channels matter for this utterance.

# Why graphs? A 60-second GNN primer

Spectro-temporal features have no canonical ordering — but they *do* have relationships. A graph neural network respects exactly that:

- *•* Permutation invariance: the order of input nodes does not matter.
- *•* A node's update depends only on its *neighbours* (and itself), then neighbours-of-neighbours as depth grows.
- *•* Depth = the longest neighbour path the model integrates.

Every layer is *message* (compute messages on edges) then *aggregate* (pool them permutation-invariantly).

Target node A aggregates messages from its neighbours *N*(*A*) = *{B, C, D}*, recursively.

The Graph Convolution Network update. Initialize embeddings with node features, *h*(0) *<sup>v</sup>* = *x<sup>v</sup>* , then for *k* = 0*,..., K* ↓ 1:

$$h_{v}^{(k+1)} = \sigma \left( W_{k} \sum_{u \in \mathcal{N}(v)} \frac{h_{u}^{(k)}}{|\mathcal{N}(v)|} + B_{k} h_{v}^{(k)} \right),$$

and read out *z<sup>v</sup>* = *h*(*K*) *<sup>v</sup>* .

- *•* The sum is a *permutation-invariant* pooling of neighbours.
- *• Wk* transforms neighbours; *Bk* is a self-loop (keep your own state).
- *•* ϑ is a nonlinearity (e.g. ReLU).

# The limitation GAT fixes

GCN weights every neighbour equally (1*/|N*(*v*)*|*). But for spoof cues, some spectro-temporal neighbours matter far more than others. We want *learned*, content-dependent edge weights — that is attention.

The Graph Attention layer learns how much each neighbour *j* should influence node *i*:

$$\alpha_{ij} = \frac{\exp \left( \operatorname{LeakyReLU} \left( \vec{a}^{\top} \left[ \right. W \vec{h}_{i} \parallel W \vec{h}_{j} \left. \right] \right) \right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left( \operatorname{LeakyReLU} \left( \vec{a}^{\top} \left[ \right. W \vec{h}_{i} \parallel W \vec{h}_{k} \left. \right] \right) \right)},$$

then aggregates with K attention heads:

$$\vec{h}_i' = \sigma \left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \vec{h}_j \right).$$

#### Reading the formula

- $[\cdot || \cdot]$ : concatenate the two transformed endpoints.
- $\vec{a}$ : a learned attention vector scoring each edge.
- softmax over  $\mathcal{N}_i$  normalizes the weights to a distribution.
- Multi-head averaging stabilizes training.

AASIST uses a variant of this on a  $\ensuremath{\mathit{spectro-temporal}}$  graph.

#### A Heterogeneous Stacking Graph Attention Layer.

Attention weight from u to n (note the element-wise product and tanh inside the map):

$$\alpha_{u,n} = \frac{\exp(W_{\mathsf{map}}(h_n \odot h_u))}{\sum_{w \in \mathcal{M}(n) \cup \{n\}} \exp(W_{\mathsf{map}}(h_n \odot h_w))},$$

$$m_n = \sum_{u \in \mathcal{M}(n) \cup \{n\}} \alpha_{u,n} h_u,$$

$$o_n = \text{SeLU}(\text{BN}(W_{\mathsf{att}} m_n + W_{\mathsf{res}} h_n)).$$

#### "Heterogeneous" & "stacking"

- Heterogeneous: different attention parameters for the three edge types — spectral–spectral (S–S), spectral–temporal (S–T), temporal–temporal (T–T).
- Stack node: one extra node that pools both the S and T sub-graphs into a global summary.

A residual connection  $(W_{res}h_n)$  keeps the node's own information.

#### Two graph utilities AASIST needs

Graph BatchNorm. Stack the node-feature matrices of the graphs in a batch, *<sup>G</sup>*1⇑R*<sup>N</sup>*1↗*<sup>d</sup>* , *<sup>G</sup>*2⇑R*<sup>N</sup>*2↗*<sup>d</sup>* , into one

$$G \in \mathbb{R}^{(N_1+N_2)\times d},$$

and apply ordinary 1-D BatchNorm over the (now-pooled) node dimension. Graphs may have di!erent node counts; concatenation makes the statistics well-defined.

Graph (top-*k*) pooling. Score nodes by projection onto a learned vector *p*, keep the top *k*, gate them, and drop the rest:

$$\begin{split} y &= X^{\ell} \, \rho \big/ \|\rho\|, \quad \mathrm{idx} = \mathrm{top}\text{-}k(y), \\ \tilde{X} &= X^{\ell} [\mathrm{idx}, :], \quad \tilde{y} = \sigma \big( y [\mathrm{idx}] \big), \\ X^{\ell+1} &= \tilde{X} \odot \tilde{y}, \quad A^{\ell+1} = A^{\ell} [\mathrm{idx}, \mathrm{idx}]. \end{split}$$

In AASIST *p* is *not* normalized.

# AASIST: the full pipeline

- *•* Encoder *F*: RawNet2 *without* GRU/FC/FMS.
- *•* Build a *spectral* graph *G<sup>s</sup>* = max*<sup>t</sup> |F|* and a *temporal* graph *G<sup>t</sup>* = max*<sup>s</sup> |F|*; combine into *Gst*.

- *•* Two parallel HS-GAL branches share a *stack node*; graph-pool after the first, residual after the second.
- *•* Element-wise max of the two graphs; read out by concatenating stack node + node-wise max/avg of S and T; a linear layer classifies.

<span id="page-36-0"></span>[Self-Supervised Front-Ends](#page-36-0)

Part 5

Pre-train on *unlabeled* audio: a CNN maps the waveform  $\mathcal X$  to latents  $\mathcal Z$ ; a quantizer produces targets  $\mathcal Q$ ; a Transformer over *masked* latents yields context  $\mathcal C$ . The contrastive loss asks the context at a masked step to identify its own quantized target among distractors:

$$\mathcal{L}_{\textit{m}} = -\log \frac{\exp \left( \operatorname{sim}(c_{t}, q_{t}) / \kappa \right)}{\sum_{\tilde{q} \in \mathcal{Q}_{t}} \exp \left( \operatorname{sim}(c_{t}, \tilde{q}) / \kappa \right)},$$

with cosine similarity  $\sin$ , temperature  $\kappa$ , and a diversity term to use the full codebook.

Pre-trained representations transfer across tasks — including spoof detection, even though it never saw a spoof in pre-training.

Keep AASIST's powerful spectro-temporal graph reasoning, but feed it *self-supervised* embeddings instead of raw-waveform features. This swap is one of the largest single jumps in modern anti-spoofing accuracy — the front-end was the bottleneck.

## Two attack families live in di!erent places

#### Project countermeasure embeddings to 2-D (t-SNE). The geometry tells the story:

- *•* Synthesized speech (TTS/VC) splits into *many tight, well-separated clusters* — one per generation algorithm. Each synthesizer leaves a distinct, *learnable* signature.
- *•* Replayed speech scatters *di*!*usely* around the bona-fide cloud — it *is* real speech, only re-channelled, so it overlaps far more.

synthesis: tight clusters

replay: di!use overlap

t-SNE of CM embeddings (synthesized vs. recorded).

Design implication: synthesis detection is closer to *classification*; replay detection is closer to *anomaly detection* — which is exactly why Part II reached for one-class models.

## Wrap-up: one detector, three channels

The attacker always *adds a channel* — a loudspeaker, a vocoder, a neural decoder. The front-end's job is to make that channel measurable; the back-end's job is to test for it. From sinc filters to graph attention to SSL, every advance is a sharper estimate of the same likelihood ratio.

| Questions?                   |                               |                                 |                          |         |  |  |  |  |  |
|------------------------------|-------------------------------|---------------------------------|--------------------------|---------|--|--|--|--|--|
| Next<br>time:<br>adversarial | robustness,<br>generalization | to<br>unseen<br>attacks,<br>and | partial<br>/<br>deepfake | spoofs. |  |  |  |  |  |

## References & further reading

#### Metrics & detection theory

- *•* Kinnunen et al., *t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing CMs and ASV*, Odyssey 2018.
- *•* Brümmer & du Preez, *Application-independent evaluation of speaker detection* (BOSARIS / calibration), Comp. Speech & Lang. 2006.

#### Replay & room acoustics

- *•* ASVspoof 2017/2019 *Physical Access* corpora & evaluation plans.
- *•* Sabine, *Collected Papers on Acoustics*, 1922 (*T*60).
- *•* Schroeder, *New Method of Measuring Reverberation Time*, JASA 1965.

#### Architectures

- *•* Wu et al., *LightCNN / Max-Feature-Map*, IEEE TIFS 2018.
- *•* Ravanelli & Bengio, *SincNet: Speaker Recognition from Raw Waveform*, SLT 2018.
- *•* Jung et al., *RawNet2*, ICASSP 2021; *AASIST*, ICASSP 2022.
- *•* Veli"kovi# et al., *Graph Attention Networks*, ICLR 2018; Kipf & Welling, *GCN*, ICLR 2017.
- *•* Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.

#### Self-supervised front-ends

- *•* Baevski et al., *wav2vec 2.0*, NeurIPS 2020.
- *•* Tak et al., *Automatic Speaker Verification Spoofing & Deepfake Detection using Wav2Vec 2.0 and Data Augmentation*, Odyssey 2022.

#### Backup: the t-DCF, a little more precisely

 $The \ tandem \ DCF \ averages \ the \ cost \ of \ the \ combined \ CM \rightarrow ASV \ cascade \ over \ target, \ non-target, \ and \ spoof \ trials. \ In \ normalized \ form,$ 

$$t-DCF(\tau_{cm}) = \beta P_{miss}^{cm}(\tau_{cm}) + P_{fa}^{cm}(\tau_{cm}),$$

where  $\beta$  collects the prior probabilities and the costs  $C_{miss}$ ,  $C_{fa}$  together with the ASV's own miss / false-alarm / spoof-acceptance rates evaluated at a fixed ASV operating point. Two consequences:

- A CM is only as useful as its interaction with the ASV a CM that rejects spoofs the ASV would have rejected anyway buys little.
- Reporting  $min \text{ t-DCF} = \min_{\tau_{cm}} \text{ t-DCF}$  removes the CM's threshold choice, just as min-DCF does for a standalone detector.

This is the standard primary metric in the ASVspoof challenge series; EER is the secondary, threshold-free sanity check.

#### Backup: why the ideal band-pass is a di!erence of sincs

Let the desired frequency response be the indicator of the band [*f*min*, f*max] (and its mirror image). Write it as the di\$erence of two low-pass rectangles:

$$G(f) = \mathbb{1}_{[-f_{\mathsf{max}}, f_{\mathsf{max}}]}(f) - \mathbb{1}_{[-f_{\mathsf{min}}, f_{\mathsf{min}}]}(f).$$

The inverse Fourier transform of a low-pass rectangle of cut-o\$ *fc* is

$$\mathcal{F}^{-1}\big[\mathbb{1}_{\left[-f_c,f_c\right]}\big](n) = 2f_c\operatorname{sinc}(2\pi f_c n), \qquad \operatorname{sinc}(x) = \frac{\sin x}{x}.$$

By linearity,

$$g[n] = 2f_{\text{max}}\operatorname{sinc}(2\pi f_{\text{max}}n) - 2f_{\text{min}}\operatorname{sinc}(2\pi f_{\text{min}}n),$$

exactly the sinc-layer kernel. Truncating *n* to a finite length introduces Gibbs ripple; the *Hamming window w*[*n*] multiplies *g*[*n*] to suppress it — trading a little band-edge sharpness for much lower side-lobes. Only (*f*min*, f*max) are learned.

#### Backup: the VAE ELBO in one derivation

Start from the marginal log-likelihood and insert the encoder  $q_{\phi}(z \mid x)$ :

$$\log p_{\theta}(x) = \log \int p_{\theta}(x \mid z) p(z) dz = \log \mathbb{E}_{q_{\phi}} \left[ \frac{p_{\theta}(x \mid z) p(z)}{q_{\phi}(z \mid x)} \right].$$

Jensen's inequality ( $\log \mathbb{E} \geq \mathbb{E} \log$ ) gives the evidence lower bound:

$$\log p_{\theta}(x) \ \geq \ \mathbb{E}_{q_{\phi}}\big[\log p_{\theta}(x\mid z)\big] - \mathrm{KL}\big(q_{\phi}(z\mid x) \parallel p(z)\big) \ \equiv \ \mathcal{L}_{\mathsf{ELBO}}.$$

The gap is exactly  $\mathrm{KL}(q_{\phi}(z\mid x)\|p_{\theta}(z\mid x))\geq 0$ , so maximizing the ELBO simultaneously fits the data and tightens the posterior approximation. At test time, a bona-fide-trained VAE assigns high reconstruction probability to genuine speech and low to spoofs — the anomaly score of Part II.

Reparameterization  $z = \mu_X + \sigma_X \odot \epsilon$  makes the expectation differentiable w.r.t.  $\phi$ , which is what lets us train the encoder by backprop.

#### Backup: a CM design checklist

#### Front-end

- *•* Raw waveform vs. cepstra vs. SSL embeddings.
- *•* If the attack is sub-band, *learn* the band (sinc) rather than fix it.
- *•* Augment with codecs, devices, and rooms you expect at test time.

#### Back-end

- *•* Synthesis → discriminative / graph attention (classification-like).
- *•* Replay / unseen → one-class / anomaly (GMM, VAE, OC-softmax).

#### Evaluation

- *•* Report EER *and* min t-DCF; never tune ω on the test set.
- *•* Test *cross-dataset* unseen attacks are the real bar.
- *•* Inspect the ROC, not just the EER scalar; check the low-FAR regime that deployment actually uses.

#### Fusion

*•* Combine complementary front-ends; calibrate scores before fusing.