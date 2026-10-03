# BlueCodec: A Neural Speech Autoencoder for Multilingual TTS

## Technical Report — Stage 1: Speech Autoencoder

---

## 1. Overview

BlueCodec is a neural speech autoencoder designed to serve as the acoustic backbone of a latent-diffusion-based TTS system. Following the general design principle of SupertonicTTS [1], training is divided into three stages: (1) a speech autoencoder that compresses audio into a compact continuous latent space and reconstructs it, (2) a text-to-latent diffusion module, and (3) a duration predictor. This document covers Stage 1 in full. Sections 10–12 (Sept 2026) add an encoder trained against the frozen official Supertonic-3 vocoder, and the end-of-clip latent spike found in it.

The autoencoder is trained within a Generative Adversarial Network (GAN) framework, combining reconstruction, adversarial, and feature matching objectives. The resulting model functions as a neural vocoder with a low-dimensional latent bottleneck, operating at 44.1 kHz with a 24-dimensional latent space at approximately 86 Hz.

---

## 2. Input Representation

A key architectural departure from SupertonicTTS is the input feature representation fed to the encoder. Rather than using standard log-mel spectrograms alone, we use a **dual-branch linear-mel representation** implemented in `LinearMelSpectrogram`:

1. **Log-linear spectrogram** — computed via STFT (n_fft=2048, hop=512), then log-compressed: 1025 frequency bins.
2. **Log-mel spectrogram** — the linear spectrogram projected through a Mel filterbank (228 bands), then log-compressed.

These two representations are concatenated along the frequency axis, yielding a **1253-channel** input feature map at 86 Hz frame rate. This combined representation preserves both fine-grained spectral detail (from the linear bins) and perceptually weighted structure (from the mel bins), which we found in preliminary experiments to accelerate training loss convergence compared to mel-only inputs.

The spectrogram is computed with `torch.no_grad()` at each training step to avoid storing intermediate activations.

---

## 3. Architecture

### 3.1 Encoder (~25.6M parameters)

The encoder is based on the **Vocos** ConvNeXt backbone [2], modified for latent encoding:

```
Input: [B, 1253, T/512]  (dual log-linear-mel feature)
  │
  ├── Conv1d stem: 1253 → 512  (kernel=7, BatchNorm)
  │
  ├── ConvNeXt Block × 10  (dilation=1 for all blocks)
  │     Each block:
  │       DW-Conv1d (groups=C, kernel=7) → LayerNorm
  │       → PW-Conv 512→2048 → GELU → PW-Conv 2048→512
  │       → γ-scale (learnable, init=1e-6) → residual add
  │
  └── Conv1d proj: 512 → 24  (1×1) + LayerNorm
Output: [B, 24, T/512]  (~86 Hz latent)
```

The encoder uses **non-causal** (standard bilateral padding) convolutions throughout. The original Fourier head of Vocos is replaced by a 1×1 linear projection to the 24-dimensional latent space, followed by channel-wise LayerNorm. The encoder is not used at TTS inference time; its efficient architecture enables fast latent encoding during Stage 2 training.

### 3.2 Decoder (~25.3M parameters)

The decoder mirrors the encoder structure but introduces **causal convolutions** to enable streaming inference:

```
Input: [B, 24, T/512]  (latent z)
  │
  ├── CausalConv1d stem: 24 → 512  (kernel=7)
  │
  ├── Causal ConvNeXt Block × 10
  │     Dilations: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
  │     Same structure as encoder, but left-pad only
  │     (padding = (kernel−1) × dilation, left side)
  │
  ├── BatchNorm1d (512)
  │
  └── VocoderHead:
        CausalConv1d 512 → 2048 (kernel=3)
        → PReLU
        → Conv1d 2048 → 512 (1×1)
        → transpose [B,512,T] → [B,T,512]
        → reshape [B, T×512]  (sub-pixel expansion)
Output: [B, T_audio]  (44.1 kHz waveform)
```

The dilated causal ConvNeXt blocks (pattern `[1,2,4,1,2,4,1,1,1,1]`) provide an effective receptive field while strictly preserving causality. The VocoderHead is inspired by WaveNeXt [3], using a sub-pixel flattening strategy to expand frame-level features directly into the time-domain waveform, without transposed convolutions. Higher hidden dimensionality (2048) and PReLU nonlinearity improve representational capacity over the original design.

The decoder stores learned `latent_mean` and `latent_std` buffers for normalization at inference time, populated from statistics computed over the training set after Stage 1.

### 3.3 Causal vs. Non-Causal Convolution

```
Non-causal (Encoder):
  padding = (kernel−1) × dilation // 2   (symmetric, both sides)

Causal (Decoder):
  padding = (kernel−1) × dilation        (left side only)
  → no future context, suitable for real-time streaming
```

---

## 4. Discriminators

Two discriminators operate in parallel to provide perceptual adversarial signal.

### 4.1 Multi-Period Discriminator (MPD)

Five `DiscriminatorP` sub-networks, one per period `p ∈ {2, 3, 5, 7, 11}`. Each sub-network reshapes the waveform into a 2D tensor `[B, 1, T/p, p]` and applies a stack of Conv2d layers (channels: 1→16→64→256→512→512→1, stride=3 except last). Weight normalization is applied. Feature maps from all intermediate layers are collected for the feature matching loss.

### 4.2 Multi-Resolution Discriminator (MRD)

Three `DiscriminatorR` sub-networks at STFT resolutions `(n_fft, hop, win) ∈ {(512,128,512), (1024,256,1024), (2048,512,2048)}`. Each sub-network computes the log-magnitude spectrogram and processes it with a 2D Conv stack (channels: 1→16→16→16→16→16→1). **Spectral normalization** (rather than weight normalization) is applied to all MRD layers, providing additional training stability.

---

## 5. Loss Functions

### 5.1 Generator Loss

$$\mathcal{L}_G = 45 \cdot \mathcal{L}_\text{recon} + 1 \cdot \mathcal{L}_\text{adv} + 0.1 \cdot \mathcal{L}_\text{fm}$$

**Reconstruction loss** $\mathcal{L}_\text{recon}$ — multi-resolution mel L1, averaged over three STFT scales:

| Scale  | n_fft | hop  | win  | n_mels |
|--------|-------|------|------|--------|
| Small  | 1024  | 256  | 1024 | 64     |
| Medium | 2048  | 512  | 2048 | 128    |
| Large  | 4096  | 1024 | 4096 | 128    |

$$\mathcal{L}_\text{recon} = \frac{1}{3} \sum_s \left\| \text{Mel}_s(y) - \text{Mel}_s(\hat{y}) \right\|_1$$

Note: the reconstruction mel transforms use `MelSpectrogramNoLog` (linear amplitude, no log), while the input features to the encoder use the log-compressed `LinearMelSpectrogram`. This decouples the training signal from the input representation.

**Adversarial loss** $\mathcal{L}_\text{adv}$ — LSGAN generator objective:

$$\mathcal{L}_\text{adv} = \sum_D \mathbb{E}\left[(1 - D(\hat{y}))^2\right]$$

**Feature matching loss** $\mathcal{L}_\text{fm}$ — L1 distance between intermediate discriminator feature maps:

$$\mathcal{L}_\text{fm} = \frac{1}{N} \sum_\ell \mathbb{E}\left[\left\|\text{feat}^{(\ell)}_\text{real} - \text{feat}^{(\ell)}_\text{fake}\right\|_1\right]$$

### 5.2 Discriminator Loss

LSGAN discriminator objective (real → 1, fake → −1):

$$\mathcal{L}_D = \sum_D \mathbb{E}\left[(D(y) - 1)^2 + (D(\hat{y}) + 1)^2\right]$$

---

## 6. Training Configuration

### 6.1 Hyperparameters

| Parameter                   | Value                              |
|-----------------------------|------------------------------------|
| Hardware                    | 2× NVIDIA RTX 3090 (PyTorch DDP, NCCL) |
| Total iterations            | 1,500,000                          |
| Wall-clock time             | ~4 weeks                           |
| Optimizer                   | AdamW (β₁=0.8, β₂=0.99, wd=0.01) |
| Learning rate               | 2×10⁻⁴                             |
| LR schedule                 | CosineAnnealingLR → η_min=1×10⁻⁶  |
| Batch size                  | 128 (across 2 GPUs)                |
| Audio sample rate           | 44,100 Hz                          |
| Input hop size              | 512 samples (~11.6 ms)             |
| Latent frame rate           | ~86 Hz                             |
| Latent dimensionality       | 24                                 |
| Audio crop length           | 0.19 s (~8,379 samples @ 44.1 kHz) |
| Grad clip (encoder/decoder) | 5.0                                |
| Grad clip (discriminators)  | 1.0                                |
| Discriminator warmup        | 0 steps (active from step 1)       |
| λ_recon / λ_adv / λ_fm     | 45 / 1 / 0.1                       |

### 6.2 Training Step

Each iteration proceeds as follows:

1. Load a batch of raw audio segments `[B, 1, T]`.
2. Compute the dual-channel input spectrogram `[B, 1253, T/512]` under `torch.no_grad()`.
3. Forward pass: `encoder → z → decoder → ŷ`.
4. Randomly crop both `y` and `ŷ` to a 0.19 s window.
5. **Discriminator step** (detached `ŷ`): compute $\mathcal{L}_D$ from MPD + MRD, backward, clip gradients, `opt_d.step()`.
6. **Generator step** (full `ŷ`): compute $\mathcal{L}_G$ from MPD + MRD, backward, clip gradients, `opt_g.step()`.
7. Step both cosine learning rate schedulers.

---

## 7. Training Data

The autoencoder was trained on a large, multilingual corpus totalling approximately **11,000 hours** of speech across **more than 6 million audio files**, with particular emphasis on broad acoustic diversity and high audio quality at 44.1 kHz. All files were segmented to a maximum duration of 15 seconds prior to training.

### 7.1 English Corpora

| Dataset         | Hours   | Speakers | Notes                                                   |
|-----------------|---------|----------|---------------------------------------------------------|
| **HiFi-TTS v1** | ~292    | 10       | High-SNR (≥32 dB), ≥13 kHz bandwidth, 44.1 kHz [4]    |
| **HiFi-TTS v2** | ~2,000  | —        | Subset of large-scale LibriVox corpus, 44.1 kHz; pre-segmented into chunks ≤15 s (~1M files) [5] |
| **LibriTTS**    | ~585    | 2,456    | Multi-speaker audiobooks, 24 kHz (upsampled) [6]       |
| **LJSpeech**    | ~24     | 1        | Single-speaker (female), clean studio quality [7]      |
| **VCTK-44k**    | ~44     | 110      | Multi-accent English, resampled to 44.1 kHz [8]        |

### 7.2 Multilingual Corpora

| Dataset         | Language | Hours   | Speakers | Notes                                                    |
|-----------------|----------|---------|----------|----------------------------------------------------------|
| **MLS German**  | German   | ~1,995  | 176      | Multilingual LibriSpeech, LibriVox audiobooks [9]        |
| **de_DE**       | German   | —       | —        | Additional German TTS data                               |
| **Thorsten-DE** | German   | —       | 1        | Single-speaker German TTS corpus (Thorsten Müller)       |
| **es_ES**       | Spanish  | —       | —        | Spanish TTS data                                         |
| **it_IT**       | Italian  | —       | —        | Italian TTS data                                         |

### 7.3 Hebrew Corpora

All Hebrew audio was resampled to 44.1 kHz mono prior to training.

| Dataset                          | Hours  | Speakers | Notes                                                                           |
|----------------------------------|--------|----------|---------------------------------------------------------------------------------|
| **SententicDataTTS** [10]        | ~2,000 | 10       | Hebrew & English, 5 male + 5 female speakers; generated via Chatterbox & MamreTTS; 351 GB total |
| **RanLevi40h** [11]              | ~40    | 1        | _Osim Historia_ podcast (Ran Levi), natural narrative speech, 44.1 kHz mono    |
| **Knesset-VOX-IPA** [12]        | ~2,000 | Multi    | Israeli Knesset parliamentary sessions; IPA-transcribed; derived from VoxKnesset [13] |

The Hebrew data constitutes a significant portion of the overall corpus and enables the autoencoder to model the phonological characteristics of Modern Hebrew, including its distinctive fricatives, pharyngeals, and vowel patterns. All Hebrew datasets provide IPA phoneme annotations, which are used in Stage 2 (text-to-latent) training.

---

## 8. Key Differences from SupertonicTTS

While BlueCodec draws architectural inspiration from SupertonicTTS [1], several design choices diverge. The table below uses exact architecture details from the SupertonicTTS paper (arXiv:2503.23108, Appendix A.1).

| Aspect                        | SupertonicTTS [1]                                                                   | BlueCodec                                                                              |
|-------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Encoder input**             | 228-dim log-mel spectrogram (FFT=2048, Hann window)                                 | **1253-dim: log-linear (1025) + log-mel (228) concatenated**                           |
| **Encoder stem**              | Conv1d 228→512 (k=7) + BatchNorm                                                    | Conv1d 1253→512 (k=7) + BatchNorm                                                      |
| **Encoder blocks**            | 10× ConvNeXt (dim=512, intermediate=2048, k=7, dilation=1), non-causal             | 10× ConvNeXt (dim=512, intermediate=2048, k=7, dilation=1), non-causal                |
| **Encoder projection**        | Linear 512→24 + LayerNorm                                                           | Conv1d 1×1 512→24 + LayerNorm                                                          |
| **Decoder stem**              | CausalConv1d 24→512 (k=7) + BatchNorm                                               | CausalConv1d 24→512 (k=7), **no BatchNorm in stem**                                    |
| **Decoder blocks**            | 10× dilated CausalConvNeXt (dim=512, intermediate=2048, k=7) + BatchNorm           | 10× dilated CausalConvNeXt (dim=512, intermediate=2048, k=7) + BatchNorm               |
| **Decoder dilations**         | `[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]`                                                   | `[1, 2, 4, 1, 2, 4, 1, 1, 1, 1]`                                                      |
| **Vocoder head**              | CausalConv1d 512→2048 (k=3) → Linear 2048→512 → reshape                            | CausalConv1d 512→2048 (k=3) → **PReLU** → Conv1d 1×1 2048→512 → transpose+reshape     |
| **Latent dimensionality**     | 24                                                                                  | 24                                                                                     |
| **MPD**                       | 5 sub-nets, periods {2,3,5,7,11}, 6 conv layers (16,64,256,512,512,1), weight norm | 5 sub-nets, periods {2,3,5,7,11}, 6 conv layers (16,64,256,512,512,1), weight norm    |
| **MRD**                       | FFT {512,1024,2048}, hop=FFT/4, 6 Conv2d layers (Table 7), weight norm             | FFT {512,1024,2048}, hop=FFT/4, 6 Conv2d layers (Table 7), **spectral norm**          |
| **Recon loss domain**         | Log-mel spectrogram                                                                 | **Linear-amplitude mel (no log)**                                                      |
| **Loss weights**              | λ_recon=45, λ_adv=1, λ_fm=0.1                                                      | λ_recon=45, λ_adv=1, λ_fm=0.1                                                         |
| **Adversarial crop length**   | 0.19 s                                                                              | 0.19 s                                                                                 |
| **Optimizer**                 | AdamW (lr=2×10⁻⁴, batch=128)                                                       | AdamW (lr=2×10⁻⁴, batch=128)                                                          |
| **Total iterations**          | 1,500,000                                                                           | 1,500,000                                                                              |
| **Sample rate**               | 44.1 kHz                                                                            | 44.1 kHz                                                                               |
| **Training hardware**         | 4× NVIDIA RTX 4090                                                                  | **2× NVIDIA RTX 3090**                                                                 |
| **Training data (AE)**        | ~11,167 h, ~14,000 speakers, English + internal                                     | **~11,000 h, 6M+ files, multilingual (EN/DE/ES/IT/HE)**                               |

The three genuine architectural differences are: **(1)** the dual-channel input representation (log-linear + log-mel) vs. mel-only, providing finer spectral detail to the encoder; **(2)** the addition of a PReLU nonlinearity in the vocoder head between the two projection layers; and **(3)** spectral normalization in the MRD instead of weight normalization. Notably, the loss formulation, crop strategy, optimizer, iteration count, and sample rate are identical to SupertonicTTS. The primary practical difference is the multilingual training corpus — BlueCodec extends coverage to German, Spanish, Italian, and Hebrew — trained on consumer-grade 3090 hardware rather than 4090s.

---

## 9. Developer Insights & Architectural Rationale

While BlueCodec shares its foundational DNA with SupertonicTTS [1], the deviations introduced here were not arbitrary. They were chosen to improve training stability, synthesis quality, and phonetic coverage while preserving the efficiency of the original design.

### 9.1 The Dual-Branch Input Advantage

Standard mel spectrograms are effective because they approximate human auditory perception, but that compression inevitably discards part of the high-frequency detail present in the original spectrum. BlueCodec therefore feeds the encoder a 1253-channel dual-branch representation formed by concatenating 1025 log-linear bins with 228 log-mel bins. This gives the model both raw spectral precision and perceptually organized structure at the same time. In practice, this richer representation improved early optimization behavior and accelerated reconstruction-loss convergence relative to mel-only inputs.

### 9.2 PReLU in the Sub-Pixel Vocoder

The decoder must expand a low-rate latent sequence at roughly 86 Hz into a full 44.1 kHz waveform, making the vocoder head a critical capacity bottleneck. Inspired by WaveNeXt, BlueCodec keeps the sub-pixel flattening strategy but inserts a PReLU nonlinearity between the causal projection layers. This is a small modification architecturally, yet an important one functionally: it increases the expressiveness of the head and improves gradient flow when reconstructing fine temporal structure and high-frequency waveform detail.

### 9.3 Taming GAN Instability with Spectral Normalization

Adversarial training at high audio fidelity is often unstable, especially when discriminators become too strong early in training. SupertonicTTS uses weight normalization throughout its discriminators, but in BlueCodec the Multi-Resolution Discriminator was deliberately switched to spectral normalization. By constraining the effective Lipschitz behavior of the MRD layers, this change reduces the risk of discriminator domination and helps maintain a smoother optimization trajectory over the full 1.5 million-step run.

### 9.4 A Richer, Multilingual Latent Space

Many open speech autoencoders are implicitly biased toward English phonetics because their training data is dominated by English corpora. BlueCodec instead learns its 24-dimensional latent space from a broad multilingual mixture spanning approximately 11,000 hours and more than 6 million files. The strong inclusion of Hebrew is especially important: corpora such as [SententicDataTTS](https://huggingface.co/datasets/notmax123/SententicDataTTS), [RanLevi40h](https://huggingface.co/datasets/notmax123/RanLevi40h), and [Knesset-VOX-IPA](https://huggingface.co/datasets/notmax123/Knesset-VOX-IPA) expose the model to phonetic patterns that are rare or absent in English-only training, including emphatic fricatives, uvulars, and pharyngeal-adjacent articulations. This materially improves the inclusivity and robustness of the latent representation.

### 9.5 High-Fidelity on Consumer Hardware

Achieving real-time 44.1 kHz streaming synthesis is often associated with large compute clusters. BlueCodec shows that strong neural audio compression can be trained on consumer hardware. Efficient ConvNeXt blocks (depthwise-separable convolutions instead of heavy recurrence), a compact 24-channel latent bottleneck, and the sub-pixel expansion head keep the autoencoder footprint modest (on the order of ~51M parameters for encoder + decoder) while targeting full-band reconstruction.

SupertonicTTS reports Stage 1 training on four RTX 4090 GPUs [1]. Using the same iteration budget and batch size on **two RTX 3090** GPUs with PyTorch DDP, this build completed in roughly **four weeks** — a practical datapoint that architectural efficiency and a well-matched training recipe matter as much as raw accelerator count for this class of model.

---

## 10. Encoder Trained Against the Frozen Supertonic-3 Vocoder (Sept 2026)

### 10.1 Idea

Supertone's official Supertonic-3 release ships its vocoder (`onnx/vocoder.onnx` on [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3), OpenRAIL-M). Its decoder has exactly our `LatentDecoder1D` layout — CausalConv1d stem, 10 dilated causal ConvNeXt blocks, BatchNorm, PReLU sub-pixel head — so its 103 ONNX initializers map 1:1 onto our 101 decoder tensors (`tts.ae.decoder.*` by name, three anonymous initializers to `embed.net.weight`/`embed.net.bias`/`head.act.weight`; the graph's `latent_mean`, `latent_std` and `normalizer.scale = 0.25` are kept aside). The one difference is padding: every Pad node in the graph is `mode='edge'`, so the port uses replicate causal padding (`pad_mode="replicate"`). On a 4 s clip the port matches onnxruntime to a max abs difference of 7.2e-6 (waveform peak 0.68); the same weights with zero padding differ by 0.46.

We keep that decoder **frozen** and train **only our encoder** through it: the encoder learns to emit the official latent space, the decoder is never updated. The weights are not redistributed; `load_supertonic3_decoder` in `bluecodec/autoencoder/latent_decoder.py` downloads them from the official repo at load time (pinned revision `3cadd1ee`, vocoder md5 `68e5b768810cb3c2cf7a27f3ce2494e3`).

### 10.2 Recipe

| Parameter | Value |
|-----------|-------|
| Encoder init | 1.5M-step encoder (identical to the one in `model.safetensors`) |
| Decoder | official Supertonic-3 vocoder, frozen (`requires_grad=False`, `eval()`), asserted unchanged at every save |
| Trained | encoder + MPD + MRD |
| Optimizer / LR | AdamW (β₁=0.8, β₂=0.99, wd=0.01), 8.5×10⁻⁵ cosine → 1×10⁻⁶ over 300k steps |
| Batch | 2× RTX 5090, 64 segments × 61,740 samples per GPU |
| Losses | λ_recon=45 on **full-band log mel over the whole segment**; λ_adv=1, λ_fm=0.1 averaged over MPD+MRD layers together (paper Eq. 6); adversarial terms on the 0.19 s crop |
| D warm-up | first 10k steps reconstruction-only |
| Wall clock | 2026-09-22 15:22 → 2026-09-23 05:33 |
| Released | `ae_290000.pt` (the loop exits at 300k before that save) |

Command line and flag reference: [docs/training.md §4](../docs/training.md).

### 10.3 Round-trip audit

Round trip on 10 reference clips from BlueTTS's voice-cloning bench (`libri_*`, LibriTTS recordings at 24 kHz resampled to 44.1 kHz, so the source's 12–22 kHz band is nearly empty and a positive delta there is energy the decoder adds). Bare encoding; the edge-padded numbers are in §11.

| Codec | front-end L1 | front-end L1 >8k | mel228 L1 | mel228 L1 >8 kHz | 12–22 kHz vs source | DNSMOS |
|-------|------:|------:|------:|------:|------:|------:|
| BlueCodec 1.5M (`model.safetensors`, zero padding) | 0.4426 | 0.3757 | 1.0094 | 2.2624 | +19.56 dB | 3.40 |
| 1.5M encoder + decoder-only retrain (full-band log-mel loss, replicate padding; not published) | 0.3604 | 0.2341 | **0.8495** | **1.7274** | **+1.50 dB** | 3.41 |
| **Encoder trained against the frozen Supertonic-3 vocoder** | **0.3385** | **0.2251** | 0.8944 | 1.9642 | +7.63 dB | **3.42** |

*front-end L1* is the audit metric BlueTTS has used historically as its "log-mel L1": the L1 of `log(clamp(F, 1e-5))` where `F` is the encoder front end (1025 log-linear + 228 log-mel channels); *front-end L1 >8k* is the same over channels `[int(1253·8000/22050):]`. *mel228 L1* is the plain L1 between 228-band log mels (clamp 1e-5), *>8 kHz* over the bands centred above 8 kHz. The 12–22 kHz column is the reconstruction's mean STFT power in 12–22.05 kHz minus the source's; DNSMOS is the overall MOS of the reconstruction resampled to 16 kHz.

Reading: the Supertonic-3-decoder encoder is best on the historical metric and DNSMOS, but the DNSMOS spread (3.40–3.42) is within noise for 10 clips, the retrained decoder is better on the plain mel L1, and the official decoder adds ~6 dB more top-octave energy than the retrained decoder on band-limited sources. Both are large improvements over the published 1.5M decoder (+19.56 dB). This is a 10-clip round-trip audit, not a listening test.

### 10.4 Downstream caveat

This encoder moves the latent space: any stats file, text-to-latent or duration checkpoint and exported voice built on the 1.5M encoder is invalid with it. In BlueTTS a 40k-step text-to-latent finetune onto this autoencoder (Sept 2026) was rejected — cloning similarity on the 10-voice bench dropped by ~45% — so an AE swap is treated as a from-scratch or long-restart event for the downstream models, never a short finetune.

---

## 11. End-of-Clip Edge Code (Sept 2026)

### 11.1 Finding

With bare encoding, the Supertonic-3-decoder encoder writes an **edge code** into the last ~3 raw latent frames of every clip that ends right after audio. Measured in the normalised compressed domain (`((z − mean)/std)·0.25` with that encoder's latent stats), over 50 clips (10 bench references + 40 training clips cut to native / 3072k / 3072k+512 / 3072k+1700 samples):

- the last compressed frame is **8.9×** the clip's median frame norm on average (p50 9.2, max 14.1); the 1.5M encoder: 1.46×;
- the excess sits on **raw channel 6** (~30σ of the latent stats); raw frames −3/−2/−1 are 2.6× / 6.1× / 10.8× the median, and `compress_latents` replicates the last raw frame into the remainder, so the whole last compressed frame carries it.

A second measurement on the 10 bench references (native length): last frame **9.44×** mean (p50 9.76, max 11.99), top raw channel 6 in 10/10 clips, raw frames −3/−2/−1 at 2.62× / 6.73× / 12.08×. The magnitude depends on the material: on 50 clips drawn from the Italian/Spanish/`test` training corpora (which end in near-silence) it is 3.38× mean (p50 2.38, max 14.59), mostly on raw channel 2.

### 11.2 Cause

The encoder is non-causal with zero convolution padding, so its last frames see the array edge. It was trained against a frozen *causal* decoder on fixed 61,740-sample segments: 61,740 = 120 × 512 + 300, so every training segment ended with the same partial frame (300/512 samples supervised) and the array edge always in the same place. The encoder learned to use that edge. Within the clip, the official decoder renders the code harmlessly (round-trip error over the tail −56.9 vs −56.4 dB), but rendered *as content* — by a flow model that copied it, or through the replicate pad — it is a burst ~38 dB above the silence it replaces (mean −48 vs −86 dBFS RMS).

### 11.3 Fix: edge-padded encoding

`bluecodec.utils.encode_wav_edge_padded` (and `BlueCodec.encode(audio, edge_pad_chunks=2)`) uses the official helper's layout — pad to a multiple of 3072 samples, keep `ceil(L/3072)` compressed frames, never keep the STFT centre frame past the end — plus **2 extra chunks of silence**, so no kept frame sees the edge.

| Encoding | last frame / median, 50 mixed clips | 10 refs | 50 it/es/test clips | last frame vs edge-free (mean / max) |
|---|---:|---:|---:|---:|
| bare (classic) | 8.9× | 9.44× | 3.38× | 625% / 737% (10 refs); 272% / 897% (50 clips) |
| official layout only (pad to 3072) | — | 1.84× | 1.53× | 99% / 363% (50 mixed); 98% / 229% (10 refs); 83% / 254% (50 clips) |
| **edge-padded, 2 chunks** | **1.35×** | **1.44×** | **1.13×** | **2.3% / 4.2%** (50 mixed); 1.9% / 2.4% (10 refs); 4.0% / 4.2% (50 clips) |
| edge-free reference (1 s silence appended) | 1.37× | 1.45× | 1.14× | 0 |

Reconstruction is unchanged by the padding (§10.3 clips, bare → edge-padded): front-end L1 0.3385 → 0.3387, 12–22 kHz +7.63 → +7.63 dB, DNSMOS 3.422 → 3.423; for the 1.5M encoder with the retrained decoder 0.3603 → 0.3604.

### 11.4 Downstream effect (BlueTTS, Sept 2026)

A text-to-latent model trained on this encoder's targets, encoded as a batch, learns the edge code from each batch's longest row and ends every B=1 render with it (last 4 generated frames 7–10× the body norm, on channels 6/19/2/23; decoded quietly, about −49 dBFS). On BlueTTS's Hebrew intelligibility test with the same model, encoding the **reference** clip bare versus edge-padded moved PER 0.076 → 0.018 and WER 0.168 → 0.059; on the 10-voice bench with references cut to 3.0/4.3 s, Hebrew PER 0.073 → 0.023. BlueTTS's evaluation now encodes references edge-padded, and its inference script and text-to-latent trainer have opt-ins to do the same. No encoder retrain was needed for this fix, and the existing latent stats and duration predictor stay valid.

---

## 12. Edge-Fixed Encoder (Sept 2026)

The edge-fixed encoder (`encoder_supertonic3_decoder_edge_fixed/encoder.safetensors` on the Hub) removes the edge code at the source instead of by padding.

### 12.1 Recipe

The §10 encoder (`ae_290000.pt`) continued against the same frozen official vocoder, with encoder, discriminators and AdamW moments carried over:

| Steps | lr | Additions |
|-------|----|-----------|
| 290k → 300k | 2×10⁻⁵, 1k warm-up, cosine | **edge-aware batches**: variable-length segments, half ending at the clip's true end; 85% of batches cut so every row ends within one 3072-sample chunk of a shared, 3072-aligned array end (a quarter exactly at it, a quarter within 512 samples); loss masked per clip at `ceil(L/3072)·3072`; latents in the official layout |
| 300k → 310k | same | + **tail consistency** (weight 15, last 3 raw frames); encoder BatchNorm frozen |
| 310k → 350k | 2×10⁻⁵, 1k warm-up, cosine over 40k | tail consistency as the relative L2 of each clip's **last compressed frame** (all 6 sub-frames, normalised with the 290k encoder's latent stats) against an encoding of the same clip with 2 extra chunks of silence (no gradient), weight 7.5 |

The tail term is needed because the frozen decoder renders the edge code as faithfully as a true latent, so reconstruction and adversarial losses give almost no gradient against it. These terms were run with BlueTTS's trainer and are not flags of this repository's `train_autoencoder.py`.

### 12.2 Results

Round trip on the §10.3 clips, official layout (`encode(audio, edge_pad_chunks=0)`; bare and `edge_pad_chunks=2` give the same numbers to ±0.001):

| Codec | front-end L1 | front-end L1 >8k | mel228 L1 | 12–22 kHz vs source | DNSMOS |
|-------|------:|------:|------:|------:|------:|
| Supertonic-3-decoder encoder (290k, §10.3, bare) | 0.3385 | 0.2251 | 0.8944 | +7.63 dB | 3.42 |
| **Edge-fixed encoder (350k)** | 0.3387 | **0.2233** | **0.8839** | **+7.11 dB** | **3.44** |

End of clip, measured as in §11.3 on the same 50 mixed clips (each encoder normalised with its own latent stats):

| Encoding | 290k: last / median | 290k: last frame vs edge-free (mean / max) | edge-fixed: last / median | edge-fixed: vs edge-free (mean / max) |
|---|---:|---:|---:|---:|
| official layout (pad to 3072) | 1.76× | 99% / 363% | **1.36×** | **5.3% / 33%** (40/50 within 10%) |
| edge-padded, 2 chunks | 1.36× | 2.3% / 4.2% | 1.37× | 0.5% / 0.9% |
| edge-free reference | 1.38× | 0 | 1.37× | 0 |

On the 10 references alone the official layout gives 1.44× (edge-free 1.45×), 8/10 within 10%. With the official layout the end-of-clip spike is gone; the clips still outside 10% are ones whose audio runs to the very end of the array. Edge-padded encoding remains the closest to edge-free and costs nothing, so it is still recommended for latents that feed another model.

### 12.3 Downstream caveat

The edge-fixed encoder's latents are in the Supertonic-3 latent space but are not the 290k encoder's: stats files, text-to-latent and duration checkpoints and exported voices built on one are invalid with the other (§10.4 applies).

---

## References

[1] SupertonicTTS (2025). *SupertonicTTS: A Lightweight and Flexible Text-to-Speech System with Latent Diffusion.* arXiv:2503.23108.

[2] Siuzdak, H. (2023). *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis.* arXiv:2306.00814.

[3] WaveNeXt — sub-pixel waveform generation from frame-level features.

[4] Bakhturina, E., Lavrukhin, V., Ginsburg, B., & Zhang, Y. (2021). *Hi-Fi Multi-Speaker English TTS Dataset.* arXiv:2104.01497. [OpenSLR-109](http://www.openslr.org/109/)

[5] Langman, R., et al. (2025). *HiFiTTS-2: A Large-Scale High Bandwidth Speech Dataset.* Proc. Interspeech 2025. [HuggingFace](https://huggingface.co/datasets/nvidia/hifitts-2)

[6] Zen, H., et al. (2019). *LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech.* arXiv:1904.02882.

[7] Ito, K., & Johnson, L. (2017). *The LJ Speech Dataset.* https://keithito.com/LJ-Speech-Dataset/

[8] Veaux, C., Yamagishi, J., & MacDonald, K. (2017). *CSTR VCTK Corpus.* University of Edinburgh.

[9] Pratap, V., et al. (2020). *MLS: A Large-Scale Multilingual Dataset for Speech Research.* arXiv:2012.03411.

[10] notmax123 (2025). *SententicDataTTS.* HuggingFace. https://huggingface.co/datasets/notmax123/SententicDataTTS

[11] notmax123 (2025). *RanLevi40h — Osim Historia Hebrew Audio Dataset.* HuggingFace. https://huggingface.co/datasets/notmax123/RanLevi40h

[12] notmax123 (2025). *Knesset VOX IPA.* HuggingFace. https://huggingface.co/datasets/notmax123/Knesset-VOX-IPA

[13] Ben-David, E., et al. (2025). *VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging Speaker Modeling.* arXiv:2603.01270.

[14] Supertone Inc. (2026). *Supertonic 3.* Hugging Face: https://huggingface.co/Supertone/supertonic-3 (model under BigScience OpenRAIL-M; sample code: https://github.com/supertone-inc/supertonic, MIT).
