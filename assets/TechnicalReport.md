# BlueCodec: A Neural Speech Autoencoder for Multilingual TTS

## Technical Report — Stage 1: Speech Autoencoder

---

## 1. Overview

BlueCodec is a neural speech autoencoder designed to serve as the acoustic backbone of a latent-diffusion-based TTS system. Following the general design principle of SupertonicTTS [1], training is divided into three stages: (1) a speech autoencoder that compresses audio into a compact continuous latent space and reconstructs it, (2) a text-to-latent diffusion module, and (3) a duration predictor. This document covers Stage 1 in full.

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

## 9. Next Steps

After Stage 1 completes, latent statistics must be computed before Stage 2:

```bash
uv run compute_latent_stats.py --tts-json config/tts.json
# → outputs: stats_real_data.pt
```

This computes per-channel mean and standard deviation of the latent space over the training set, which are loaded into the decoder's `latent_mean` and `latent_std` buffers for normalization at inference time.

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
