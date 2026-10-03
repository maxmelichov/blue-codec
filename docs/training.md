# Training Documentation

## 🗂️ 1. Data Preparation

The Autoencoder is designed to be trained on large-scale, high-quality audio datasets. Since it only requires audio (no transcripts), you can use any collection of speech or general audio.

### Recommended Datasets

For high-quality speech reconstruction, we recommend using a mix of the following datasets:

- **LibriTTS / LibriTTS-R:** Large-scale corpus of English speech (approx. 585 hours). The "-R" version is restored for higher quality.
- **LJSpeech:** A standard single-speaker English dataset (approx. 24 hours).
- **VCTK:** Multi-speaker English dataset with various accents (approx. 44 hours).
- **Hi-Fi TTS:** High-fidelity multi-speaker English dataset (approx. 291 hours).
- **Common Voice:** Massive multi-language dataset (thousands of hours).

### 2. Configure Training

Update `config/tts.json` to point to your dataset directories. You can provide a single path or a list of paths. The system will recursively scan these directories for audio files (`.wav`, `.flac`, `.mp3`, etc.).

```json
    "ae": {
        "data": {
            "train_metadata": [
                "/path/to/LibriTTS",
                "/path/to/hifi-tts",
                "/path/to/custom_audio"
            ],
            "val_metadata": "/path/to/validation_audio"
        }
    }
```

**Custom Data:**
You can simply provide the path to any directory containing your audio files. The training script handles the recursive discovery of supported audio formats.


## 🚀 3. Training the Autoencoder (AE)

The Autoencoder learns to compress audio into a low-dimensional latent space.

```bash
uv run train_autoencoder.py
```

- **Output:** Checkpoints are saved to `checkpoints/ae/`.
- **Options:**
  - `--resume path/to/ckpt.pt`: Resume training from a specific checkpoint.
  - `--eval_input path/to/audio.wav`: Run reconstruction evaluation on a specific file during training.
  - **Distributed Training:** For faster training on multiple GPUs, use `torchrun` via `uv run`:
    ```bash
    uv run torchrun --nproc_per_node=2 train_autoencoder.py --resume checkpoints/ae/ae_latest.pt
    ```

---

## 🎯 4. Encoder-Only Training Against a Frozen Decoder (Sept 2026)

This mode trains **only the encoder** (plus the discriminators) against a **frozen decoder**, e.g. the official vocoder (a frozen pretrained vocoder; see the README's *References and acknowledgements*), loaded 1:1 into `LatentDecoder1D` with replicate ("edge") causal padding. The decoder stays in the graph, so the reconstruction and adversarial gradients flow *through* it into the encoder, which learns to emit the latent space that decoder expects.

The official decoder weights are **not** part of this repository. `--decoder supertonic3` downloads `onnx/vocoder.onnx` from its official Hugging Face repo (OpenRAIL-M) at start-up; reading it needs `pip install onnx`. The command that reproduces the released official-vocoder encoder:

```bash
uv pip install onnx
uv run torchrun --nproc_per_node=2 train_autoencoder.py \
    --encoder_only --decoder supertonic3 \
    --init_encoder path/to/model.safetensors \
    --lr 8.5e-5 --batch_size 64 --total_steps 300000 --d_warmup 10000 \
    --recon_logmel_fullband --fm_composite --lambda_recon 45 \
    --checkpoint_dir checkpoints/ae_official_vocoder
```

| Flag | Meaning |
|------|---------|
| `--encoder_only` | Decoder frozen (`requires_grad=False`, kept in `eval()` so its BatchNorm statistics never move, not wrapped in DDP, not in any optimizer). `opt_g` holds the encoder only. The decoder is asserted bit-identical at every save. |
| `--decoder supertonic3` | The official vocoder from Hugging Face. A path to an AE `.pt` checkpoint freezes that checkpoint's decoder instead. |
| `--init_encoder` | Encoder initialisation: an AE `.pt` checkpoint or a BlueCodec `.safetensors` (e.g. the Hub `model.safetensors`). Fresh optimizer, step 0. |
| `--total_steps` | Loop length and cosine `T_max` (lr down to 1e-6). |
| `--batch_size` | Per-process batch size (overrides `ae.train.batch_size`). |
| `--d_warmup N` | Discriminators frozen and a reconstruction-only generator loss for the first N steps. |
| `--recon_logmel_fullband` | Reconstruction L1 on **log** mels up to sr/2, over the **whole segment** (the 0.19 s crop is kept for the adversarial terms only). Default: linear-magnitude mels capped at 12 kHz on the crop. |
| `--fm_composite` | Feature matching averaged over the MPD and MRD layers together (paper Eq. 6). Default: the two averages are summed, an effective λ_fm of 0.2. |
| `--lambda_recon` | Reconstruction weight (default 45). |

Checkpoints written in `--encoder_only` mode store `decoder_source` instead of the decoder weights, so such a checkpoint can be shared without redistributing the official vocoder's weights. Resuming with `--resume` restores the encoder, discriminators, optimizers and schedule; the decoder is reloaded from `--decoder`.

### The released run

| Setting | Value |
|---------|-------|
| Encoder init | the 1.5M-step encoder (identical to the encoder in the Hub `model.safetensors`); discriminators also from the 1.5M training checkpoint |
| Decoder | official `vocoder.onnx` (md5 `68e5b768…`, Hub revision `3cadd1ee`), frozen |
| Optimizer | AdamW (β = 0.8, 0.99, wd 0.01), fresh state |
| LR | 8.5e-5, cosine to 1e-6 over 300k steps |
| Batch | 2 GPUs × 64 segments of 61,740 samples (1.4 s) |
| Losses | λ_recon 45 on full-band log mel over the whole segment, λ_adv 1, λ_fm 0.1 (composite) |
| D warm-up | 10k steps reconstruction-only |
| Hardware / time | 2× RTX 5090, 2026-09-22 15:22 → 2026-09-23 05:33 |
| Released checkpoint | `ae_290000.pt` (the loop ends at 300k before the 300k save) |
| Mel loss (TensorBoard) | 3.17 (first 50 logged points) → 1.19 (last 200) |

When starting from the Hub `model.safetensors`, the discriminators start from scratch (the Hub file has no discriminator weights); the 10k-step D warm-up covers that start.

**After training an encoder, recompute the latent statistics** with the new encoder, and use edge-padded encoding (`BlueCodec.encode(..., edge_pad_chunks=2)` / `bluecodec.utils.encode_wav_edge_padded`) for anything a downstream model will see. See the README's *Latent conventions* section.

### Edge-fixed encoder

The edge-fixed encoder (Hub `encoder_supertonic3_decoder_edge_fixed/`) removes the end-of-clip edge code at the source rather than by padding. It is the released `ae_290000.pt` continued for 60k steps (to 350k) against the same frozen vocoder, with encoder, discriminators and optimizer moments carried over, AdamW lr 2e-5 with a 1k-step warm-up and cosine decay, and two additions: edge-aware batches (variable-length segments, half ending at the clip's true end, batches padded to a multiple of 3072 samples, loss masked per clip at `ceil(L/3072) * 3072`) and, from step 300k, a tail-consistency loss with the encoder's BatchNorm frozen (relative L2 between each clip's last compressed latent frame and an encoding of the same clip with 2 extra chunks of silence; weight 7.5 from step 310k). These two terms were run with BlueTTS's trainer and are not flags of `train_autoencoder.py` here. Results: [technical report §12](../assets/TechnicalReport.md#12-edge-fixed-encoder-sept-2026).

---

## 📊 Model Training Details

The 1.5M-step pretrained model (`model.safetensors`) was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
