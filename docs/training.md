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

This mode trains **only the encoder** (plus the discriminators) against a **frozen decoder**, e.g. the official vocoder (see the README's *References and acknowledgements*). The decoder stays in the graph, so the gradients flow *through* it into the encoder, which learns to emit the latent space that decoder expects. `--decoder supertonic3` downloads the official `onnx/vocoder.onnx` at start-up (needs `pip install onnx`); its weights are never written into checkpoints.

```bash
uv pip install onnx
uv run torchrun --nproc_per_node=2 train_autoencoder.py \
    --encoder_only --decoder supertonic3 \
    --init_encoder path/to/model.safetensors \
    --lr 8.5e-5 --batch_size 64 --total_steps 300000 --d_warmup 10000 \
    --recon_logmel_fullband --fm_composite \
    --checkpoint_dir checkpoints/ae_official_vocoder
```

| Flag | Meaning |
|------|---------|
| `--encoder_only` | Decoder frozen (no gradients, `eval()`, not DDP-wrapped, not in any optimizer), asserted unchanged at every save. |
| `--decoder supertonic3` | The official vocoder from Hugging Face. A path to an AE `.pt` checkpoint freezes that checkpoint's decoder instead. |
| `--init_encoder` | Encoder initialisation: an AE `.pt` checkpoint or a BlueCodec `.safetensors` (e.g. the Hub `model.safetensors`). Fresh optimizer, step 0. |
| `--total_steps` | Loop length and cosine `T_max` (lr down to 1e-6). |
| `--batch_size` | Per-process batch size (overrides `ae.train.batch_size`). |
| `--d_warmup N` | Discriminators frozen and a reconstruction-only generator loss for the first N steps. |
| `--recon_logmel_fullband` | Reconstruction L1 on **log** mels up to sr/2 over the whole segment (default: linear mels capped at 12 kHz on the 0.19 s crop). |
| `--fm_composite` | Feature matching averaged over the MPD and MRD layers together (paper Eq. 6). |

`--resume` restores the encoder, discriminators, optimizers and schedule; the decoder is reloaded from `--decoder`.

### The released run

| Setting | Value |
|---------|-------|
| Encoder init | the 1.5M-step encoder (identical to the encoder in the Hub `model.safetensors`) |
| Decoder | official `vocoder.onnx` (Hub revision `3cadd1ee`), frozen |
| Stage 1 (0 → 290k) | the command above: AdamW (β = 0.8, 0.99, wd 0.01), lr 8.5e-5 cosine, 2 GPUs × 64 segments of 61,740 samples, λ_recon 45, λ_adv 1, λ_fm 0.1, 10k-step D warm-up |
| Stage 2 (290k → 350k) | lr 2e-5 (1k warm-up, cosine), with edge-aware batches and a tail-consistency term so the encoder's last frames match the rest of the clip (run with BlueTTS's trainer; not flags here) |
| Hardware | 2× RTX 5090 |
| Released | `encoder_supertonic3_decoder_edge_fixed/encoder.safetensors` (step 350k) |

---

## 📊 Model Training Details

The 1.5M-step pretrained model (`model.safetensors`) was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
