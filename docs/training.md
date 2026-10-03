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

## 🎯 4. Encoder-Only Training Against a Frozen Decoder

This mode trains **only the encoder** (plus the discriminators) against a **frozen decoder** taken from an existing AE checkpoint. The decoder stays in the graph, so the gradients flow *through* it into the encoder, which learns to emit the latent space that decoder expects.

```bash
uv run torchrun --nproc_per_node=2 train_autoencoder.py \
    --encoder_only --decoder path/to/ae_checkpoint.pt \
    --init_encoder path/to/model.safetensors \
    --lr 8.5e-5 --batch_size 64 --total_steps 300000 --d_warmup 10000 \
    --recon_logmel_fullband --fm_composite \
    --checkpoint_dir checkpoints/ae_encoder_only
```

| Flag | Meaning |
|------|---------|
| `--encoder_only` | Decoder frozen (no gradients, `eval()`, not DDP-wrapped, not in any optimizer), asserted unchanged at every save; checkpoints store `decoder_source` instead of the decoder weights. |
| `--decoder` | AE `.pt` checkpoint whose decoder is frozen. |
| `--init_encoder` | Encoder initialisation: an AE `.pt` checkpoint or a BlueCodec `.safetensors` (e.g. the Hub `model.safetensors`). Fresh optimizer, step 0. |
| `--total_steps` | Loop length and cosine `T_max` (lr down to 1e-6). |
| `--batch_size` | Per-process batch size (overrides `ae.train.batch_size`). |
| `--checkpoint_dir` | Where checkpoints, logs and eval audio go (default `checkpoints/ae`). |
| `--d_warmup N` | Discriminators frozen and a reconstruction-only generator loss for the first N steps. |
| `--recon_logmel_fullband` | Reconstruction L1 on **log** mels up to sr/2 over the whole segment (default: linear mels capped at 12 kHz on the 0.19 s crop). |
| `--fm_composite` | Feature matching averaged over the MPD and MRD layers together (paper Eq. 6). |

`--resume` restores the encoder, discriminators, optimizers and schedule; the decoder is reloaded from `--decoder`. After training an encoder, recompute any latent statistics with it: its latent space differs from the original encoder's.

---

## 🧩 5. E12b Training Terms

E12b is a full autoencoder continued from the 1.5M-step model for 600k steps with four additions to the losses above. Each is an opt-in flag; with none of them set the recipes above are unchanged. E12b weights will be published separately.

| Flag | Meaning |
|------|---------|
| `--edge_aware` | Half of the crops end at the clip's true end. Batches are padded to a multiple of 3072 samples (hop 512 × 6) and, 85% of the time, cut so every clip ends in the last 3072-sample chunk, as a single clip does when encoded. The loss is masked per clip beyond `ceil(L / 3072) * 3072`. |
| `--tail_consistency W` | Relative L2 between each clip's last compressed latent frame and the same frame encoded with two extra chunks of silence (target without gradient). Needs `--edge_aware` and `--tail_stats`. Freezes the encoder BatchNorm unless `--no_bn_freeze`. |
| `--tail_stats PATH` | Latent stats file (`mean`, `std` over the 144 compressed channels) normalising the tail term. |
| `--hiband_w W` | L1 of the log-magnitude STFT (n_fft 2048, hop 512) above 12 kHz. |
| `--floor_w W`, `--floor_db D` | On 1024-sample frames whose source is below `D` dBFS (default −60), penalises decoded level above `D`. |
| `--drop_list FILE` | Audio paths (one per line) to leave out of training. |
| `--decoder_pad_mode replicate` | Replicate causal padding in the decoder (E12b); the 1.5M decoder uses zeros. |
| `--continue_cosine N`, `--lr_warmup K` | With `--resume`: keep the step counter and optimizer state, then a new cosine from `--lr` to 1e-6 over `N` steps after an optional `K`-step linear warm-up. |

E12b's main stage (steps 50k → 600k), resuming from a full training checkpoint:

```bash
uv run torchrun --nproc_per_node=2 train_autoencoder.py \
    --resume path/to/ae_checkpoint.pt --batch_size 64 \
    --recon_logmel_fullband --fm_composite --decoder_pad_mode replicate \
    --edge_aware --tail_consistency 7.5 --tail_stats path/to/latent_stats.pt \
    --hiband_w 2.0 --floor_w 10 --drop_list path/to/drop_list.txt \
    --lr 1.234e-5 --continue_cosine 550000 --total_steps 600000 \
    --checkpoint_dir checkpoints/ae_e12b
```

The first 50k steps used a 5k-step warm-up to 1.25e-5 with `--no_bn_freeze`. Details: [technical report §10](../assets/TechnicalReport.md#10-e12b-training-additions).

The trainer falls back to CPU (gloo) when no GPU is visible, which is enough for a smoke test.

---

## 📊 Model Training Details

The current pretrained model was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
