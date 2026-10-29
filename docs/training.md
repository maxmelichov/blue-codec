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
| `--decoder` | AE `.pt` checkpoint whose decoder is frozen, or `official` (section 5). |
| `--init_encoder` | Encoder initialisation: an AE `.pt` checkpoint or a BlueCodec `.safetensors` (e.g. the Hub `model.safetensors`). Fresh optimizer, step 0. |
| `--total_steps` | Loop length and cosine `T_max` (lr down to 1e-6). |
| `--batch_size` | Per-process batch size (overrides `ae.train.batch_size`). |
| `--checkpoint_dir` | Where checkpoints, logs and eval audio go (default `checkpoints/ae`). |
| `--d_warmup N` | Discriminators frozen and a reconstruction-only generator loss for the first N steps. |
| `--recon_logmel_fullband` | Reconstruction L1 on **log** mels up to sr/2 over the whole segment (default: linear mels capped at 12 kHz on the 0.19 s crop). |
| `--fm_composite` | Feature matching averaged over the MPD and MRD layers together (paper Eq. 6). |

`--resume` restores the encoder, discriminators, optimizers and schedule; the decoder is reloaded from `--decoder`. After training an encoder, recompute any latent statistics with it: its latent space differs from the original encoder's.

---

## 🧭 5. Two-Stage Training on the Official Supertonic-3 Decoder

The current recipe. The decoder is the official Supertonic-3 vocoder (downloaded at a pinned revision, mapped 1:1 onto `LatentDecoder1D`; its output matches the ONNX graph to ~1e-6 relative L2). A new encoder learns that decoder's latent space first, then both are fine-tuned together. Everything the recipe changes is in `config/tts_st3.json`; set `ae.data.train_metadata` and `ae.data.val_metadata` (held-out audio) there.

```bash
# Stage 1: encoder only, decoder frozen
uv run torchrun --nproc_per_node=2 train_autoencoder.py --arch_config config/tts_st3.json \
    --encoder_only --decoder official --checkpoint_dir checkpoints/ae_st3

# Stage 2: when the held-out loss of stage 1 plateaus, continue from its last checkpoint with the decoder trainable
uv run torchrun --nproc_per_node=2 train_autoencoder.py --arch_config config/tts_st3.json \
    --stage2 --resume checkpoints/ae_st3/ae_latest.pt --checkpoint_dir checkpoints/ae_st3 --total_steps 1500000
```

Use `--official_onnx path/to/vocoder.onnx` to read a local copy instead of downloading it. A stage-2 run that stopped is resumed with the same `--stage2 --resume .../ae_latest.pt` command.

**Losses** are the paper's (SupertonicTTS sec. 3.1.2 / App. B.1) and nothing else: multi-resolution mel L1 x45 on **linear** magnitudes up to sr/2 over the whole segment (`"recon": "linmel_fullband"`), LSGAN adversarial x1 and feature matching x0.1 averaged over the MPD and MRD layers together (`"fm_composite": true`), adversarial terms on 0.19 s crops.

**Model**: the 1253-feature encoder input (1025 linear + 228 mel bins) and 10 ConvNeXt blocks with a final LayerNorm, as in Supertonic-3; `"pad_mode": "replicate"` (encoder and decoder) pads the convolutions with edge values instead of zeros, like every Pad node of the official graphs. On a fresh start the encoder's final LayerNorm weight/bias are set to the official latent std/mean (`"enc_ln_init": "official"`), so the first latents already fall in the range the decoder was trained on.

**Optimisation** (`ae.train`):

| Key | Value | Why |
|-----|-------|-----|
| `lr`, `betas` | 2e-4, (0.9, 0.999) | AdamW, the paper's rate |
| `no_decay_norms` | true | No weight decay on norms, biases, PReLU slopes and layer-scale gammas |
| `clip_generator` | false | Encoder/decoder gradients are not clipped; discriminators are clipped at 1 |
| `ema` | 0.999 | Generator EMA, saved as `encoder_ema` / `decoder_ema`; held-out checks and eval renders use it |
| `sync_bn` | true | Encoder BatchNorm statistics over all GPUs, not per GPU |
| `lr_schedule` | `plateau` | 500 warm-up steps, then a flat lr, halved (G and D) when the held-out recon loss has not improved by 1% for 3 checks (`eval_interval` 10k steps), never below `min_lr` 1e-5 |
| `dec_lr_scale`, `dec_warmup_steps` | 0.1, 2000 | Stage 2: decoder lr = 0.1 x lr, ramped linearly from 0 over 2000 steps |

TF32 is disabled, so training is strict fp32.

**Stage 1** freezes the decoder (no gradients, `eval()`, checked unchanged at every save); checkpoints record `decoder_source` instead of the decoder weights. **Stage 2** loads the decoder that stage 1 used and makes it trainable. The encoder keeps its Adam moments; the decoder starts with fresh ones, which is why its lr ramps up. The decoder's BatchNorm statistics stay frozen (its affine parameters still train). The official decoder's final BatchNorm must keep its stored statistics, because re-estimating them on the new encoder's latents breaks reconstruction. Every checkpoint stores the `ae` config it was trained with (`ae_config`) and its `stage`. Training stops at `--total_steps` with a final save.

Results on an internal held-out set (EMA weights, mel L1 as in training):

| | linear-mel L1 | log-mel L1 |
|---|---:|---:|
| Stage 1, plateau | ~0.55 | |
| Stage 2, step 420k | 0.435 | 0.506 |
| Previous autoencoder (E12b), same set | | 0.505 |

On 200 held-out clips the stage-2 model scores PESQ-WB 3.41, STOI 0.971, speaker cosine (input vs reconstruction) 0.971 and a phoneme error rate of 7.8%. Its latents are a new latent space: recompute any latent statistics with this encoder.

---

## 📊 Model Training Details

The current pretrained model was trained with the following specifications:
- **Hardware:** 2× NVIDIA RTX 3090 GPUs
- **Duration:** 4 weeks
- **Steps:** 1.5 million steps
- **Dataset:** 6 million files across different languages (~11,000 hours of audio)
