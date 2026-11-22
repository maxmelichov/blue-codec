# BlueCodec

Speech autoencoder that compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz, then reconstructs the waveform.

## Installation

```bash
uv add "bluecodec @ git+https://github.com/maxmelichov/blue-codec.git"
```

## Usage

See `examples/` folder for a full working example.

## Pretrained Model

Pretrained weights: [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) on Hugging Face, one file
(`model.safetensors`, the BlueV3 codec; its architecture config is stored inside the file):

```python
from bluecodec import BlueCodec
codec = BlueCodec.from_pretrained("notmax123/blue-codec")
latents = codec.encode(audio)        # audio [B, L] at 44.1 kHz -> [B, 24, L // 512 + 1]
audio_out = codec.decode(latents)
```

The earlier 1.5M-step codec stays on the Hub under the tag `v3.0.0`:
`BlueCodec.from_pretrained("notmax123/blue-codec", revision="v3.0.0")`.

## Reconstruction quality

Blue codec is a continuous-latent audio autoencoder built as the latent space for a flow-matching text-to-speech model, not as a low-bitrate codec. It maps 44.1 kHz audio to 24 unquantised channels at 86.13 frames per second and back, so it has no bitrate, and the table below compares reconstruction only. On this test Blue has the highest STOI by a small margin and sits with VibeVoice behind DAC in PESQ and speaker similarity; its mel distance is sixth of the twelve systems, and DAC is the best system in PESQ, mel distance and SIM.

| Codec | Licence | Latent | Rate | PESQ ↑ | STOI ↑ | Mel ↓ | SIM ↑ | MOS ↑ |
|---|---|---|---|---|---|---|---|---|
| **Blue codec** (44.1 kHz) | MIT (code), OpenRAIL-M (weights) | continuous, 24 ch | 86.13 Hz (2,067 values/s) | 3.57 | **0.980** | 1.338 | 0.984 | 3.22 |
| DAC (44.1 kHz) | MIT | RVQ, 9 × 1024 | 86.13 Hz, 7.75 kbps | **3.91** | 0.975 | **0.842** | **0.988** | 3.19 |
| EnCodec (24 kHz) | MIT | RVQ, 8 × 1024 | 75 Hz, 6 kbps | 2.63 | 0.943 | 1.655 | 0.968 | 2.96 |
| EnCodec (48 kHz) | MIT | RVQ, 4 × 1024 | 150 Hz, 6 kbps | 2.06 | 0.909 | 1.375 | 0.952 | 2.67 |
| Mimi (24 kHz) | CC BY 4.0 | RVQ, 8 × 2048 | 12.5 Hz, 1.1 kbps | 2.35 | 0.923 | 1.889 | 0.946 | 3.07 |
| SNAC (24 kHz) | MIT | multi-scale RVQ, 3 × 4096 | 12 / 23 / 47 Hz, 0.98 kbps | 2.40 | 0.921 | 1.575 | 0.946 | 3.21 |
| SNAC (32 kHz) | MIT | multi-scale RVQ, 4 × 4096 | 11 / 21 / 42 / 84 Hz, 1.9 kbps | 2.66 | 0.932 | 1.317 | 0.951 | 3.15 |
| SNAC (44.1 kHz) | MIT | multi-scale RVQ, 4 × 4096 | 14 / 29 / 58 / 116 Hz, 2.6 kbps | 2.79 | 0.935 | 1.130 | 0.958 | 3.13 |
| WavTokenizer large-speech (24 kHz) | MIT | single VQ, 4096 | 75 Hz, 0.9 kbps | 2.63 | 0.933 | 1.580 | 0.947 | 3.24 |
| WavTokenizer large-unify (24 kHz) | MIT | single VQ, 4096 | 40 Hz, 0.48 kbps | 1.92 | 0.883 | 1.852 | 0.915 | 3.07 |
| VibeVoice acoustic tokenizer (24 kHz) | MIT | continuous, 64 ch | 7.5 Hz (480 values/s) | 3.53 | 0.973 | 1.321 | 0.984 | **3.25** |
| ACE-Step 1.5 VAE (48 kHz, music) | MIT | continuous, 64 ch | 25 Hz (1,600 values/s) | 3.25 | 0.965 | 1.031 | 0.982 | 3.12 |

Bold marks the best value in each column. Values are means over the 200 utterances; 95% confidence intervals over the 24 speakers are within ±0.15 PESQ, ±0.006 STOI, ±0.04 Mel, ±0.013 SIM and ±0.07 MOS.

The test set is 200 utterances from 24 speakers (8 or 9 each) of the McGill TSP speech database (Harvard sentences recorded at 48 kHz in an anechoic room, Simplified BSD licence); each utterance had its edge silence trimmed, was resampled to the codec's sample rate with no level change, and was encoded and decoded once with the codec's public checkpoint. PESQ is wide-band PESQ (ITU-T P.862.2) and STOI is short-time objective intelligibility, both at 16 kHz; Mel is the multi-scale mel-spectrogram distance from the DAC paper, computed at 44.1 kHz for every system so it also counts the band that 24 kHz codecs cannot carry; SIM is the cosine similarity between WavLM-base-plus-SV speaker embeddings of the input and the reconstruction; MOS is DNSMOS P.835 overall, a reference-free estimate that gives the uncoded recordings 3.19.

## Training

For detailed instructions on how to train the Autoencoder, please refer to the [Training Documentation](docs/training.md).

The trainer also supports encoder-only training against a frozen decoder and the optional training terms of the E12b model (see the documentation).

Two-stage training on the official Supertonic-3 decoder: [section 6](docs/training.md#-6-two-stage-training-on-the-official-supertonic-3-decoder).

## References and acknowledgements

The official decoder is the vocoder of [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) by Supertone Inc. (SupertonicTTS, [arXiv:2503.23108](https://arxiv.org/abs/2503.23108)). Its weights are under the [OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), are downloaded at run time and not bundled here; stage-2 checkpoints contain a fine-tuned copy of them.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
