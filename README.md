# BlueCodec

Speech autoencoder that compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz, then reconstructs the waveform.

## Installation

```bash
uv add "bluecodec @ git+https://github.com/maxmelichov/blue-codec.git"
```

## Usage

See `examples/` folder for a full working example.

```python
from bluecodec import BlueCodec

codec = BlueCodec.from_pretrained("notmax123/blue-codec")                            # 1.5M-step BlueCodec
# codec = BlueCodec.from_pretrained("notmax123/blue-codec", decoder="supertonic3")   # encoder for the Supertonic-3 vocoder (needs `pip install onnx`)

latents = codec.encode(audio)                      # [1, L] @ 44.1 kHz -> [1, 24, L // 512 + 1]
latents = codec.encode(audio, edge_pad_chunks=2)   # edge-padded      -> [1, 24, 6 * ceil(L / 3072)]
reconstructed = codec.decode(latents)[..., :audio.shape[-1]]
```

### Encoding with edge padding

The encoder is non-causal with zero convolution padding, so the last few latent frames of a clip "see" the end of the array. The encoder trained for the Supertonic-3 vocoder learned to write an **edge code** there: the last compressed frame of a clip that ends right after speech sits at ~9x the median latent norm (see the [technical report](assets/TechnicalReport.md#11-end-of-clip-edge-code-sept-2026)). It does not change the reconstruction, but a downstream model trained on or conditioned by those latents copies it.

`codec.encode(audio, edge_pad_chunks=2)` removes it: the clip is zero-padded to a multiple of 3072 samples (hop 512 x compression 6) plus 2 extra chunks of silence, and only the `6 * ceil(L / 3072)` frames that cover the audio are kept. The reconstruction is unchanged, and the last frame lands within a few percent of an encoding with 1 s of silence after the clip. **Use it whenever that encoder's latents feed another model** (TTS targets, reference clips, latent statistics). `bluecodec.utils.encode_wav_edge_padded` does the same for compressed latents. `edge_pad_chunks=None` (the default) keeps the original behaviour.

## Pretrained Models

| Model | Files on [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) | Encoder | Decoder | Load with |
|-------|------|---------|---------|-----------|
| **BlueCodec 1.5M** | `model.safetensors` | 1.5M steps | ours, 1.5M steps | `BlueCodec.from_pretrained("notmax123/blue-codec")` |
| **Supertonic-3-decoder encoder** (Sept 2026) | `encoder_supertonic3_decoder/encoder.safetensors` (+ `ae_290000.pt`, decoder-free training checkpoint) | 1.5M encoder + 290k steps against the frozen Supertonic-3 vocoder | **official Supertonic-3 vocoder**, downloaded from [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3) at load time, never redistributed here | `BlueCodec.from_pretrained("notmax123/blue-codec", decoder="supertonic3")` |
| **Edge-fixed encoder** | *reserved: `encoder_supertonic3_decoder_edge_fixed/` — retrain in progress, not released* | — | official Supertonic-3 vocoder | — |

The 1.5M model was trained on 2x NVIDIA RTX 3090 GPUs for 4 weeks for 1.5 million steps on 6 million files with different languages, totaling about 11,000 hours of audio.

The Supertonic-3-decoder encoder keeps our encoder architecture and front end but retargets it, for 290k steps, to the latent space of Supertone's official Supertonic-3 vocoder, which stays frozen. Round-trip audit on 10 reference clips (details and caveats in the [technical report](assets/TechnicalReport.md#10-encoder-trained-against-the-frozen-supertonic-3-vocoder-sept-2026)):

| Codec | log-mel L1* | 12-22 kHz vs source | DNSMOS |
|-------|------------:|--------------------:|-------:|
| BlueCodec 1.5M (`model.safetensors`) | 0.4426 | +19.56 dB | 3.40 |
| 1.5M encoder + retrained decoder (not published) | 0.3604 | +1.50 dB | 3.41 |
| **Supertonic-3-decoder encoder** + official vocoder | **0.3385** | +7.63 dB | **3.42** |

\* L1 of the log of the encoder's input features (see the report); on a plain 228-band log-mel L1 the retrained-decoder row is best.

### Licensing of the Supertonic-3 decoder

The decoder weights are **Supertone Inc.'s**, released under the [BigScience OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE). BlueCodec does not bundle or re-upload them: `load_supertonic3_decoder` in `bluecodec/autoencoder/latent_decoder.py` downloads `onnx/vocoder.onnx` from `Supertone/supertonic-3` (pinned revision `3cadd1ee`; anonymous download works) and maps it 1:1 onto `LatentDecoder1D` with replicate ("edge") causal padding. The port matches onnxruntime to float round-off. Reading the ONNX file needs `pip install onnx`. Training checkpoints written in `--encoder_only` mode do not contain the decoder. Because the encoder was trained through Supertone's model, treat it as subject to OpenRAIL-M's use-based restrictions (Attachment A) as well.

## Latent conventions (for downstream TTS)

These are the conventions the [BlueTTS](https://github.com/maxmelichov/BlueTTS) text-to-latent and duration models rely on:

- The codec produces **raw latents `[B, 24, T]` at 86.13 Hz** (44100 / 512).
- `compress_latents(z, factor=6)` folds them to **`[B, 144, T/6]`** (channel index `c * 6 + j`): 14.35 Hz frames that each cover 3072 samples. `decompress_latents` inverts it. A remainder is filled by replicating the last frame (a zero latent is a code the decoder never saw). The flow model and the duration predictor operate on the compressed representation.
- Normalisation is per compressed channel: **`((z - mean) / std) * 0.25`**, with `mean`/`std` (`[144]`) from a stats file computed by encoding the training corpus with the same encoder. Sampling reverses both before `decompress_latents` and the decoder.
- **A stats file is only valid for the encoder and corpus it was computed with.** BlueTTS stats files record the AE checkpoint's md5 and the corpus CSV in a `provenance` entry. Changing the encoder invalidates the stats and every downstream checkpoint trained on them.
- The Supertonic-3-decoder encoder's latents live in the Supertonic-3 latent space, not in the 1.5M model's; the two encoders are not interchangeable.

## Training

For detailed instructions on how to train the Autoencoder (including encoder-only training against a frozen decoder), please refer to the [Training Documentation](docs/training.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The official Supertonic-3 vocoder is not part of this project; it is governed by Supertone's OpenRAIL-M license (see above).
