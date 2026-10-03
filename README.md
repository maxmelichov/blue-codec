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
# codec = BlueCodec.from_pretrained("notmax123/blue-codec", decoder="supertonic3")   # official-vocoder encoder + official vocoder (needs `pip install onnx`)

latents = codec.encode(audio)                      # [1, L] @ 44.1 kHz -> [1, 24, L // 512 + 1]
reconstructed = codec.decode(latents)[..., :audio.shape[-1]]
```

## Pretrained Models

| Model | Files on [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) | Encoder | Decoder | Load with |
|-------|------|---------|---------|-----------|
| **BlueCodec 1.5M** | `model.safetensors` | 1.5M steps | ours, 1.5M steps | `BlueCodec.from_pretrained("notmax123/blue-codec")` |
| **Official-vocoder encoder** (Sept 2026) | `encoder_supertonic3_decoder_edge_fixed/encoder.safetensors` | 1.5M encoder + 350k steps against the frozen official vocoder | **official vocoder**, downloaded from its Hugging Face repo at load time, never redistributed here (see [References](#references-and-acknowledgements)) | `BlueCodec.from_pretrained("notmax123/blue-codec", decoder="supertonic3")` |

The 1.5M model was trained on 2x NVIDIA RTX 3090 GPUs for 4 weeks for 1.5 million steps on 6 million files with different languages, totaling about 11,000 hours of audio.

The official-vocoder encoder keeps our encoder architecture and front end but retargets it to the latent space of a frozen pretrained vocoder (the official vocoder). Its latents are not interchangeable with the 1.5M model's. Round-trip audit on 10 reference clips (details in the [technical report](assets/TechnicalReport.md#10-encoder-trained-against-a-frozen-pretrained-vocoder-sept-2026)):

| Codec | log-mel L1* | 228-band log-mel L1 | 12-22 kHz vs source | DNSMOS |
|-------|------------:|--------------------:|--------------------:|-------:|
| BlueCodec 1.5M (`model.safetensors`) | 0.4426 | 1.0094 | +19.56 dB | 3.40 |
| **Official-vocoder encoder** + official vocoder | **0.3388** | **0.8839** | **+7.10 dB** | **3.44** |

\* L1 of the log of the encoder's input features (see the report).

## Training

For detailed instructions on how to train the Autoencoder (including encoder-only training against a frozen decoder), please refer to the [Training Documentation](docs/training.md).

## References and acknowledgements

The official vocoder is the decoder of **Supertonic 3** by Supertone Inc. ([Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3); SupertonicTTS paper: [arXiv:2503.23108](https://arxiv.org/abs/2503.23108)). Its weights are **Supertone Inc.'s**, released under the [BigScience OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), and are **not bundled or re-uploaded** here: `bluecodec/autoencoder/latent_decoder.py` downloads `onnx/vocoder.onnx` from that repo at load time (pinned revision `3cadd1ee`; anonymous download works) and maps it 1:1 onto `LatentDecoder1D` with replicate ("edge") causal padding. Reading the ONNX file needs `pip install onnx`. Training checkpoints written in `--encoder_only` mode do not contain the decoder. Because the official-vocoder encoder was trained through that model, treat it as subject to OpenRAIL-M's use-based restrictions (Attachment A) as well.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The official vocoder is not part of this project; it is governed by its own OpenRAIL-M license (see [References and acknowledgements](#references-and-acknowledgements)).
