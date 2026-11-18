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

## Training

For detailed instructions on how to train the Autoencoder, please refer to the [Training Documentation](docs/training.md).

The trainer also supports encoder-only training against a frozen decoder and the optional training terms of the E12b model (see the documentation).

Two-stage training on the official Supertonic-3 decoder: [section 6](docs/training.md#-6-two-stage-training-on-the-official-supertonic-3-decoder).

## References and acknowledgements

The official decoder is the vocoder of [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) by Supertone Inc. (SupertonicTTS, [arXiv:2503.23108](https://arxiv.org/abs/2503.23108)). Its weights are under the [OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE), are downloaded at run time and not bundled here; stage-2 checkpoints contain a fine-tuned copy of them.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
