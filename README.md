# BlueCodec

Speech autoencoder that compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz, then reconstructs the waveform.

## Installation

```bash
uv add "bluecodec @ git+https://github.com/maxmelichov/blue-codec.git"
```

## Usage

See `examples/` folder for a full working example.

## Pretrained Model

The model was trained on 2x NVIDIA RTX 3090 GPUs for 4 weeks for 1.5 million steps on 6 million files with different languages, totaling about 11,000 hours of audio.

Pretrained weights: [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) on Hugging Face.

## Training

For detailed instructions on how to train the Autoencoder, please refer to the [Training Documentation](docs/training.md).

The trainer also supports encoder-only training against a frozen decoder and the optional training terms of the E12b model (see the documentation).

The current recipe trains in two stages on the official Supertonic-3 decoder: see [section 6](docs/training.md#-6-two-stage-training-on-the-official-supertonic-3-decoder) for the commands and held-out results.

## References and acknowledgements

The official decoder is the vocoder of **Supertonic 3** by Supertone Inc. ([Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3); SupertonicTTS paper: [arXiv:2503.23108](https://arxiv.org/abs/2503.23108)). Its weights are released under the [BigScience OpenRAIL-M license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) and are **not bundled** here: `latent_decoder.supertonic3_decoder_state` downloads `onnx/vocoder.onnx` at a pinned revision and maps it 1:1 onto `LatentDecoder1D` (needs `pip install onnx`). Stage-2 checkpoints contain a fine-tuned copy of those weights, so treat them as subject to OpenRAIL-M as well.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
