# BlueCodec

Speech autoencoder that compresses 44.1 kHz audio into a compact 24-dimensional continuous latent representation at ~86 Hz, then reconstructs the waveform.

## Installation

```bash
uv add "bluecodec @ git+https://github.com/maxmelichov/blue-codec.git"
```

## Usage

```python
import torchaudio
from bluecodec import BlueCodec

codec = BlueCodec.from_pretrained("notmax123/blue-codec", device="cuda")

audio, sr = torchaudio.load("audio.wav")
audio = audio.to("cuda")

latents = codec.encode(audio)
reconstructed = codec.decode(latents)

torchaudio.save("reconstructed.wav", reconstructed.cpu(), 44100)
```

See `examples/` for a full working example.

## Model

Pretrained weights: [notmax123/blue-codec](https://huggingface.co/notmax123/blue-codec) on Hugging Face.
