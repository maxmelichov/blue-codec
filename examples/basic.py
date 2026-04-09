"""
Basic encode/decode example for BlueCodec.

Download a sample audio file first:
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav

Then run:
    python examples/basic.py
"""

import torchaudio
from bluecodec import BlueCodec

device = "cuda"
codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=device)

audio, sr = torchaudio.load("female1.wav", backend="soundfile")
if sr != 44100:
    audio = torchaudio.functional.resample(audio, sr, 44100)
audio = audio.to(device)  # [1, T]

latents = codec.encode(audio)
print(f"Latent shape: {latents.shape}")

reconstructed = codec.decode(latents)
torchaudio.save("reconstructed.wav", reconstructed.cpu(), 44100)
print("Saved reconstructed.wav")
