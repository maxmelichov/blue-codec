import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D, REPLICATE_DECODER_CFG
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram

HOP = 512

# E12b: encoder + decoder continued from the 1.5M model. Its decoder pads causally with "replicate", and it
# was trained on audio padded to a multiple of 3072 samples (hop 512 x compression 6).
E12B_FILE = "e12b/model.safetensors"
E12B_CHUNK = 3072
E12B_DECODER_CFG = REPLICATE_DECODER_CFG


def _weights_path(repo_id, filename, revision=None):
    """`repo_id` is a Hub repo id, a local directory laid out like one, or a .safetensors file."""
    if os.path.isfile(repo_id):
        return repo_id
    if os.path.isdir(repo_id):
        path = os.path.join(repo_id, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _file_config(path):
    """The architecture config stored in the file's metadata (`bluecodec_config`), or None for older files."""
    with safe_open(path, "pt") as f:
        raw = (f.metadata() or {}).get("bluecodec_config")
    return json.loads(raw) if raw else None


class BlueCodec(nn.Module):
    def __init__(self, decoder_cfg=None, chunk=None, encoder_cfg=None):
        super().__init__()
        self.encoder = LatentEncoder(cfg=encoder_cfg)
        self.decoder = LatentDecoder1D(cfg=decoder_cfg)
        self.mel_transform = LinearMelSpectrogram(n_mels=228)
        # chunk: pad the audio to a multiple of `chunk` samples before encoding (None: no padding)
        self.chunk = chunk

    @classmethod
    def from_pretrained(cls, repo_id="notmax123/blue-codec", filename=None, device="cpu", model=None, revision=None):
        """model=None: `model.safetensors` (the BlueV3 codec; its architecture config is stored in the file).
        model="e12b": E12b (`e12b/model.safetensors`). `filename` overrides the file; `repo_id` may also be a local
        directory or a .safetensors file. The 1.5M-step codec: revision="v3.0.0" (its root `model.safetensors`)."""
        if model is None:
            path = _weights_path(repo_id, filename or "model.safetensors", revision)
            cfg = _file_config(path)
            if cfg is None:   # older file without a stored config: the 1.5M-step layout
                codec = cls()
            else:
                codec = cls(decoder_cfg={**REPLICATE_DECODER_CFG, **cfg["decoder"]}, encoder_cfg=cfg["encoder"])
            sd = load_file(path, device=device)
        elif model == "e12b":
            codec = cls(decoder_cfg=dict(E12B_DECODER_CFG), chunk=E12B_CHUNK)
            filename = filename or E12B_FILE
            try:
                sd = load_file(_weights_path(repo_id, filename, revision), device="cpu")
            except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError) as e:
                raise FileNotFoundError(
                    f"E12b weights not found at {repo_id}/{filename}: they are not published on the Hub yet. "
                    "Pass a local directory as repo_id if you have them.") from e
        else:
            raise ValueError(f"model must be None or 'e12b', got {model!r}")
        codec.encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
        codec.decoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")}, strict=True)
        codec.to(device)
        codec.eval()
        return codec

    @torch.no_grad()
    def encode(self, audio):
        """audio [B, L] at 44.1 kHz -> latents [B, 24, T]; T = L // 512 + 1, or 6 * ceil(L / 3072) for E12b."""
        if self.chunk is None:
            return self.encoder(self.mel_transform(audio))
        L = audio.shape[-1]
        L_pad = -(-L // self.chunk) * self.chunk
        z = self.encoder(self.mel_transform(F.pad(audio, (0, L_pad - L))))
        return z[..., : L_pad // HOP]   # drops the extra frame of the centred STFT

    @torch.no_grad()
    def decode(self, latents):
        return self.decoder(latents)
