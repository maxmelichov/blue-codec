import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram


class BlueCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder1D()
        self.mel_transform = LinearMelSpectrogram(n_mels=228)

    @classmethod
    def from_pretrained(cls, repo_id="notmax123/blue-codec", filename="model.safetensors", device="cpu"):
        model = cls()
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        sd = load_file(path, device=device)
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def encode(self, audio):
        mel = self.mel_transform(audio)
        return self.encoder(mel)

    @torch.no_grad()
    def decode(self, latents):
        return self.decoder(latents)
