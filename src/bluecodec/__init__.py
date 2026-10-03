import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram

# Encoder trained against the frozen official vocoder. Only the encoder is hosted in notmax123/blue-codec;
# the decoder is downloaded from its official repo at load time (see README).
SUPERTONIC3_ENCODER_FILE = "encoder_supertonic3_decoder_edge_fixed/encoder.safetensors"


class BlueCodec(nn.Module):
    def __init__(self, decoder_cfg=None):
        super().__init__()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder1D(cfg=decoder_cfg)
        self.mel_transform = LinearMelSpectrogram(n_mels=228)

    @classmethod
    def from_pretrained(cls, repo_id="notmax123/blue-codec", filename="model.safetensors", device="cpu", decoder=None):
        """decoder=None: encoder + decoder from `filename` (the 1.5M-step model).
        decoder="supertonic3": our encoder trained for the official vocoder, plus that vocoder,
        downloaded from its official repo (OpenRAIL-M, not redistributed here; needs `pip install onnx`)."""
        if decoder is None:
            model = cls()
            sd = load_file(hf_hub_download(repo_id=repo_id, filename=filename), device=device)
            model.load_state_dict(sd, strict=False)
        elif decoder == "supertonic3":
            from bluecodec.autoencoder.latent_decoder import SUPERTONIC3_DECODER_CFG, supertonic3_decoder_state
            filename = SUPERTONIC3_ENCODER_FILE if filename == "model.safetensors" else filename
            model = cls(decoder_cfg=dict(SUPERTONIC3_DECODER_CFG))
            sd = load_file(hf_hub_download(repo_id=repo_id, filename=filename), device="cpu")
            model.encoder.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
            model.decoder.load_state_dict(supertonic3_decoder_state(model.decoder.state_dict())[0], strict=True)
        else:
            raise ValueError(f"decoder must be None or 'supertonic3', got {decoder!r}")
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
