import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram, decompress_latents, encode_wav_edge_padded

# Encoder trained against the frozen official Supertonic-3 vocoder (Sept 2026). Only the encoder is
# hosted in notmax123/blue-codec; the decoder is downloaded from Supertone/supertonic-3 at load time.
SUPERTONIC3_ENCODER_FILE = "encoder_supertonic3_decoder/encoder.safetensors"


class BlueCodec(nn.Module):
    def __init__(self, decoder_cfg=None):
        super().__init__()
        self.encoder = LatentEncoder()
        self.decoder = LatentDecoder1D(cfg=decoder_cfg)
        self.mel_transform = LinearMelSpectrogram(n_mels=228)

    @classmethod
    def from_pretrained(cls, repo_id="notmax123/blue-codec", filename="model.safetensors", device="cpu", decoder=None):
        """decoder=None: encoder + decoder from `filename` (the 1.5M-step model).
        decoder="supertonic3": our encoder trained for the official Supertonic-3 vocoder, plus that vocoder,
        downloaded from Supertone/supertonic-3 (OpenRAIL-M, not redistributed here; needs `pip install onnx`)."""
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
    def encode(self, audio, edge_pad_chunks=None):
        """audio [B, L] at 44.1 kHz -> latents [B, 24, T].
        edge_pad_chunks=None: original behaviour (T = L // 512 + 1).
        edge_pad_chunks=2 (recommended for the Supertonic-3-decoder encoder): pad to a multiple of 3072 samples
        plus 2 silent chunks and keep T = 6 * ceil(L / 3072), so no kept frame sees the array edge."""
        if edge_pad_chunks is None:
            return self.encoder(self.mel_transform(audio))
        zc, _ = encode_wav_edge_padded(self.encoder, self.mel_transform, audio, factor=6, hop=512, edge_pad_chunks=edge_pad_chunks)
        return decompress_latents(zc, factor=6, target_channels=24)

    @torch.no_grad()
    def decode(self, latents):
        return self.decoder(latents)
