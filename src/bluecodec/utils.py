import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

def compress_latents(z: torch.Tensor, factor: int = 6) -> torch.Tensor:
    """
    Compress latent sequence by grouping 'factor' frames.
    Input: [B, C, T]
    Output: [B, C * factor, T // factor]
    """
    B, C, T = z.shape
    # Pad if necessary
    if T % factor != 0:
        pad = factor - (T % factor)
        z = torch.nn.functional.pad(z, (0, pad))
        T = T + pad
        
    z = z.view(B, C, T // factor, factor)
    z = z.permute(0, 1, 3, 2)             # [B, 24, 6, T_low]
    z = z.flatten(1, 2)                   # [B, 144, T_low]
    return z

def decompress_latents(z: torch.Tensor, factor: int = 6, target_channels: int = 24) -> torch.Tensor:
    """
    Decompress latents (inverse of compress_latents).
    Input: [B, 144, T_low]
    Output: [B, 24, T_high]
    """
    B, C_total, T_low = z.shape
    # z: [B, 144, T_low]
    # Unflatten 144 -> 24, 6
    z = z.view(B, target_channels, factor, T_low) # [B, 24, 6, T_low]
    # Permute back to [B, 24, T_low, 6]
    z = z.permute(0, 1, 3, 2)
    # Reshape to [B, 24, T_high]
    z = z.flatten(2, 3) # [B, 24, 6*T_low]
    return z

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=None,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=1.0,
        )
    
    def forward(self, audio):
        mel = self.mel(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if mel.dim() == 4 and mel.shape[1] == 1:
             mel = mel.squeeze(1)
        return mel

class MelSpectrogramNoLog(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=12000,
        power=1.0,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            center=True,
            power=power,
        )

    def forward(self, audio):
        mel = self.mel(audio)
        # No log here
        if mel.dim() == 4 and mel.shape[1] == 1:
            mel = mel.squeeze(1)
        return mel

class LinearMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=1253,
        f_min=0,
        f_max=None,
    ):
        super().__init__()
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            power=1.0,
        )
        self.mel_scale = T.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
        )

    def forward(self, audio):
        spec = self.spectrogram(audio)
        mel = self.mel_scale(spec)
        
        spec = torch.log(torch.clamp(spec, min=1e-5))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        # Concatenate along the channel/frequency dimension (dim=1 for [B, C, T])
        return torch.cat([spec, mel], dim=1)