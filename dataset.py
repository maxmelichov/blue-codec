import csv
import os
import random

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from bluecodec.audio_utils import ensure_sr

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _scan_dir(root):
    return sorted(os.path.join(d, f) for d, _, files in os.walk(root) for f in files if f.lower().endswith(AUDIO_EXTS))


def _read_metadata(path):
    """`wav_path|...` metadata (LJSpeech style): first column only, relative to `<dir>/wavs` if it exists."""
    root = os.path.dirname(path)
    wavs = os.path.join(root, "wavs")
    base = wavs if os.path.isdir(wavs) else root
    files = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="|"):
            if not row:
                continue
            p = os.path.join(base, row[0])
            if not os.path.splitext(p)[1]:
                p = next((p + e for e in AUDIO_EXTS if os.path.exists(p + e)), p + ".wav")
            files.append(p)
    return files


class TTSDataset(Dataset):
    """Audio-only dataset for autoencoder training: mono float waveforms at `sample_rate`,
    randomly cropped to `segment_size` samples when longer.

    `data_sources`: one or more directories (scanned recursively) or `|`-separated metadata files.
    """

    def __init__(self, data_sources, sample_rate=44100, segment_size=None):
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.files = []
        for src in [data_sources] if isinstance(data_sources, str) else data_sources:
            if os.path.isdir(src):
                found = _scan_dir(src)
            elif os.path.isfile(src):
                found = _read_metadata(src)
            else:
                print(f"warning: data source not found, skipped: {src}")
                continue
            print(f"{src}: {len(found)} audio files")
            self.files += found
        if not self.files:
            raise ValueError("no audio files found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            wav, sr = sf.read(self.files[idx], dtype="float32", always_2d=True)   # [T, C]
        except Exception as e:
            print(f"skipping unreadable file {self.files[idx]}: {e}")
            return self[random.randrange(len(self))]
        wav = torch.from_numpy(wav).mean(dim=1)                                  # mono [T]
        if sr != self.sample_rate:
            wav = ensure_sr(wav, sr, self.sample_rate).squeeze(0)
        if self.segment_size is not None and wav.shape[0] > self.segment_size:
            start = random.randint(0, wav.shape[0] - self.segment_size)
            wav = wav[start:start + self.segment_size]
        return wav


def collate_fn(batch):
    """Right-pad with zeros to the longest clip: [B, 1, T]."""
    n = max(w.shape[0] for w in batch)
    return torch.stack([F.pad(w, (0, n - w.shape[0])) for w in batch]).unsqueeze(1)
