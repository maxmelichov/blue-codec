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
    `end_crop_prob`: probability that a crop ends at the clip's true end instead of a random position.
    `drop_list`: optional text file of audio paths (one per line) to leave out.
    """

    def __init__(self, data_sources, sample_rate=44100, segment_size=None, end_crop_prob=0.0, drop_list=None):
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.end_crop_prob = end_crop_prob
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
        if drop_list:
            with open(drop_list, encoding="utf-8") as fh:
                drop = {line.strip() for line in fh if line.strip()}
            n = len(self.files)
            self.files = [f for f in self.files if f not in drop]
            print(f"{drop_list}: dropped {n - len(self.files)} of {n} files")
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
            max_start = wav.shape[0] - self.segment_size
            at_end = self.end_crop_prob > 0 and random.random() < self.end_crop_prob
            start = max_start if at_end else random.randint(0, max_start)
            wav = wav[start:start + self.segment_size]
        return wav


def collate_fn(batch):
    """Right-pad with zeros to the longest clip: [B, 1, T]."""
    n = max(w.shape[0] for w in batch)
    return torch.stack([F.pad(w, (0, n - w.shape[0])) for w in batch]).unsqueeze(1)


def _edge_length(wav_len, array_len, chunk):
    """Length of the tail of a clip kept so that it ends within the last `chunk` samples of the array."""
    hi = min(array_len, wav_len)
    lo = array_len - chunk + 1
    if hi < lo:
        return wav_len                                  # too short: kept whole, silence-padded
    r = random.random()
    if r < 0.25:
        return hi                                       # audio up to the array end
    if r < 0.5:
        return random.randint(max(lo, hi - 511), hi)    # within one hop of it
    return random.randint(lo, hi)


def collate_fn_edge(batch, chunk=3072, aligned_prob=0.85, min_frac=0.4):
    """Edge-aware batches (`--edge_aware`): the array length is a multiple of `chunk` samples.

    With probability `aligned_prob`, an array length Lp (a multiple of `chunk`, between `min_frac` and 1 times the
    longest clip) is drawn and each clip keeps its last samples so that it ends in the last chunk of the array,
    the way a single clip ends when it is encoded. Returns (wavs [B, 1, T], lengths [B]).
    """
    if random.random() < aligned_prob:
        longest = max(w.shape[0] for w in batch)
        Lp = max(chunk, -(-random.randint(int(min_frac * longest), longest) // chunk) * chunk)
        batch = [w[w.shape[0] - _edge_length(w.shape[0], Lp, chunk):] for w in batch]
    lengths = torch.tensor([w.shape[0] for w in batch])
    T = -(-int(lengths.max()) // chunk) * chunk
    return torch.stack([F.pad(w, (0, T - w.shape[0])) for w in batch]).unsqueeze(1), lengths
