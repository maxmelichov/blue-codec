import sys
import os
import copy
import json
import math
import subprocess
import atexit
import argparse
import random
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from dataset import TTSDataset, collate_fn, collate_fn_edge
from bluecodec.audio_utils import ensure_sr
from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D, supertonic3_decoder_state
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram, compress_latents

HOP, CHUNK = 512, 3072   # latent hop; edge-aware layout: hop x compression 6

def _mod(m): return m.module if isinstance(m, DDP) else m

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def check_for_nan_inf(loss, name, logger):
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning(f"🚨 {name} detected NaN/Inf")
        return True
    return False

def setup_logger(save_dir, rank):
    logger = logging.getLogger("train_ae")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

def feature_loss(fmap_r, fmap_g):
    loss = 0.0
    count = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=fmap_r[0][0].device)

def feature_loss_composite(*fmap_pairs):
    # Paper Eq. (6): one average over the layers of MPD and MRD together (D is the composite).
    # feature_loss(mpd) + feature_loss(mrd) averages each group separately, i.e. lambda_fm 0.2.
    return feature_loss([d for r, _ in fmap_pairs for d in r], [d for _, g in fmap_pairs for d in g])

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr.float() - 1.0) ** 2)
        g_loss = torch.mean((dg.float() + 1.0) ** 2)
        loss += (r_loss + g_loss)
    return loss

class LogMelFullBand(nn.Module):
    """Paper reconstruction mel: LOG magnitude up to sr/2 (the default loss is linear, capped at 12 kHz)."""
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, eps=1e-5):
        super().__init__()
        self.mel = MelSpectrogramNoLog(sample_rate, n_fft, win_length, hop_length, n_mels, f_min=0, f_max=sample_rate // 2)
        self.eps = eps
    def forward(self, x): return torch.log(self.mel(x).clamp_min(self.eps))

def get_mel_transforms(data_cfg, device, recon="linmel_12k"):
    # linmel_12k (default): linear mels capped at 12 kHz. linmel_fullband: linear mels up to sr/2, the paper's
    # "L1 over multi-resolution mel spectrograms" (no log, no cap). logmel_fullband: log mels up to sr/2.
    sr = data_cfg['sample_rate']
    mel_configs = [(1024, 256, 1024, 64), (2048, 512, 2048, 128), (4096, 1024, 4096, 128)]
    if recon == "logmel_fullband":
        return [LogMelFullBand(sr, n_fft, hop, win, n_mels).to(device) for (n_fft, hop, win, n_mels) in mel_configs]
    f_max = {"linmel_12k": 12000, "linmel_fullband": sr // 2}[recon]
    # keywords: MelSpectrogramNoLog takes (sample_rate, n_fft, win_length, hop_length); passed positionally, window and
    # hop were swapped (window 256 / hop 1024 at n_fft 1024), so every resolution ran with the wrong window and hop
    return [MelSpectrogramNoLog(sr, n_fft=n_fft, win_length=win, hop_length=hop, n_mels=n_mels, f_max=f_max).to(device)
            for (n_fft, hop, win, n_mels) in mel_configs]

def chunk_ceil(n): return (n + CHUNK - 1) // CHUNK * CHUNK

@dataclass
class ExtraTerms:
    """Optional generator terms; all weights 0 is the default recipe."""
    tail_w: float = 0.0
    tail_norm: tuple = None      # (mean, std) [1, 144, 1] of the compressed latents
    hiband_w: float = 0.0
    floor_w: float = 0.0
    floor_db: float = -60.0

def tail_consistency_loss(encoder, mel_transform, audio, z, lengths, tail_norm):
    """Relative L2 between each clip's last compressed latent frame and the same frame encoded with two extra
    chunks of silence after the batch (the target, without gradient)."""
    with torch.no_grad():
        # the bare module: a DDP forward re-broadcasts buffers in place and would break the first pass's graph
        z_ref = _mod(encoder)(mel_transform(F.pad(audio.squeeze(1), (0, 2 * CHUNK))))[..., : z.shape[-1]]
    mean, std = tail_norm
    last = chunk_ceil(lengths) // CHUNK - 1
    rows = torch.arange(z.shape[0], device=z.device)
    a = ((compress_latents(z, 6) - mean) / std)[rows, :, last]
    b = ((compress_latents(z_ref, 6) - mean) / std)[rows, :, last]
    return ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-3)).mean()

def hiband_loss(y_hat, audio, n_fft=2048, min_bin=558):
    """L1 of the log-magnitude STFT above 12 kHz (bin 558 of 2048 at 44.1 kHz)."""
    win = torch.hann_window(n_fft, device=audio.device)
    def logmag(x):
        S = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=HOP, window=win, return_complex=True).abs()
        return torch.log(S[:, min_bin:, :].clamp_min(1e-5))
    return F.l1_loss(logmag(y_hat), logmag(audio))

def silence_floor_loss(y_hat, audio, floor_db, mask=None, frame=1024):
    """On frames whose source is below `floor_db` dBFS, the decoded level above `floor_db`, in units of 20 dB."""
    n = audio.shape[-1] // frame
    def level_db(x): return 10 * x[..., : n * frame].reshape(x.shape[0], n, frame).pow(2).mean(-1).add(1e-12).log10()
    silent = (level_db(audio) < floor_db).float()
    if mask is not None: silent = silent * mask[:, 0, : n * frame : frame]
    return (torch.relu(level_db(y_hat) - floor_db) * silent).sum() / silent.sum().clamp_min(1.0) / 20.0

def extra_losses(terms, encoder, mel_transform, audio, y_hat, z, lengths, mask):
    """{name: (weight, loss)} for the enabled ExtraTerms."""
    out = {}
    if terms.tail_w > 0: out["tail"] = (terms.tail_w, tail_consistency_loss(encoder, mel_transform, audio, z, lengths, terms.tail_norm))
    if terms.hiband_w > 0: out["hiband"] = (terms.hiband_w, hiband_loss(y_hat, audio))
    if terms.floor_w > 0: out["floor"] = (terms.floor_w, silence_floor_loss(y_hat, audio, terms.floor_db, mask))
    return out

def random_crop(audio, y_hat, crop_len, limit):
    """The same random crop of both signals, within the first `limit` samples."""
    if limit <= crop_len: return audio, y_hat
    start = torch.randint(0, limit - crop_len + 1, (1,)).item()
    return audio[..., start : start + crop_len], y_hat[..., start : start + crop_len]

def recon_l1(real, fake, mel_transforms):
    """The reconstruction term: mel L1 averaged over the resolutions. real/fake: [B, 1, T] or [B, T]."""
    # squeeze per transform, as before: one shared view would change the order in which gradients accumulate
    loss = sum(F.l1_loss(tf(real.squeeze(1)), tf(fake.squeeze(1))) for tf in mel_transforms)
    return loss / len(mel_transforms) if mel_transforms else loss

def train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
               update_discriminator=True, lambda_recon=45.0, recon_full_segment=False, fm_composite=False, encoder_only=False, recon_only=False,
               terms=None, clip_generator=True):
    # recon_only: no adversarial or feature-matching terms for the generator.
    # encoder_only: the frozen decoder stays in the graph, so gradients reach the encoder through it.
    # clip_generator=False: only the discriminators are clipped (at 1); encoder/decoder gradients are not rescaled.
    lambda_adv, lambda_fm = 1.0, 0.1
    batch, lengths = batch if isinstance(batch, (tuple, list)) else (batch, None)   # lengths: edge-aware batches
    audio = batch.to(device)
    if audio.dim() == 2: audio = audio.unsqueeze(1) 

    with torch.no_grad():
        mel = mel_transform_input(audio.squeeze(1))
    
    z = encoder(mel)
    mask = None
    if lengths is not None:
        # Keep the array's own frames (the centred STFT adds one) and supervise each clip up to ceil(L / 3072) * 3072.
        lengths = lengths.to(device)
        supervised = chunk_ceil(lengths)
        z = z[..., : audio.shape[-1] // HOP]
        mask = (torch.arange(audio.shape[-1], device=device)[None] < supervised[:, None]).float().unsqueeze(1)
    y_hat = decoder(z)
    if y_hat.dim() == 2: y_hat = y_hat.unsqueeze(1)
    
    if y_hat.shape[-1] != audio.shape[-1]:
        min_len = min(y_hat.shape[-1], audio.shape[-1])
        y_hat, audio = y_hat[..., :min_len], audio[..., :min_len]
    crop_limit = audio.shape[-1]
    if mask is not None:
        y_hat, audio = y_hat * mask, audio * mask
        if int(supervised.min()) > crop_len: crop_limit = int(supervised.min())   # D never sees the masked zeros
    audio_crop, y_hat_crop = random_crop(audio, y_hat, crop_len, crop_limit)

    loss_d_total = 0.0
    if update_discriminator:
        y_hat_detached = y_hat_crop.detach()
        y_df_hat_r, y_df_hat_g, _, _ = mpd(audio_crop, y_hat_detached)
        y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audio_crop, y_hat_detached)
        loss_d_total = discriminator_loss(y_df_hat_r, y_df_hat_g) + discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        if check_for_nan_inf(loss_d_total, "Discriminator Loss", logger): return None
        opt_d.zero_grad()
        loss_d_total.backward()
        torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1.0)
        opt_d.step()
        loss_d_total = loss_d_total.item()

    # Reconstruction on the adversarial crop, or on the whole segment with recon_full_segment.
    rec_real, rec_fake = (audio, y_hat) if recon_full_segment else (audio_crop, y_hat_crop)
    L_recon = recon_l1(rec_real, rec_fake, mel_transforms_loss)

    if not recon_only:
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio_crop, y_hat_crop)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(audio_crop, y_hat_crop)
        L_adv = generator_loss(y_df_hat_g) + generator_loss(y_ds_hat_g)
        if fm_composite: L_fm = feature_loss_composite((fmap_f_r, fmap_f_g), (fmap_s_r, fmap_s_g))
        else: L_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
        loss_g_total = (lambda_recon * L_recon) + (lambda_adv * L_adv) + (lambda_fm * L_fm)
    else:
        loss_g_total = lambda_recon * L_recon
    extras = extra_losses(terms or ExtraTerms(), encoder, mel_transform_input, audio, y_hat, z, lengths, mask)
    for w, loss in extras.values(): loss_g_total = loss_g_total + w * loss
    
    if check_for_nan_inf(loss_g_total, "Generator Loss", logger): return None
    opt_g.zero_grad()
    loss_g_total.backward()
    if clip_generator:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
        if not encoder_only: torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
    opt_g.step()
    return loss_g_total.item(), loss_d_total, L_recon.item(), {k: loss.item() for k, (_, loss) in extras.items()}

def freeze_batchnorm(model):
    """BatchNorm layers in eval mode (running statistics used and not updated); _BatchNorm also covers SyncBatchNorm."""
    for m in _mod(model).modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def make_scheduler(opt, t_max, warmup=0):
    """Cosine decay to 1e-6 over `t_max` steps, after an optional linear warm-up from 1% of the lr."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)
    if not warmup: return cosine
    linear = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(opt, [linear, cosine], milestones=[warmup])

def cosine_schedulers(opt_g, opt_d, t_max, warmup=0):
    """make_scheduler for G and D: (the checkpointed schedulers, the G one that reports the lr)."""
    schedulers = {'scheduler_g': make_scheduler(opt_g, t_max, warmup), 'scheduler_d': make_scheduler(opt_d, t_max, warmup)}
    return schedulers, schedulers['scheduler_g']

def init_distributed(local_rank):
    """NCCL on the local GPU, or gloo on CPU when no GPU is visible. Returns (device, DDP device_ids)."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        return torch.device(f'cuda:{local_rank}'), [local_rank]
    dist.init_process_group(backend='gloo', init_method='env://')
    return torch.device('cpu'), None

def evaluate(encoder, decoder, mel_transform, input_wav_path, output_dir, step, device, target_sr, rank, keep_decoder_eval=False):
    if rank != 0: return
    encoder.eval(); decoder.eval()
    try:
        wav, sr = torchaudio.load(input_wav_path)
        if sr != target_sr: wav = ensure_sr(wav, sr, target_sr)
        wav = wav.mean(dim=0, keepdim=True).to(device)
        with torch.no_grad():
            enc_mod = encoder.module if isinstance(encoder, DDP) else encoder
            dec_mod = decoder.module if isinstance(decoder, DDP) else decoder
            y_hat = dec_mod(enc_mod(mel_transform(wav)))
            max_val = y_hat.abs().max().item()
            y_hat_final = y_hat.squeeze()
            if max_val > 1.0: y_hat_final /= (max_val + 1e-8)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"step_{step}_{os.path.basename(input_wav_path)}_recon.wav")
        sf.write(output_path, y_hat_final.cpu().numpy(), target_sr)
    except Exception as e: print(f"Evaluation failed: {e}")
    encoder.train()
    if not keep_decoder_eval: decoder.train()   # a frozen decoder stays in eval (BatchNorm running stats)

@torch.no_grad()
def heldout_recon(encoder, decoder, mel_transform, mel_transforms_loss, clips, device):
    """Mean reconstruction L1 (the training recon term) over fixed held-out clips, one clip at a time (no padding)."""
    encoder.eval(); decoder.eval()
    total = 0.0
    for wav in clips:
        wav = wav.to(device).unsqueeze(0)                       # [1, T]
        y_hat = decoder(encoder(mel_transform(wav))).reshape(1, -1)
        n = min(y_hat.shape[-1], wav.shape[-1])
        total += recon_l1(wav[..., :n], y_hat[..., :n], mel_transforms_loss).item()
    return total / len(clips)

@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm, 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)   # BatchNorm running stats follow the raw model

def param_groups(named, lr, no_decay_norms):
    """AdamW groups. no_decay_norms: no weight decay on norms, biases, PReLU slopes and layer-scale gammas
    (decay on a scale pulls it toward 0, which is not a regulariser for these)."""
    named = list(named)
    if not no_decay_norms: return [{'params': [p for _, p in named], 'lr': lr}]
    nd = lambda n, p: p.ndim <= 1 or n.endswith('gamma')
    return [{'params': [p for n, p in named if not nd(n, p)], 'lr': lr},
            {'params': [p for n, p in named if nd(n, p)], 'lr': lr, 'weight_decay': 0.0}]

class PlateauLR:
    """Linear warm-up, then a flat lr (the paper gives a flat 2e-4) that is halved, for G and D together, when the
    held-out reconstruction has not improved by `threshold` (relative) for `patience` checks; never below `min_lr`.
    A group with g['ramp_from'] set ramps linearly from 0 over `ramp` steps from that step: stage 2 starts a
    pretrained decoder with fresh Adam moments, whose first full-size steps would otherwise jolt it."""
    def __init__(self, opts, lr, warmup, min_lr, patience, threshold, ramp=0):
        self.opts, self.lr, self.warmup, self.min_lr = opts, lr, warmup, min_lr
        self.patience, self.threshold, self.ramp = patience, threshold, ramp
        self.scale, self.best, self.bad = 1.0, math.inf, 0

    def step(self, step):
        warm = min(1.0, (step + 1) / self.warmup) if self.warmup else 1.0
        for o in self.opts:
            for g in o.param_groups:
                f = self.scale * warm
                if g.get('ramp_from') is not None and self.ramp:
                    f *= min(1.0, max(0, step - g['ramp_from'] + 1) / self.ramp)
                g['lr'] = g['base_lr'] * f

    def report(self, val):
        """Returns True when this check halved the lr."""
        if val < self.best * (1.0 - self.threshold):
            self.best, self.bad = val, 0
            return False
        self.bad += 1
        if self.bad < self.patience: return False
        self.bad = 0
        new = max(self.scale * 0.5, self.min_lr / self.lr)
        halved, self.scale = new < self.scale, new
        return halved

    def get_last_lr(self): return [g['lr'] for g in self.opts[0].param_groups]
    def state_dict(self): return {'scale': self.scale, 'best': self.best, 'bad': self.bad}
    def load_state_dict(self, sd): self.scale, self.best, self.bad = sd['scale'], sd['best'], sd['bad']

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def load_checkpoint(args, ckpt, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, dec_groups, logger):
    """Loads weights, optimizer and scheduler state; returns (step, epoch), (0, 0) for a fresh optimizer and schedule.
    dec_groups (stage 2) are added to opt_g here: a stage-2 checkpoint saved them, a stage-1 one did not."""
    def load_sd(model, sd):
        if any(k.startswith('module.') for k in sd.keys()): sd = {k.replace('module.', ''): v for k, v in sd.items()}
        md = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        if hasattr(model, 'module'): model.module.load_state_dict(filtered, strict=False)
        else: model.load_state_dict(filtered, strict=False)

    def load_state():
        opt_g.load_state_dict(ckpt['opt_g']); opt_d.load_state_dict(ckpt['opt_d'])
        for k, v in schedulers.items():
            if k in ckpt: v.load_state_dict(ckpt[k])
        return ckpt['step'] + 1, ckpt.get('epoch', 0)

    def opt_match(opt, sd):
        return sd and len(opt.param_groups) == len(sd['param_groups']) and all(len(g['params']) == len(sg['params']) for g, sg in zip(opt.param_groups, sd['param_groups']))

    resumed = None
    if ckpt is not None:
        if args.local_rank == 0: logger.info(f"Resuming from {args.resume}...")
        load_sd(encoder, ckpt['encoder'])
        if not args.encoder_only and 'decoder' in ckpt: load_sd(decoder, ckpt['decoder'])   # encoder_only: decoder comes from --decoder
        load_sd(mpd, ckpt['mpd']); load_sd(mrd, ckpt['mrd'])
        # Stage 1 -> 2: the encoder keeps its Adam moments (same tensors, same groups, saved first), loaded
        # before the decoder groups are appended with fresh moments.
        if args.stage2 and ckpt.get('stage') != 2: resumed = load_state()
    for g in dec_groups: opt_g.add_param_group(g)
    if ckpt is None or resumed: return resumed or (0, 0)

    if not args.finetune:
        if opt_match(opt_g, ckpt.get('opt_g')) and opt_match(opt_d, ckpt.get('opt_d')): return load_state()
        if args.local_rank == 0: logger.warning("Optimizer state does not match the model: fresh optimizer, step 0")
    return 0, 0

def save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, logger, ckpt_dir="checkpoints/ae", decoder_source=None, extra=None):
    state = {
        "step": step, "epoch": epoch,
        "encoder": _mod(encoder).state_dict(),
        "mpd": _mod(mpd).state_dict(), "mrd": _mod(mrd).state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        **{k: v.state_dict() for k, v in schedulers.items()},
        **(extra or {}),
    }
    # A frozen decoder is not written into the checkpoint; it is recorded by its source path.
    if decoder_source is None: state["decoder"] = _mod(decoder).state_dict()
    else: state["decoder_source"] = decoder_source
    torch.save(state, os.path.join(ckpt_dir, f"ae_{step}.pt"))
    torch.save(state, os.path.join(ckpt_dir, "ae_latest.pt"))
    cleanup_checkpoints(logger, ckpt_dir)

@torch.no_grad()
def init_encoder_ln(encoder, mean, std):
    """The encoder's final LayerNorm affine starts at the official per-channel latent std / mean, so a fresh
    encoder emits latents in the range the official decoder expects instead of ~unit scale."""
    encoder.norm.norm.weight.copy_(std.reshape(-1)); encoder.norm.norm.bias.copy_(mean.reshape(-1))

def load_init_encoder(path, encoder, device):
    # .pt checkpoint ({'encoder': ...}) or a BlueCodec .safetensors ('encoder.*' keys, e.g. HF model.safetensors)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = {k[len("encoder."):]: v for k, v in load_file(path, device=str(device)).items() if k.startswith("encoder.")}
    else:
        sd = torch.load(path, map_location=device)["encoder"]
    _mod(encoder).load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)

def cleanup_checkpoints(logger, ckpt_dir="checkpoints/ae"):
    try:
        ckpts = []
        for f in os.listdir(ckpt_dir):
            if f.startswith("ae_") and f.endswith(".pt") and f != "ae_latest.pt":
                try: ckpts.append((int(f.replace("ae_", "").replace(".pt", "")), f))
                except ValueError: pass
        ckpts.sort(key=lambda x: x[0], reverse=True)
        for _, old in ckpts[1000:]:
            path = os.path.join(ckpt_dir, old)
            if os.path.exists(path): os.remove(path); logger.info(f"Deleted: {old}")
    except Exception as e: logger.warning(f"Cleanup error: {e}")

def start_tensorboard(logdir="checkpoints/ae/logs"):
    try:
        tb_path = os.path.join(os.path.dirname(sys.executable), "tensorboard")
        if not os.path.exists(tb_path): tb_path = "tensorboard"
        proc = subprocess.Popen([tb_path, "--logdir", logdir, "--port", "8000", "--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(proc.kill)
        print(f"TensorBoard at http://localhost:8000")
    except Exception as e: print(f"TB error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Training checkpoint (.pt) to resume from')
    parser.add_argument('--eval_input', type=str, default=None, help='Audio file reconstructed into <checkpoint_dir>/eval at every save')
    parser.add_argument('--arch_config', type=str, default='config/tts.json', help='Model and data config')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--finetune', action='store_true', help='With --resume: load the weights only (fresh optimizer and schedule, step 0)')
    parser.add_argument('--lr', type=float, default=None, help='Peak learning rate (overrides ae.train.lr)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/ae', help='Checkpoints, logs and eval audio')
    parser.add_argument('--total_steps', type=int, default=1500000, help='Training stops here (and cosine T_max)')
    parser.add_argument('--batch_size', type=int, default=None, help='Per-process batch size (overrides ae.train.batch_size)')
    # --- Encoder-only training against a frozen decoder (stage 1 of the two-stage recipe) ---
    parser.add_argument('--encoder_only', action='store_true', help='Freeze the decoder (from --decoder) and train the encoder + discriminators')
    parser.add_argument('--decoder', type=str, default=None, help='Frozen decoder for --encoder_only: an AE .pt checkpoint, or "official" (Supertonic-3 vocoder)')
    parser.add_argument('--official_onnx', type=str, default=None, help='Local official vocoder.onnx (default: downloaded at a pinned revision)')
    parser.add_argument('--init_encoder', type=str, default=None, help='Encoder init: AE .pt checkpoint or BlueCodec .safetensors (fresh optimizer, step 0)')
    parser.add_argument('--d_warmup', type=int, default=0, help='Discriminators frozen, reconstruction-only loss for the first N steps')
    # --- Stage 2: the decoder becomes trainable ---
    parser.add_argument('--stage2', action='store_true', help='Resume an --encoder_only checkpoint (--resume) with the decoder trainable')
    parser.add_argument('--val_data', type=str, default=None, help='Held-out audio for lr_schedule "plateau" (default ae.data.val_metadata)')
    # --- Paper losses (defaults reproduce the 1.5M-step recipe) ---
    parser.add_argument('--recon_logmel_fullband', action='store_true', help='Reconstruction on LOG mel up to sr/2 over the whole segment')
    parser.add_argument('--fm_composite', action='store_true', help='Feature matching averaged over MPD+MRD layers together (paper Eq. 6)')
    # --- E12b additions (all off by default) ---
    parser.add_argument('--edge_aware', action='store_true',
                        help='Half of the crops end at the clip end; batches padded to a multiple of 3072 samples and cut so clips end '
                             'in the last chunk; loss masked per clip beyond ceil(L/3072)*3072')
    parser.add_argument('--tail_consistency', type=float, default=0.0,
                        help='Weight of the last-latent-frame consistency term (needs --edge_aware and --tail_stats; freezes the encoder BatchNorm)')
    parser.add_argument('--tail_stats', type=str, default=None, help='Latent stats file (mean/std over 144 compressed channels) normalising the tail term')
    parser.add_argument('--no_bn_freeze', action='store_true', help='Keep the encoder BatchNorm in train mode with --tail_consistency')
    parser.add_argument('--hiband_w', type=float, default=0.0, help='Weight of the 12-22 kHz log-magnitude STFT L1')
    parser.add_argument('--floor_w', type=float, default=0.0, help='Weight of the silence-floor term (source frames below --floor_db must decode below it)')
    parser.add_argument('--floor_db', type=float, default=-60.0, help='Silence floor in dBFS for --floor_w')
    parser.add_argument('--drop_list', type=str, default=None, help='Text file of audio paths (one per line) to leave out')
    parser.add_argument('--decoder_pad_mode', choices=['zeros', 'replicate'], default=None,
                        help='Causal padding of the decoder (default: the config, zeros; E12b uses replicate)')
    parser.add_argument('--continue_cosine', type=int, default=None,
                        help='With --resume: keep the step counter and optimizer state, restart the cosine from --lr over N steps')
    parser.add_argument('--lr_warmup', type=int, default=0, help='With --continue_cosine: linear warm-up steps from 1%% of --lr')
    args = parser.parse_args()
    if args.encoder_only and not args.decoder:
        raise SystemExit("--encoder_only needs --decoder (an AE .pt checkpoint or 'official')")
    if args.stage2 and (args.encoder_only or not args.resume):
        raise SystemExit("--stage2 continues a checkpoint: it needs --resume and excludes --encoder_only")
    if args.tail_consistency > 0 and not (args.edge_aware and args.tail_stats):
        raise SystemExit("--tail_consistency needs --edge_aware and --tail_stats")
    if args.continue_cosine and not args.resume:
        raise SystemExit("--continue_cosine needs --resume")
    if args.lr_warmup and not args.continue_cosine:
        raise SystemExit("--lr_warmup needs --continue_cosine")
    freeze_bn = args.tail_consistency > 0 and not args.no_bn_freeze
    ckpt_dir = args.checkpoint_dir

    if 'WORLD_SIZE' in os.environ: args.local_rank = int(os.environ['LOCAL_RANK'])
    device, ddp_ids = init_distributed(args.local_rank)
    set_seed(args.seed + args.local_rank)
    logger = setup_logger(ckpt_dir, args.local_rank)
    
    writer = None
    if args.local_rank == 0:
        logger.info(f"Training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
        start_tensorboard(os.path.join(ckpt_dir, "logs"))
        writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, "logs"))

    with open(args.arch_config, "r") as f: arch_cfg = json.load(f)
    ae_cfg = arch_cfg['ae']
    data_cfg, train_cfg = ae_cfg['data'], ae_cfg['train']
    # Recipe switches live in ae.train (config/tts_st3.json); every missing key keeps the default recipe.
    recon = "logmel_fullband" if args.recon_logmel_fullband else train_cfg.get('recon', 'linmel_12k')
    fm_composite = args.fm_composite or train_cfg.get('fm_composite', False)
    ema_decay = float(train_cfg.get('ema', 0.0))
    no_decay_norms = train_cfg.get('no_decay_norms', False)
    clip_generator = train_cfg.get('clip_generator', True)
    lr_schedule = train_cfg.get('lr_schedule', 'cosine')
    dec_lr_scale = float(train_cfg.get('dec_lr_scale', 1.0))
    if args.stage2 and lr_schedule != 'plateau':
        raise SystemExit("--stage2 needs ae.train.lr_schedule 'plateau' (it ramps the decoder's lr from the switch step)")
    if args.continue_cosine and lr_schedule != 'cosine':
        raise SystemExit("--continue_cosine needs ae.train.lr_schedule 'cosine'")
    if not train_cfg.get('tf32', True):
        # Strict fp32: cuDNN convolutions run in TF32 (~10 mantissa bits) by default on Ampere and newer GPUs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    dataset = TTSDataset(data_cfg['train_metadata'], sample_rate=data_cfg['sample_rate'], segment_size=data_cfg.get('segment_size'),
                         end_crop_prob=0.5 if args.edge_aware else 0.0, drop_list=args.drop_list)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size or train_cfg['batch_size'], sampler=sampler, num_workers=train_cfg['num_workers'],
                            collate_fn=collate_fn_edge if args.edge_aware else collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    decoder_cfg = dict(ae_cfg['decoder'])
    if args.decoder_pad_mode: decoder_cfg['pad_mode'] = args.decoder_pad_mode
    ckpt = torch.load(args.resume, map_location='cpu') if args.resume else None

    encoder = LatentEncoder(cfg=ae_cfg['encoder']).to(device)
    # Frozen decoder (stage 1) or the one stage 1 used (stage 2: its checkpoints record only where it came from).
    dec_src = args.decoder if args.encoder_only else None
    if args.stage2 and 'decoder' not in ckpt: dec_src = ckpt.get('decoder_source')
    enc_ln_init = train_cfg.get('enc_ln_init') == 'official' and not args.resume and not args.init_encoder
    # official vocoder.onnx, parsed once: (decoder state, latent mean, latent std, normalizer scale)
    official = supertonic3_decoder_state(args.official_onnx) if dec_src == "official" or enc_ln_init else None
    decoder = LatentDecoder1D(cfg=decoder_cfg).to(device)
    if dec_src == "official":
        if decoder_cfg.get('pad_mode') != 'replicate':
            raise SystemExit("the official decoder needs ae.decoder.pad_mode 'replicate' (see config/tts_st3.json)")
        decoder.load_state_dict(official[0], strict=True)
    elif dec_src:
        decoder.load_state_dict(torch.load(dec_src, map_location=device)['decoder'])
    if args.stage2 and dec_src and args.local_rank == 0:
        logger.info(f"Stage 2: decoder from {dec_src}, trainable at lr x{dec_lr_scale}")
    decoder_source = None
    if args.encoder_only:
        # frozen, so not DDP-wrapped: it has nothing to synchronise
        decoder_source = args.decoder
        decoder.requires_grad_(False); decoder.eval()
    # Stage 2 keeps the decoder's BatchNorm statistics: the official decoder's final BatchNorm re-estimated on a
    # new encoder's latents no longer matches the weights after it, and reconstruction breaks.
    dec_bns = [m for m in decoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)] if args.stage2 else []

    if enc_ln_init:
        init_encoder_ln(encoder, official[1], official[2])
        if args.local_rank == 0: logger.info("Encoder final LayerNorm initialised to the official latent std / mean")
    if train_cfg.get('sync_bn', False) and dist.get_world_size() > 1:
        # per-GPU BatchNorm statistics over a small per-GPU batch are noisy enough to collapse the encoder stem
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder = DDP(encoder, device_ids=ddp_ids)
    if not args.encoder_only: decoder = DDP(decoder, device_ids=ddp_ids)
    mpd = DDP(MultiPeriodDiscriminator().to(device), device_ids=ddp_ids)
    mrd = DDP(MultiResolutionDiscriminator().to(device), device_ids=ddp_ids)

    spec_cfg = ae_cfg['encoder'].get('spec_processor', {})
    mel_transform_input = LinearMelSpectrogram(sample_rate=spec_cfg.get('sample_rate', data_cfg['sample_rate']), n_fft=spec_cfg.get('n_fft', 2048), hop_length=spec_cfg.get('hop_length', 512), win_length=spec_cfg.get('win_length', 2048), n_mels=spec_cfg.get('n_mels', 1253)).to(device)
    mel_transforms_loss = get_mel_transforms(data_cfg, device, recon)
    
    lr = args.lr if args.lr else float(train_cfg['lr'])
    betas = tuple(train_cfg.get('betas', (0.8, 0.99)))
    wd = float(train_cfg.get('weight_decay', 0.01))
    enc_named = list(_mod(encoder).named_parameters())
    if args.encoder_only or args.stage2: g_groups = param_groups(enc_named, lr, no_decay_norms)
    else: g_groups = param_groups(enc_named + list(_mod(decoder).named_parameters()), lr, no_decay_norms)
    # stage 2: the decoder in groups of its own, after the encoder's, so a stage-1 optimizer state still loads
    dec_groups = param_groups(_mod(decoder).named_parameters(), lr * dec_lr_scale, no_decay_norms) if args.stage2 else []
    n_enc_groups = len(g_groups)
    opt_g = torch.optim.AdamW(g_groups, lr=lr, betas=betas, weight_decay=wd)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=lr, betas=betas, weight_decay=wd)
    if lr_schedule == 'plateau':
        lr_sched = PlateauLR([opt_g, opt_d], lr, train_cfg.get('warmup_steps', 0), float(train_cfg.get('min_lr', 1e-5)),
                             train_cfg.get('plateau_patience', 3), train_cfg.get('plateau_threshold', 0.01), train_cfg.get('dec_warmup_steps', 0))
        schedulers = {'lr_plateau': lr_sched}
    else:
        schedulers, lr_sched = cosine_schedulers(opt_g, opt_d, args.total_steps)
    
    step, epoch = load_checkpoint(args, ckpt, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, dec_groups, logger)
    if args.init_encoder and not args.resume:
        load_init_encoder(args.init_encoder, encoder, device)
        if args.local_rank == 0: logger.info(f"Encoder initialised from {args.init_encoder}")
    if args.continue_cosine:
        for opt in (opt_g, opt_d):
            for g in opt.param_groups: g['lr'] = g['initial_lr'] = lr
        schedulers, lr_sched = cosine_schedulers(opt_g, opt_d, args.continue_cosine, args.lr_warmup)
        if args.local_rank == 0: logger.info(f"New schedule from step {step}: {args.lr_warmup} warm-up steps, cosine over {args.continue_cosine} steps from lr {lr}")
    if lr_schedule == 'plateau':
        # The step the decoder became trainable (its lr ramps from there); kept across stage-2 resumes.
        dec_from = ckpt.get('dec_trainable_from', step) if args.stage2 else None   # only stage-2 checkpoints record it
        for i, g in enumerate(opt_g.param_groups):
            g['base_lr'], g['ramp_from'] = (lr, None) if i < n_enc_groups else (lr * dec_lr_scale, dec_from)
        for g in opt_d.param_groups: g['base_lr'], g['ramp_from'] = lr, None
        val_src = args.val_data or data_cfg.get('val_metadata')
        if not val_src: raise SystemExit("lr_schedule 'plateau' needs held-out audio: --val_data or ae.data.val_metadata")
        val_ds = TTSDataset(val_src, sample_rate=data_cfg['sample_rate'])   # not cropped: first segment_size samples below
        val_clips = [val_ds[i][:data_cfg.get('segment_size')] for i in range(min(len(val_ds), train_cfg.get('val_clips', 64)))]
        eval_interval = train_cfg.get('eval_interval', 10000)
    terms = ExtraTerms(tail_w=args.tail_consistency, hiband_w=args.hiband_w, floor_w=args.floor_w, floor_db=args.floor_db)
    if args.tail_consistency > 0:
        stats = torch.load(args.tail_stats, map_location='cpu')
        terms.tail_norm = (stats['mean'].view(1, -1, 1).to(device), stats['std'].view(1, -1, 1).to(device))

    # Generator EMA (ema > 0): what the held-out check and the eval render use. A frozen decoder is its own EMA.
    enc_ema = dec_ema = None
    eval_enc = lambda: enc_ema if enc_ema is not None else encoder
    eval_dec = lambda: dec_ema if dec_ema is not None else decoder
    if ema_decay > 0:
        enc_ema = copy.deepcopy(_mod(encoder)).eval().requires_grad_(False)
        dec_ema = decoder if args.encoder_only else copy.deepcopy(_mod(decoder)).eval().requires_grad_(False)
        if step > 0:
            if 'encoder_ema' in ckpt: enc_ema.load_state_dict(ckpt['encoder_ema'])
            if 'decoder_ema' in ckpt and not args.encoder_only: dec_ema.load_state_dict(ckpt['decoder_ema'])
    # What must never move: the whole frozen decoder (stage 1), or the decoder's BatchNorm running statistics
    # (stage 2; the affine trains). Taken after load_checkpoint, which may load the stage-2 decoder.
    def frozen_state():
        if args.encoder_only: return decoder.state_dict()
        return {f"{i}.{k}": v for i, m in enumerate(dec_bns) for k, v in m.named_buffers()}
    frozen_ref = {k: v.detach().cpu().clone() for k, v in frozen_state().items()}

    def set_train_modes():
        encoder.train()
        if freeze_bn: freeze_batchnorm(encoder)
        if args.encoder_only: decoder.eval(); return
        decoder.train()
        for m in dec_bns: m.eval()
    set_train_modes()

    def save(step):
        assert all(torch.equal(v.detach().cpu(), frozen_ref[k]) for k, v in frozen_state().items()), "frozen decoder state changed"
        extra = {"ae_config": ae_cfg, "stage": 1 if args.encoder_only else 2 if args.stage2 else 0}
        if args.stage2: extra["dec_trainable_from"] = dec_from
        if enc_ema is not None:
            extra["encoder_ema"] = enc_ema.state_dict()
            if not args.encoder_only: extra["decoder_ema"] = dec_ema.state_dict()
        save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, schedulers, logger, ckpt_dir, decoder_source, extra)

    crop_len = int(data_cfg['sample_rate'] * 0.19)
    g_meter, d_meter, mel_meter = AverageMeter(), AverageMeter(), AverageMeter()
    last_saved = None
    
    while step < args.total_steps:
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True) if args.local_rank == 0 else dataloader
        for batch in pbar:
            if step >= args.total_steps: break
            if lr_schedule == 'plateau': lr_sched.step(step)
            result = train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
                                update_discriminator=(step > args.d_warmup), recon_only=(0 < args.d_warmup and step <= args.d_warmup),
                                recon_full_segment=(recon != 'linmel_12k'),
                                fm_composite=fm_composite, encoder_only=args.encoder_only, terms=terms, clip_generator=clip_generator)
            if result is None: break
            loss_g, loss_d, loss_mel, extras = result
            if lr_schedule != 'plateau':
                for s in schedulers.values(): s.step()
            if enc_ema is not None:
                ema_update(enc_ema, _mod(encoder), ema_decay)
                if not args.encoder_only: ema_update(dec_ema, _mod(decoder), ema_decay)

            if lr_schedule == 'plateau' and step > 0 and step % eval_interval == 0:
                # rank 0 scores, every rank applies the same decision
                val = torch.zeros(1, device=device)
                if args.local_rank == 0:
                    val[0] = heldout_recon(_mod(eval_enc()), _mod(eval_dec()), mel_transform_input, mel_transforms_loss, val_clips, device)
                    set_train_modes()
                dist.broadcast(val, 0)
                halved = lr_sched.report(val.item())
                if args.local_rank == 0:
                    writer.add_scalar("Loss/Heldout_Recon", val.item(), step)
                    logger.info(f"step {step}: held-out recon {val.item():.4f} (best {lr_sched.best:.4f})" + (f", lr halved to {lr * lr_sched.scale:.2e}" if halved else ""))
            
            if args.local_rank == 0:
                g_meter.update(loss_g); d_meter.update(loss_d); mel_meter.update(loss_mel)
                if step % 10 == 0:
                    writer.add_scalar("Loss/Generator", loss_g, step); writer.add_scalar("Loss/Discriminator", loss_d, step); writer.add_scalar("Loss/Mel", loss_mel, step); writer.add_scalar("Training/LR", lr_sched.get_last_lr()[0], step)
                    for name, value in extras.items(): writer.add_scalar(f"Loss/{name}", value, step)
                pbar.set_postfix({"Step": step, "G": f"{g_meter.avg:.4f}", "D": f"{d_meter.avg:.4f}", "Mel": f"{mel_meter.avg:.4f}", "LR": f"{lr_sched.get_last_lr()[0]:.2e}"})
                if step % train_cfg['save_interval'] == 0:
                    save(step); last_saved = step
                    if args.eval_input:
                        evaluate(eval_enc(), eval_dec(), mel_transform_input, args.eval_input, os.path.join(ckpt_dir, "eval"), step, device, data_cfg['sample_rate'], args.local_rank, keep_decoder_eval=args.encoder_only)
                        set_train_modes()   # evaluate() leaves the encoder (and a trainable decoder) in train mode, BatchNorm included
            step += 1
        epoch += 1
    if args.local_rank == 0 and step >= args.total_steps and last_saved != step - 1:
        save(step - 1)   # the final weights, whatever save_interval is
    dist.destroy_process_group()

if __name__ == "__main__": main()
