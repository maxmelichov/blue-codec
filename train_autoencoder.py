import sys
import os
import json
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
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
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

def get_mel_transforms(data_cfg, device, logmel_fullband=False):
    mel_configs = [(1024, 256, 1024, 64), (2048, 512, 2048, 128), (4096, 1024, 4096, 128)]
    if logmel_fullband:
        return [LogMelFullBand(data_cfg['sample_rate'], n_fft, hop, win, n_mels).to(device)
                for (n_fft, hop, win, n_mels) in mel_configs]
    return [MelSpectrogramNoLog(data_cfg['sample_rate'], n_fft, hop, win, n_mels).to(device)
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

def train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
               update_discriminator=True, lambda_recon=45.0, recon_full_segment=False, fm_composite=False, encoder_only=False, recon_only=False,
               terms=None):
    # recon_only: no adversarial or feature-matching terms for the generator.
    # encoder_only: the frozen decoder stays in the graph, so gradients reach the encoder through it.
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
    L_recon = sum(F.l1_loss(tf(rec_real.squeeze(1)), tf(rec_fake.squeeze(1))) for tf in mel_transforms_loss)
    L_recon = L_recon / len(mel_transforms_loss) if mel_transforms_loss else L_recon

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
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
    if not encoder_only: torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
    opt_g.step()
    return loss_g_total.item(), loss_d_total, L_recon.item(), {k: loss.item() for k, (_, loss) in extras.items()}

def freeze_batchnorm(model):
    """BatchNorm layers in eval mode (running statistics used and not updated)."""
    for m in _mod(model).modules():
        if isinstance(m, nn.BatchNorm1d): m.eval()

def load_frozen_decoder(path, cfg, device):
    decoder = LatentDecoder1D(cfg=cfg).to(device)
    decoder.load_state_dict(torch.load(path, map_location=device)['decoder'])
    decoder.requires_grad_(False); decoder.eval()
    return decoder

def make_scheduler(opt, t_max, warmup=0):
    """Cosine decay to 1e-6 over `t_max` steps, after an optional linear warm-up from 1% of the lr."""
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)
    if not warmup: return cosine
    linear = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(opt, [linear, cosine], milestones=[warmup])

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

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def load_checkpoint(args, encoder, decoder, mpd, mrd, opt_g, opt_d, scheduler_g, scheduler_d, logger, device):
    if not args.resume: return 0, 0
    if args.local_rank == 0: logger.info(f"Resuming from {args.resume}...")
    ckpt = torch.load(args.resume, map_location=device)
    
    def load_sd(model, sd):
        if any(k.startswith('module.') for k in sd.keys()): sd = {k.replace('module.', ''): v for k, v in sd.items()}
        md = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        if hasattr(model, 'module'): model.module.load_state_dict(filtered, strict=False)
        else: model.load_state_dict(filtered, strict=False)

    load_sd(encoder, ckpt['encoder'])
    if not args.encoder_only and 'decoder' in ckpt: load_sd(decoder, ckpt['decoder'])   # encoder_only: decoder comes from --decoder
    load_sd(mpd, ckpt['mpd']); load_sd(mrd, ckpt['mrd'])
    
    if not args.finetune:
        def opt_match(opt, sd):
            return sd and len(opt.param_groups) == len(sd['param_groups']) and all(len(g['params']) == len(sg['params']) for g, sg in zip(opt.param_groups, sd['param_groups']))
        if opt_match(opt_g, ckpt.get('opt_g')) and opt_match(opt_d, ckpt.get('opt_d')):
            opt_g.load_state_dict(ckpt['opt_g']); opt_d.load_state_dict(ckpt['opt_d'])
            if 'scheduler_g' in ckpt: scheduler_g.load_state_dict(ckpt['scheduler_g'])
            if 'scheduler_d' in ckpt: scheduler_d.load_state_dict(ckpt['scheduler_d'])
            return ckpt['step'] + 1, ckpt.get('epoch', 0)
        if args.local_rank == 0: logger.warning("Optimizer state does not match the model: fresh optimizer, step 0")
    return 0, 0

def save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, scheduler_g, scheduler_d, logger, ckpt_dir="checkpoints/ae", decoder_source=None):
    state = {
        "step": step, "epoch": epoch,
        "encoder": _mod(encoder).state_dict(),
        "mpd": _mod(mpd).state_dict(), "mrd": _mod(mrd).state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(), "scheduler_d": scheduler_d.state_dict(),
    }
    # A frozen decoder is not written into the checkpoint; it is recorded by its source path.
    if decoder_source is None: state["decoder"] = _mod(decoder).state_dict()
    else: state["decoder_source"] = decoder_source
    torch.save(state, os.path.join(ckpt_dir, f"ae_{step}.pt"))
    torch.save(state, os.path.join(ckpt_dir, "ae_latest.pt"))
    cleanup_checkpoints(logger, ckpt_dir)

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
    parser.add_argument('--total_steps', type=int, default=1500000, help='Loop length and cosine T_max')
    parser.add_argument('--batch_size', type=int, default=None, help='Per-process batch size (overrides ae.train.batch_size)')
    # --- Encoder-only training against a frozen decoder ---
    parser.add_argument('--encoder_only', action='store_true', help='Freeze the decoder (from --decoder) and train the encoder + discriminators')
    parser.add_argument('--decoder', type=str, default=None, help='AE .pt checkpoint whose decoder is frozen (with --encoder_only)')
    parser.add_argument('--init_encoder', type=str, default=None, help='Encoder init: AE .pt checkpoint or BlueCodec .safetensors (fresh optimizer, step 0)')
    parser.add_argument('--d_warmup', type=int, default=0, help='Discriminators frozen, reconstruction-only loss for the first N steps')
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
        raise SystemExit("--encoder_only needs --decoder (an AE .pt checkpoint)")
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
    
    dataset = TTSDataset(data_cfg['train_metadata'], sample_rate=data_cfg['sample_rate'], segment_size=data_cfg.get('segment_size'),
                         end_crop_prob=0.5 if args.edge_aware else 0.0, drop_list=args.drop_list)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size or train_cfg['batch_size'], sampler=sampler, num_workers=train_cfg['num_workers'],
                            collate_fn=collate_fn_edge if args.edge_aware else collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    decoder_cfg = dict(ae_cfg['decoder'])
    if args.decoder_pad_mode: decoder_cfg['pad_mode'] = args.decoder_pad_mode
    
    encoder = DDP(LatentEncoder(cfg=ae_cfg['encoder']).to(device), device_ids=ddp_ids)
    decoder_source = None
    if args.encoder_only:
        # frozen, so not DDP-wrapped: it has nothing to synchronise
        decoder = load_frozen_decoder(args.decoder, decoder_cfg, device)
        decoder_source = args.decoder
        dec_ref_sd = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    else:
        decoder = DDP(LatentDecoder1D(cfg=decoder_cfg).to(device), device_ids=ddp_ids)
    mpd = DDP(MultiPeriodDiscriminator().to(device), device_ids=ddp_ids)
    mrd = DDP(MultiResolutionDiscriminator().to(device), device_ids=ddp_ids)

    spec_cfg = ae_cfg['encoder'].get('spec_processor', {})
    mel_transform_input = LinearMelSpectrogram(sample_rate=spec_cfg.get('sample_rate', data_cfg['sample_rate']), n_fft=spec_cfg.get('n_fft', 2048), hop_length=spec_cfg.get('hop_length', 512), win_length=spec_cfg.get('win_length', 2048), n_mels=spec_cfg.get('n_mels', 1253)).to(device)
    mel_transforms_loss = get_mel_transforms(data_cfg, device, logmel_fullband=args.recon_logmel_fullband)
    
    lr = args.lr if args.lr else float(train_cfg['lr'])
    g_params = list(encoder.parameters()) if args.encoder_only else list(encoder.parameters()) + list(decoder.parameters())
    opt_g = torch.optim.AdamW(g_params, lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=lr, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler_g = make_scheduler(opt_g, args.total_steps)
    scheduler_d = make_scheduler(opt_d, args.total_steps)
    
    step, epoch = load_checkpoint(args, encoder, decoder, mpd, mrd, opt_g, opt_d, scheduler_g, scheduler_d, logger, device)
    if args.init_encoder and not args.resume:
        load_init_encoder(args.init_encoder, encoder, device)
        if args.local_rank == 0: logger.info(f"Encoder initialised from {args.init_encoder}")
    if args.continue_cosine:
        for opt in (opt_g, opt_d):
            for g in opt.param_groups: g['lr'] = g['initial_lr'] = lr
        scheduler_g = make_scheduler(opt_g, args.continue_cosine, args.lr_warmup)
        scheduler_d = make_scheduler(opt_d, args.continue_cosine, args.lr_warmup)
        if args.local_rank == 0: logger.info(f"New schedule from step {step}: {args.lr_warmup} warm-up steps, cosine over {args.continue_cosine} steps from lr {lr}")
    terms = ExtraTerms(tail_w=args.tail_consistency, hiband_w=args.hiband_w, floor_w=args.floor_w, floor_db=args.floor_db)
    if args.tail_consistency > 0:
        stats = torch.load(args.tail_stats, map_location='cpu')
        terms.tail_norm = (stats['mean'].view(1, -1, 1).to(device), stats['std'].view(1, -1, 1).to(device))
    if freeze_bn: freeze_batchnorm(encoder)
    crop_len = int(data_cfg['sample_rate'] * 0.19)
    g_meter, d_meter, mel_meter = AverageMeter(), AverageMeter(), AverageMeter()
    
    while step < args.total_steps:
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True) if args.local_rank == 0 else dataloader
        for batch in pbar:
            if step >= args.total_steps: break
            result = train_step(batch, encoder, decoder, mpd, mrd, mel_transform_input, mel_transforms_loss, opt_g, opt_d, device, crop_len, logger,
                                update_discriminator=(step > args.d_warmup), recon_only=(0 < args.d_warmup and step <= args.d_warmup),
                                recon_full_segment=args.recon_logmel_fullband,
                                fm_composite=args.fm_composite, encoder_only=args.encoder_only, terms=terms)
            if result is None: break
            loss_g, loss_d, loss_mel, extras = result
            scheduler_g.step(); scheduler_d.step()
            
            if args.local_rank == 0:
                g_meter.update(loss_g); d_meter.update(loss_d); mel_meter.update(loss_mel)
                if step % 10 == 0:
                    writer.add_scalar("Loss/Generator", loss_g, step); writer.add_scalar("Loss/Discriminator", loss_d, step); writer.add_scalar("Loss/Mel", loss_mel, step); writer.add_scalar("Training/LR", scheduler_g.get_last_lr()[0], step)
                    for name, value in extras.items(): writer.add_scalar(f"Loss/{name}", value, step)
                pbar.set_postfix({"Step": step, "G": f"{g_meter.avg:.4f}", "D": f"{d_meter.avg:.4f}", "Mel": f"{mel_meter.avg:.4f}", "LR": f"{scheduler_g.get_last_lr()[0]:.2e}"})
                if step % train_cfg['save_interval'] == 0:
                    if args.encoder_only:   # the frozen decoder must never move
                        assert all(torch.equal(v.detach().cpu(), dec_ref_sd[k]) for k, v in decoder.state_dict().items()), "frozen decoder changed"
                    save_checkpoint(step, epoch, encoder, decoder, mpd, mrd, opt_g, opt_d, scheduler_g, scheduler_d, logger, ckpt_dir, decoder_source)
                    if args.eval_input: evaluate(encoder, decoder, mel_transform_input, args.eval_input, os.path.join(ckpt_dir, "eval"), step, device, data_cfg['sample_rate'], args.local_rank, keep_decoder_eval=args.encoder_only)
                    if freeze_bn: freeze_batchnorm(encoder)   # evaluate() puts the encoder back in train mode
            step += 1
        epoch += 1

if __name__ == "__main__": main()
