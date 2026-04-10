import sys
import os
import json
import subprocess
import atexit
import argparse
import random
import logging

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

from dataset import TTSDataset, collate_fn
from bluecodec.audio_utils import ensure_sr
from bluecodec.autoencoder.latent_encoder import LatentEncoder
from bluecodec.autoencoder.latent_decoder import LatentDecoder1D
from bluecodec.autoencoder.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bluecodec.utils import MelSpectrogramNoLog, LinearMelSpectrogram

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
    if count == 0:
        return torch.tensor(0.0, device=fmap_r[0][0].device)
    return loss / count 

def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        loss += l
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((dr - 1.0) ** 2)
        g_loss = torch.mean((dg + 1.0) ** 2)
        loss += (r_loss + g_loss)
    return loss

def train_step(
    batch,
    encoder,
    decoder,
    mpd,
    mrd,
    mel_transform_input,
    mel_transforms_loss,
    opt_g,
    opt_d,
    device,
    crop_len,
    logger,
    update_discriminator=True
):
    lambda_recon = 45.0
    lambda_adv = 1.0
    lambda_fm = 0.1

    audio = batch.to(device)
    if audio.dim() == 2:
        audio = audio.unsqueeze(1) 

    with torch.no_grad():
        mel = mel_transform_input(audio.squeeze(1))
    
    z = encoder(mel)
    y_hat = decoder(z) 
    
    if y_hat.dim() == 2:
        y_hat = y_hat.unsqueeze(1)
    
    if y_hat.shape[-1] != audio.shape[-1]:
        min_len = min(y_hat.shape[-1], audio.shape[-1])
        y_hat = y_hat[..., :min_len]
        audio = audio[..., :min_len]

    if audio.shape[-1] > crop_len:
        start_idx = torch.randint(0, audio.shape[-1] - crop_len + 1, (1,)).item()
        audio_crop = audio[..., start_idx : start_idx + crop_len]
        y_hat_crop = y_hat[..., start_idx : start_idx + crop_len]
    else:
        audio_crop = audio
        y_hat_crop = y_hat

    loss_d_total = 0.0
    if update_discriminator:
        y_hat_detached = y_hat_crop.detach()
        y_df_hat_r, y_df_hat_g, _, _ = mpd(audio_crop, y_hat_detached)
        loss_d_mpd = discriminator_loss(y_df_hat_r, y_df_hat_g)
        y_ds_hat_r, y_ds_hat_g, _, _ = mrd(audio_crop, y_hat_detached)
        loss_d_mrd = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_d_total = loss_d_mpd + loss_d_mrd
        
        if check_for_nan_inf(loss_d_total, "Discriminator Loss", logger):
            return None, None, None

        opt_d.zero_grad()
        loss_d_total.backward()
        torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1.0)
        opt_d.step()
        loss_d_total = loss_d_total.item()

    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio_crop, y_hat_crop)
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(audio_crop, y_hat_crop)
    
    loss_gen_mpd = generator_loss(y_df_hat_g)
    loss_gen_mrd = generator_loss(y_ds_hat_g)
    L_adv = loss_gen_mpd + loss_gen_mrd

    loss_fm_mpd = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_mrd = feature_loss(fmap_s_r, fmap_s_g)
    L_fm = loss_fm_mpd + loss_fm_mrd
    
    L_recon = 0.0
    for mel_tf in mel_transforms_loss:
        m_real = mel_tf(audio_crop.squeeze(1))
        m_fake = mel_tf(y_hat_crop.squeeze(1))
        L_recon += F.l1_loss(m_real, m_fake)

    L_recon = L_recon / len(mel_transforms_loss) if len(mel_transforms_loss) > 0 else L_recon
    loss_g_total = (lambda_recon * L_recon) + (lambda_adv * L_adv) + (lambda_fm * L_fm)
    
    if check_for_nan_inf(loss_g_total, "Generator Loss", logger):
        return None, None, None

    opt_g.zero_grad()
    loss_g_total.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
    opt_g.step()
    
    return loss_g_total.item(), loss_d_total, L_recon.item()

def evaluate(encoder, decoder, mel_transform, input_wav_path, output_dir, step, device, target_sr, rank):
    if rank != 0: return
    
    print(f"Evaluating on {input_wav_path} at step {step}...")
    encoder.eval()
    decoder.eval()
    
    try:
        wav, sr = torchaudio.load(input_wav_path)
        if sr != target_sr:
            wav = ensure_sr(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        wav = wav.to(device)
        
        with torch.no_grad():
            mel = mel_transform(wav)
            enc_module = encoder.module if isinstance(encoder, DDP) else encoder
            dec_module = decoder.module if isinstance(decoder, DDP) else decoder
            z = enc_module(mel)
            y_hat = dec_module(z)
            
            max_val = y_hat.abs().max().item()
            mean_val = y_hat.mean().item()
            print(f"  > Output Stats - Max: {max_val:.4f}, Mean: {mean_val:.4f}")
            
            y_hat_final = y_hat.squeeze()
            if max_val > 1.0:
                y_hat_final = y_hat_final / (max_val + 1e-8)
    
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input_wav_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"step_{step}_{name}_recon.wav")
        sf.write(output_path, y_hat_final.cpu().numpy(), target_sr)
        print(f"  Saved: {output_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

    encoder.train()
    decoder.train()

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_input', type=str, default=None, help='Path to audio file for evaluation')
    parser.add_argument('--arch_config', type=str, default='configs/tts.json', help='Path to AE architecture config JSON')
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for DDP')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--finetune', action='store_true', help='Finetune: reset optimizer/scheduler/step')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate for finetuning')
    args = parser.parse_args()

    if 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f'cuda:{args.local_rank}')
    
    set_seed(args.seed + args.local_rank)
    logger = setup_logger("checkpoints/ae", args.local_rank)
    writer = None
    if args.local_rank == 0:
        logger.info(f"Starting Training on {torch.cuda.get_device_name(device)}")
        try:
            tensorboard_path = os.path.join(os.path.dirname(sys.executable), "tensorboard")
            if not os.path.exists(tensorboard_path):
                tensorboard_path = "tensorboard"
            tb_process = subprocess.Popen(
                [tensorboard_path, "--logdir", "checkpoints/ae/logs", "--port", "8000", "--bind_all"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            atexit.register(tb_process.kill)
            print(f"TensorBoard running at http://localhost:8000")
        except Exception as e:
            print(f"Failed to auto-start TensorBoard: {e}")
        writer = SummaryWriter(log_dir="checkpoints/ae/logs")

    with open(args.arch_config, "r") as f:
        arch_cfg = json.load(f)
    
    ae_cfg = arch_cfg['ae']
    encoder_arch = ae_cfg['encoder']
    decoder_arch = ae_cfg['decoder']
    data_cfg = ae_cfg['data']
    train_cfg = ae_cfg['train']
    
    dataset = TTSDataset(
        data_cfg['train_metadata'], 
        sample_rate=data_cfg['sample_rate'],
        segment_size=data_cfg.get('segment_size', None)
    )
    
    train_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size=train_cfg['batch_size'], shuffle=False, 
        sampler=train_sampler, num_workers=train_cfg['num_workers'],
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    
    encoder = LatentEncoder(cfg=encoder_arch).to(device)
    decoder = LatentDecoder1D(cfg=decoder_arch).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)
    
    encoder = DDP(encoder, device_ids=[args.local_rank])
    decoder = DDP(decoder, device_ids=[args.local_rank])
    mpd = DDP(mpd, device_ids=[args.local_rank])
    mrd = DDP(mrd, device_ids=[args.local_rank])

    encoder_spec_cfg = encoder_arch.get('spec_processor', {})
    mel_transform_input = LinearMelSpectrogram(
        sample_rate=encoder_spec_cfg.get('sample_rate', data_cfg['sample_rate']),
        n_fft=encoder_spec_cfg.get('n_fft', 2048),
        hop_length=encoder_spec_cfg.get('hop_length', 512),
        win_length=encoder_spec_cfg.get('win_length', encoder_spec_cfg.get('n_fft', 2048)),
        n_mels=encoder_spec_cfg.get('n_mels', 1253)
    ).to(device)
    
    mel_configs = [(1024, 256, 1024, 64), (2048, 512, 2048, 128), (4096, 1024, 4096, 128)]
    mel_transforms_loss = [
        MelSpectrogramNoLog(data_cfg['sample_rate'], n_fft, hop, win, n_mels).to(device)
        for (n_fft, hop, win, n_mels) in mel_configs
    ]
    
    learning_rate = args.lr if args.lr is not None else float(train_cfg['lr'])
    opt_g = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), 
                              lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), 
                              lr=learning_rate, betas=(0.8, 0.99), weight_decay=0.01)
    
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=1_500_000, eta_min=1e-6)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=1_500_000, eta_min=1e-6)
    
    crop_len = int(data_cfg['sample_rate'] * 0.19)
    start_step = 0
    start_epoch = 0
    
    if args.resume:
        if args.local_rank == 0:
            logger.info(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        
        def load_state_dict_flexible(model, state_dict):
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and v.shape != model_dict[k].shape:
                    if args.local_rank == 0:
                        print(f"Skipping '{k}' due to shape mismatch (ckpt: {v.shape}, model: {model_dict[k].shape})")
                    continue
                filtered_state_dict[k] = v
            if hasattr(model, 'module'):
                model.module.load_state_dict(filtered_state_dict, strict=False)
            else:
                model.load_state_dict(filtered_state_dict, strict=False)
        
        load_state_dict_flexible(encoder, ckpt['encoder'])
        load_state_dict_flexible(decoder, ckpt['decoder'])
        load_state_dict_flexible(mpd, ckpt['mpd'])
        load_state_dict_flexible(mrd, ckpt['mrd'])
        
        if not args.finetune:
            opt_g_state = ckpt.get('opt_g', None)
            opt_d_state = ckpt.get('opt_d', None)
            
            def optimizer_state_matches(opt, state_dict):
                if not state_dict: return False
                if len(opt.param_groups) != len(state_dict['param_groups']): return False
                for group, state_group in zip(opt.param_groups, state_dict['param_groups']):
                    if len(group['params']) != len(state_group['params']): return False
                return True

            if optimizer_state_matches(opt_g, opt_g_state) and optimizer_state_matches(opt_d, opt_d_state):
                opt_g.load_state_dict(opt_g_state)
                opt_d.load_state_dict(opt_d_state)
                if 'scheduler_g' in ckpt: scheduler_g.load_state_dict(ckpt['scheduler_g'])
                if 'scheduler_d' in ckpt: scheduler_d.load_state_dict(ckpt['scheduler_d'])
                if 'epoch' in ckpt: start_epoch = ckpt['epoch']
                start_step = ckpt['step'] + 1
            else:
                if args.local_rank == 0:
                    logger.warning("Optimizer mismatch. Resetting states.")
                start_step = 0
                start_epoch = 0
        else:
            if args.local_rank == 0:
                logger.info("Finetuning mode: Reset optimizer/scheduler states and step count.")
    
    step = start_step
    epoch = start_epoch
    g_meter = AverageMeter()
    d_meter = AverageMeter()
    mel_meter = AverageMeter()
    
    encoder.train()
    decoder.train()
    mpd.train()
    mrd.train()
    
    warmup_steps = 0
    while step < 1500000:
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True) if args.local_rank == 0 else dataloader
            
        for i, batch in enumerate(progress_bar):
            if step >= 1500000: break
            update_d = (step > warmup_steps)
            loss_g, loss_d, loss_mel = train_step(
                batch, encoder, decoder, mpd, mrd, 
                mel_transform_input, mel_transforms_loss, 
                opt_g, opt_d, device, crop_len, logger, update_discriminator=update_d
            )
            
            if loss_g is None: break
            if update_d: scheduler_d.step()
            scheduler_g.step()
            
            if args.local_rank == 0:
                g_meter.update(loss_g)
                if loss_d: d_meter.update(loss_d)
                mel_meter.update(loss_mel)
                
                if step % 10 == 0:
                    writer.add_scalar("Loss/Generator", loss_g, step)
                    if loss_d: writer.add_scalar("Loss/Discriminator", loss_d, step)
                    writer.add_scalar("Loss/Mel", loss_mel, step)
                    writer.add_scalar("Training/LearningRate", scheduler_g.get_last_lr()[0], step)
                
                progress_bar.set_postfix({
                    "Step": step, "G": f"{g_meter.avg:.4f}", "D": f"{d_meter.avg:.4f}", 
                    "Mel": f"{mel_meter.avg:.4f}", "LR": f"{scheduler_g.get_last_lr()[0]:.2e}"
                })
                
                if step % train_cfg['save_interval'] == 0:
                    state = {
                        "step": step, "epoch": epoch,
                        "encoder": encoder.module.state_dict(), "decoder": decoder.module.state_dict(),
                        "mpd": mpd.module.state_dict(), "mrd": mrd.module.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "scheduler_g": scheduler_g.state_dict(), "scheduler_d": scheduler_d.state_dict(),
                    }
                    torch.save(state, f"checkpoints/ae/ae_{step}.pt")
                    torch.save(state, "checkpoints/ae/ae_latest.pt")

                    ckpt_dir = "checkpoints/ae"
                    try:
                        checkpoints = []
                        for f in os.listdir(ckpt_dir):
                            if f.startswith("ae_") and f.endswith(".pt") and f != "ae_latest.pt":
                                try:
                                    step_val = int(f.replace("ae_", "").replace(".pt", ""))
                                    checkpoints.append((step_val, f))
                                except ValueError: pass
                        checkpoints.sort(key=lambda x: x[0], reverse=True)
                        for _, old_ckpt in checkpoints[1000:]:
                            file_to_remove = os.path.join(ckpt_dir, old_ckpt)
                            if os.path.exists(file_to_remove):
                                os.remove(file_to_remove)
                                logger.info(f"Deleted old checkpoint: {old_ckpt}")
                    except Exception as e:
                        logger.warning(f"Error during checkpoint cleanup: {e}")
                    
                    if args.eval_input:
                        evaluate(encoder, decoder, mel_transform_input, args.eval_input, 
                                 "checkpoints/ae/eval", step, device, data_cfg['sample_rate'], args.local_rank)
            step += 1
        epoch += 1

if __name__ == "__main__":
    main()
