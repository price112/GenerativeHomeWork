# train_dit_cifar10_ddpm_final.py
# FINAL, VERIFIED VERSION: Implements the official DiT method for Classifier-Free Guidance
# by using a model wrapper, ensuring compatibility and correctness.

import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import trange, tqdm
import wandb
from torch.cuda.amp import GradScaler
from PIL import Image
import tempfile
import glob
import math
from copy import deepcopy

from dit_adaptive import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from cleanfid import fid
from torch.utils.data import DataLoader


# --- Helper Utilities ---
def ema_update(ema_model, model, decay=0.9999):
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Official DiT DDPM training on CIFAR-10")
    # Model and Data
    parser.add_argument("--data_path", type=str, default="data/", help="Path to CIFAR-10 data.")
    parser.add_argument("--fid_ref_path", type=str, default="cifar10_train_images",
                        help="Path to local reference images for FID.")
    parser.add_argument("--model", type=str, default="DiT-XS/2")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    # Training
    parser.add_argument("--total_steps", type=int, default=450001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--learn_sigma", action="store_true", default=True)
    # Evaluation & Logging
    parser.add_argument("--ckpt_every", type=int, default=20000)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--sample_steps", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale for conditional sampling.")
    parser.add_argument("--fid_every", type=int, default=20000)
    parser.add_argument("--fid_num_gen", type=int, default=1000)
    parser.add_argument("--fid_batch_size", type=int, default=128)
    # Misc
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="dit-cifar10-ddpm")
    parser.add_argument("--output_dir", type=str, default="results_ddpm_cifar10/")
    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# [NEW] Model wrapper for Classifier-Free Guidance, aligned with official DiT repo
class ClassifierFreeGuidanceWrapper(nn.Module):
    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale

    def forward(self, x, t, y):
        # The model passed to the sampler must have the signature model(x, t, y)
        # This wrapper handles the CFG logic internally.

        # In unconditional generation, y will be None or not used.
        if not self.model.is_conditional or self.cfg_scale == 1.0:
            return self.model(x, t, y=y)  # Pass y for compatibility, model might ignore it

        # Perform CFG:
        # 1. Get the unconditional prediction
        y_uncond = torch.full_like(y, self.model.num_classes)
        # 2. Duplicate input for batched processing
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        y_in = torch.cat([y, y_uncond])
        # 3. Get predictions
        pred_cond, pred_uncond = self.model(x_in, t_in, y=y_in).chunk(2)
        # 4. Combine predictions
        return pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)


class DiffusionSampler:
    def __init__(self, model, diffusion, vae, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, sample_steps, use_ddim=True, cfg_scale=1.5, y_labels=None):
        self.model.eval()

        # [MODIFIED] Create the CFG wrapper model
        guided_model = ClassifierFreeGuidanceWrapper(self.model, cfg_scale)

        z = torch.randn(n_samples, self.model.in_channels, self.model.input_size, self.model.input_size,
                        device=self.device)
        model_kwargs = {}
        if self.model.is_conditional:
            if y_labels is None:
                n_per_class = math.ceil(n_samples / self.model.num_classes)
                y_labels = torch.arange(self.model.num_classes, device=self.device).repeat(n_per_class)[:n_samples]
            model_kwargs["y"] = y_labels

        diffusion_instance = create_diffusion(timestep_respacing=str(sample_steps))
        sample_fn = diffusion_instance.ddim_sample_loop if use_ddim else diffusion_instance.p_sample_loop

        # [MODIFIED] Pass the wrapper to the sampler, not the original model
        samples = sample_fn(
            guided_model, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=self.device
        )

        images = self.vae.decode(samples / 0.18215).sample
        images = (images.clamp(-1, 1) + 1) / 2
        self.model.train()
        return images


# The rest of the script is largely the same, but calls to the sampler are updated.
def prepare_cifar10_images(path, data_root):
    if os.path.exists(path) and len(glob.glob(os.path.join(path, '*.png'))) >= 50000:
        return
    print(f"Creating CIFAR-10 reference images at {path}...")
    os.makedirs(path, exist_ok=True)
    dataset = datasets.CIFAR10(root=data_root, train=True, download=True)
    for i, (img, _) in enumerate(tqdm(dataset, desc="Saving CIFAR-10 images")):
        img.save(os.path.join(path, f"cifar10_train_{i:05d}.png"))


def compute_and_log_fid(ema_model, diffusion, vae, args, step):
    print(f"\n--- Computing FID at step {step} ---")
    sampler = DiffusionSampler(ema_model, diffusion, vae, device)
    gen_dir = tempfile.mkdtemp()

    generated_count = 0
    with tqdm(total=args.fid_num_gen, desc="Generating images for FID", leave=False) as pbar:
        while generated_count < args.fid_num_gen:
            n_to_gen = min(args.fid_batch_size, args.fid_num_gen - generated_count)
            images = sampler.sample(n_samples=n_to_gen, sample_steps=50, use_ddim=True, cfg_scale=args.cfg_scale)
            for i in range(n_to_gen):
                save_image(images[i], os.path.join(gen_dir, f"img_{generated_count + i}.png"))
            generated_count += n_to_gen
            pbar.update(n_to_gen)

    score = fid.compute_fid(fdir1=args.fid_ref_path, fdir2=gen_dir, batch_size=args.fid_batch_size)
    print(f"--- FID @ step {step}: {score:.4f} ---")
    wandb.log({"FID Score": score}, step=step)

    for f in glob.glob(os.path.join(gen_dir, "*.png")): os.remove(f)
    os.rmdir(gen_dir)


def main(args):
    torch.manual_seed(args.seed)
    prepare_cifar10_images(path=args.fid_ref_path, data_root=args.data_path)

    model_string_name = args.model.replace("/", "-")
    experiment_dir = os.path.join(args.output_dir, model_string_name)
    os.makedirs(experiment_dir, exist_ok=True)

    wandb.init(project=args.project, config=args, name=f"{model_string_name}-bs{args.batch_size}-lr{args.lr}")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma
    ).to(device)
    ema_model = deepcopy(model)
    ema_update(ema_model, model, decay=0)
    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")

    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    datalooper = infiniteloop(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.total_steps
    )
    scaler = GradScaler(enabled=args.fp16)

    print(f"Training for {args.total_steps} steps...")
    with trange(args.total_steps, desc="Training Steps") as pbar:
        for step in pbar:
            model.train()
            x, y = next(datalooper)
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                z = vae.encode(x).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)
            model_kwargs = {"y": y}

            with torch.cuda.amp.autocast(enabled=args.fp16):
                loss_dict = diffusion.training_losses(model, z, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema_update(ema_model, model, decay=args.ema_decay)

            current_lr = lr_scheduler.get_last_lr()[0]
            loss_item = loss.item()
            wandb.log({"loss": loss_item, "lr": current_lr}, step=step)
            pbar.set_postfix(loss=f"{loss_item:.4f}", lr=f"{current_lr:.6f}")

            current_step = step + 1

            if current_step == 10 or current_step % args.sample_every == 0:
                print(f"\n--- Sampling at step {current_step} ---")
                sampler = DiffusionSampler(ema_model, diffusion, vae, device)
                samples = sampler.sample(n_samples=100, sample_steps=args.sample_steps, use_ddim=True,
                                         cfg_scale=args.cfg_scale)
                grid = make_grid(samples, nrow=10)
                save_path = os.path.join(experiment_dir, f"sample_{current_step}.png")
                save_image(grid, save_path)
                wandb.log({"samples": wandb.Image(save_path)}, step=step)

            if current_step == 10 or  current_step % args.ckpt_every == 0:
                print(f"\n--- Saving checkpoint at step {current_step} ---")
                torch.save({
                    "model": model.state_dict(), "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), "args": args
                }, os.path.join(experiment_dir, f"ckpt_{current_step}.pth"))

            if current_step == 10 or  current_step % args.fid_every == 0:
                compute_and_log_fid(ema_model, diffusion, vae, args, current_step)

    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)