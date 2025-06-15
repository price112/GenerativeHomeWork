# train_dit_ddpm_official_steps.py
# This script aligns with the official DiT repository's step-based training paradigm.

import argparse
import copy
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import trange
import wandb
from torch.cuda.amp import GradScaler
from PIL import Image
import tempfile
import glob

from copy import deepcopy
from dit_adaptive import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from cleanfid import fid
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.image_files = sorted(
            [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

def ema_update(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def infiniteloop(dataloader):
    """Creates an infinite iterator over a dataloader."""
    while True:
        for x, y in iter(dataloader):
            yield x, y


def parse_args():
    parser = argparse.ArgumentParser(description="Official DiT DDPM training on a custom dataset")
    # Model and Data
    parser.add_argument("--data_path", type=str, default='data/celeba_hq_256', help="Path to the folder of training images.")
    parser.add_argument("--model", type=str, default="DiT-S/4", help="DiT model configuration.")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for training.")
    parser.add_argument("--num_classes", type=int, default=0, help="Set to 0 for unconditional training.")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"], help="VAE model to use.")

    parser.add_argument("--total_steps", type=int, default=250001, help="Total number of training steps.")
    parser.add_argument("--lr", type=float, default=0.00015, help="Initial learning rate.")
    parser.add_argument("--lr_warmup_steps", type=int, default=5000, help="Number of warmup steps for learning rate.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping magnitude.")

    parser.add_argument("--ckpt_every", type=int, default=20000, help="Save checkpoint every N steps.")
    parser.add_argument("--sample_every", type=int, default=5000, help="Generate samples every N steps.")
    parser.add_argument("--fid_every", type=int, default=20000, help="Compute FID every N steps.")
    parser.add_argument("--fid_num_gen", type=int, default=1000, help="Number of samples to generate for FID.")
    parser.add_argument("--fid_batch_size", type=int, default=128, help="Batch size for FID generation.")

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="dit-celebA-ddpm")
    parser.add_argument("--output_dir", type=str, default="results_celebA_ddpm_256_2")
    return parser.parse_args()


# [FIXED] Sampler class to be compatible with older diffusion.py library versions

class DiffusionSampler:
    def __init__(self, model, diffusion, vae, device="cuda"):
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, use_ddim=True, cfg_scale=1.0):
        self.model.eval()

        model_kwargs = {}
        z = torch.randn(
            n_samples,
            self.model.in_channels,
            self.model.input_size,
            self.model.input_size,
            device=self.device
        )

        if self.model.is_conditional:
            y = torch.arange(self.model.num_classes, device=self.device).repeat_interleave(n_samples // self.model.num_classes)
            if len(y) < n_samples:
                y_extra = torch.arange(self.model.num_classes, device=self.device).repeat(1)
                y = torch.cat([y, y_extra[:n_samples - len(y)]])
            model_kwargs["y"] = y

        # Select sampling function
        sample_fn = self.diffusion.ddim_sample_loop if use_ddim else self.diffusion.p_sample_loop

        # [MODIFIED] The `guidance_scale` argument is removed from the call
        # as the user's library version does not support it.
        samples = sample_fn(
            self.model,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device
            # The 'guidance_scale' keyword argument has been removed from this call
        )

        # Decode the final latents into images
        latents = samples
        images = self.vae.decode(latents / 0.18215).sample
        images = (images.clamp(-1, 1) + 1) / 2

        self.model.train()
        return images



def compute_and_log_fid(ema_model, diffusion, vae, args, step):
    print(f"\n--- Computing FID at step {step} ---")
    ema_model.eval()
    gen_dir = tempfile.mkdtemp()
    sampler = DiffusionSampler(ema_model, diffusion, vae)

    generated_count = 0
    with tqdm(total=args.fid_num_gen, desc="Generating images for FID", leave=False) as pbar:
        while generated_count < args.fid_num_gen:
            n_to_gen = min(args.fid_batch_size, args.fid_num_gen - generated_count)
            images = sampler.sample(n_samples=n_to_gen, use_ddim=True, cfg_scale=1.0)
            for i in range(n_to_gen):
                img_path = os.path.join(gen_dir, f"img_{generated_count + i}.png")
                save_image(images[i], img_path)
            generated_count += n_to_gen
            pbar.update(n_to_gen)

    score = fid.compute_fid(fdir1=args.data_path, fdir2=gen_dir, batch_size=args.fid_batch_size)
    print(f"--- FID @ step {step}: {score:.4f} ---")
    wandb.log({"FID Score": score}, step=step)

    for f in glob.glob(os.path.join(gen_dir, "*.png")): os.remove(f)
    os.rmdir(gen_dir)
    ema_model.train()

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_string_name = args.model.replace("/", "-")
    experiment_dir = os.path.join(args.output_dir, model_string_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")
    args.output_dir = experiment_dir

    wandb.init(project=args.project, config=args)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    model = DiT_models[args.model](input_size=args.image_size // 8, num_classes=args.num_classes, learn_sigma = True).to(device)
    ema_model = deepcopy(model)
    ema_update(ema_model, model, decay=0)

    print(f"Model: {args.model}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = create_diffusion(timestep_respacing="")

    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(path=args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    datalooper = infiniteloop(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    lr_scheduler = get_scheduler(
        name="constant_with_warmup", optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.total_steps
    )
    scaler = GradScaler(enabled=args.fp16)


    print(f"Training for {args.total_steps} steps...")
    for step in trange(args.total_steps, desc="Training Steps"):
        model.train()
        x, y = next(datalooper)
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample().mul_(0.18215)

        t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)

        model_kwargs = {}
        if model.is_conditional:
            model_kwargs["y"] = y

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

        wandb.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=step)

        # Periodic evaluation using the current step number
        if step ==10 or (step+1) % args.sample_every == 0:
            print(f"\n--- Sampling at step {step} ---")
            sampler = DiffusionSampler(ema_model, diffusion, vae, device)
            samples = sampler.sample(n_samples=64, use_ddim=False)
            grid = make_grid(samples, nrow=8)
            save_path = os.path.join(args.output_dir, f"sample_{step}.png")
            save_image(grid, save_path)
            wandb.log({"samples": wandb.Image(save_path)}, step=step)

        if step ==10 or (step+1) % args.ckpt_every == 0:
            print(f"\n--- Saving checkpoint at step {step} ---")
            torch.save({
                "model": model.state_dict(), "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "args": args
            }, os.path.join(args.output_dir, f"ckpt_{step}.pth"))

        if step ==10 or (step+1) % args.fid_every == 0:
            compute_and_log_fid(ema_model, diffusion, vae, args, step)

    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    args = parse_args()
    main(args)