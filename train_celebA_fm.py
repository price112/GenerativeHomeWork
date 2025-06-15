# train_unconditional_celeba.py
# This script trains an unconditional DiT model on a custom image folder (e.g., CelebA)
# using Conditional Flow Matching in latent space, with integrated sampling and FID evaluation.

import argparse
import copy
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import trange, tqdm
import wandb
from torch.cuda.amp import GradScaler
from PIL import Image
import tempfile
import glob

from cleanfid import fid
from torchdyn.core import NeuralODE

from dit_adaptive import DiT_models
from diffusers.models import AutoencoderKL
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from diffusers.optimization import get_scheduler

from utils import ema, infiniteloop
import torch.nn.functional as F

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
        return image, 0  # Return a dummy label


def parse_args():
    parser = argparse.ArgumentParser(description="Unconditional DiT training on a custom dataset")

    # [MODIFIED] Model and Data settings
    parser.add_argument("--data_path", type=str, default='data/celeba_hq_256', help="Path to the folder of training images.")
    parser.add_argument("--model", type=str, default="DiT-S/4",
                        help="DiT model type. Choose one with a patch size appropriate for your image size.")
    parser.add_argument("--num_classes", type=int, default=0,
                        help="0 or 1 for an unconditional model, N for an conditional model")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for training. Images will be resized to this.")
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"], help="VAE model to use.")

    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=250001)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 Automatic Mixed Precision (AMP)")

    parser.add_argument("--save_step", type=int, default=5000)
    parser.add_argument("--fid_every", type=int, default=20000)
    parser.add_argument("--fid_num_gen", type=int, default=1000, help="Number of samples to generate for FID.")
    parser.add_argument("--fid_batch_size", type=int, default=64, help="Batch size for FID generation.")

    parser.add_argument("--project", type=str, default="dit-celebA-fm")
    parser.add_argument("--output_dir", type=str, default="results_celebA_fm_256_2/")

    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnconditionalFlowSampler:
    def __init__(self, model, vae, device="cuda"):
        self.model = model
        self.vae = vae
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, sample_steps=100):
        self.model.eval()

        def ode_func(t, z, **kwargs):
            t_expanded = t.expand(n_samples) if t.numel() == 1 else t
            return self.model(z, t_expanded)

        node = NeuralODE(ode_func, solver="euler")
        z0 = torch.randn(n_samples, 4, self.model.input_size, self.model.input_size, device=self.device)
        traj = node.trajectory(z0, t_span=torch.linspace(0, 1, sample_steps, device=self.device))

        z1 = traj[-1]
        images = self.vae.decode(z1 / 0.18215).sample
        images = (images.clamp(-1, 1) + 1) / 2

        self.model.train()
        return images


def compute_and_log_fid(ema_model, vae, args, step):
    print(f"\n--- Computing FID at step {step} ---")
    ema_model.eval()

    # Create a temporary directory to store generated images
    gen_dir = tempfile.mkdtemp()
    sampler = UnconditionalFlowSampler(ema_model, vae, device)

    # Generate and save images
    generated_count = 0
    with tqdm(total=args.fid_num_gen, desc="Generating images for FID") as pbar:
        while generated_count < args.fid_num_gen:
            n_to_gen = min(args.fid_batch_size, args.fid_num_gen - generated_count)
            images = sampler.sample(n_samples=n_to_gen, sample_steps=100)
            for i in range(n_to_gen):
                img_path = os.path.join(gen_dir, f"img_{generated_count + i}.png")
                save_image(images[i], img_path)
            generated_count += n_to_gen
            pbar.update(n_to_gen)

    score = fid.compute_fid(fdir1=args.data_path, fdir2=gen_dir, batch_size=args.fid_batch_size)

    print(f"--- FID @ step {step}: {score:.4f} ---")
    wandb.log({"FID Score": score}, step=step)

    # Clean up temporary directory
    for f in glob.glob(os.path.join(gen_dir, "*.png")):
        os.remove(f)
    os.rmdir(gen_dir)

    ema_model.train()


def main():
    args = parse_args()
    wandb.init(project=args.project, config=args)
    print("Training with settings:", args)

    scaler = GradScaler(enabled=args.fp16)

    # Models and VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes = args.num_classes,

    ).to(device)
    ema_model = copy.deepcopy(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # [MODIFIED] Data Loading
    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(path=args.data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=args.warmup, num_training_steps=args.total_steps
    )
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    save_dir = os.path.join(args.output_dir, args.model.replace("/", "-"))
    os.makedirs(save_dir, exist_ok=True)

    sampler = UnconditionalFlowSampler(ema_model, vae, device)

    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            x1_images, _ = next(datalooper)  # Ignore dummy label
            x1_images = x1_images.to(device)

            with torch.no_grad():
                z1 = vae.encode(x1_images).latent_dist.sample().mul_(0.18215)
            z0 = torch.randn_like(z1)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
                t, zt, ut = FM.sample_location_and_conditional_flow(z0, z1)
                vt = model(zt, t)  # Unconditional call
                loss = F.mse_loss(vt, ut)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema(model, ema_model, args.ema_decay)

            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=loss.item(), lr=current_lr)
            wandb.log({"loss": loss.item(), "lr": current_lr}, step=step)

            current_step = step + 1
            if current_step==10 or current_step % args.save_step == 0:
                print(f"\n--- Generating samples at step {current_step} ---")
                sample_images = sampler.sample(n_samples=64)
                grid = make_grid(sample_images, nrow=8)
                sample_path = os.path.join(save_dir, f"sample_step_{current_step}.png")
                save_image(grid, sample_path)
                wandb.log({"samples": wandb.Image(sample_path)}, step=step)

            if  current_step==10 or current_step % args.fid_every == 0:
                compute_and_log_fid(ema_model, vae, args, current_step)

                torch.save({"ema_model": ema_model.state_dict()},
                           os.path.join(save_dir, f"ckpt_step_{current_step}.pt"))

    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()