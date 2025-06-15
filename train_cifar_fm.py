# train_cfm_conditional_with_fid.py
# Final version with fully integrated conditional FID evaluation.

import argparse
import copy
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import trange
import wandb
from torch.cuda.amp import GradScaler

# --- Imports for FID and ODE solving ---
from cleanfid import fid
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

# --- Imports for DiT, VAE, and CFM ---
from dit_adaptive import DiT_models
from diffusers.models import AutoencoderKL
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
import math
from utils import ema, infiniteloop


def parse_args():
    parser = argparse.ArgumentParser(description="Conditional DiT training with Flow Matching and FID eval")

    parser.add_argument("--model", type=str, default="DiT-XS/2", choices=list(DiT_models.keys()))
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=450001)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--save_step", type=int, default=5000, help="Checkpoint and sample save interval")

    parser.add_argument("--fid_every", type=int, default=20000, help="Interval for FID computation. 0 to disable.")
    parser.add_argument("--fid_num_gen", type=int, default=1000, help="Number of samples to generate for FID.")
    parser.add_argument("--fid_batch_size", type=int, default=256, help="Batch size for FID generation.")
    parser.add_argument("--fid_integration_method", type=str, default="dopri5", help="ODE solver for FID generation.")
    parser.add_argument("--fid_tol", type=float, default=1e-5, help="Tolerance for high-order ODE solvers.")

    parser.add_argument("--project", type=str, default="dit-cifar-flow-matching")
    parser.add_argument("--output_dir", type=str, default="results_fm_cifar/")

    return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warmup_lr(step, warmup_steps):
    return min(step, warmup_steps) / warmup_steps


def compute_and_log_fid(ema_model, vae, args, step):

    print(f"\n--- Computing FID at step {step} ---")
    ema_model.eval()

    model_ref = ema_model.module if hasattr(ema_model, 'module') else ema_model

    @torch.no_grad()
    def fid_generator(batch_size):
        y_batch = torch.randint(0, args.num_classes, (batch_size,), device=device)

        ode_func = lambda t, z, **kwargs: model_ref(z, t.expand(batch_size), y_batch)

        z0 = torch.randn(batch_size, 4, args.image_size // 8, args.image_size // 8, device=device)
        if args.fid_integration_method == "euler":
            node = NeuralODE(ode_func, solver="euler")
            t_span = torch.linspace(0, 1, 101, device=device)
            z_final = node.trajectory(z0, t_span=t_span)[-1]
        else:
            t_span = torch.tensor([0.0, 1.0], device=device)
            z_final = \
            odeint(ode_func, z0, t_span, rtol=args.fid_tol, atol=args.fid_tol, method=args.fid_integration_method)[-1]

        images = vae.decode(z_final / 0.18215).sample
        images_uint8 = (images.clamp(-1, 1) * 127.5 + 128).to(torch.uint8)

        return images_uint8

    score = fid.compute_fid(
        gen=lambda _: fid_generator(args.fid_batch_size),
        dataset_name="cifar10",
        batch_size=args.fid_batch_size,
        dataset_res=32,
        num_gen=args.fid_num_gen,
        dataset_split="train",
        mode="legacy_tensorflow",
    )

    print(f"--- FID @ step {step}: {score:.4f} ---")
    wandb.log({"FID Score": score}, step=step)
    ema_model.train()



def main():
    args = parse_args()

    # Setup
    use_amp = args.fp16
    scaler = GradScaler(enabled=use_amp)
    wandb.init(project=args.project, config=args)
    print("Training with settings:", args)

    # Models and VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes,
        in_channels=4
    ).to(device)
    ema_model = copy.deepcopy(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    transform = transforms.Compose([
        transforms.Resize(args.image_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # Optimizer, Scheduler, and Flow Matcher
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr(step, args.warmup))

    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    save_dir = os.path.join(args.output_dir, args.model.replace("/", "-"))
    os.makedirs(save_dir, exist_ok=True)

    sampler = ConditionalFlowSampler(ema_model, vae, device)

    # Training Loop
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            x1_images, y1 = next(datalooper)
            x1_images, y1 = x1_images.to(device), y1.to(device)

            with torch.no_grad():
                z1 = vae.encode(x1_images).latent_dist.sample().mul_(0.18215)
            z0 = torch.randn_like(z1)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                t, zt, ut = FM.sample_location_and_conditional_flow(z0, z1)
                vt = model(zt, t, y1)
                loss = torch.mean((vt - ut) ** 2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            ema(model, ema_model, args.ema_decay)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=loss.item(), lr=current_lr)
            wandb.log({"loss.4f": loss.item(), "lr.6f": current_lr}, step=step)

            current_step = step + 1
            # --- Periodic Evaluation Block ---
            # --- Periodic Evaluation Block ---
            if current_step ==100 or current_step % args.save_step == 0:
                print(f"\n--- Generating samples at step {current_step} ---")
                # [MODIFIED] The call is now simpler. The sampler handles label generation.
                sample_images = sampler.sample(n_samples=100) # Generate 100 samples for a 10x10 grid
                
                grid = make_grid(sample_images, nrow=10)
                sample_path = os.path.join(save_dir, f"sample_step_{current_step}.png")
                save_image(grid, sample_path)
                wandb.log({"samples": wandb.Image(sample_path)}, step=step)

                # Checkpointing
                torch.save({
                    "net_model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                }, os.path.join(save_dir, f"ckpt_step_{current_step}.pt"))

            # --- Periodic FID Computation ---
            if current_step ==100 or current_step % args.fid_every == 0:
                compute_and_log_fid(ema_model, vae, args, current_step)

    wandb.finish()
    print("Training finished.")


class ConditionalFlowSampler:
    def __init__(self, model, vae, device="cuda"):
        self.model = model
        self.vae = vae
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples, sample_steps=100, y_labels=None):
        """
        Generates samples, with label generation logic identical to DiffusionSampler.
        If y_labels is None, it creates a balanced, column-ordered set of labels.
        """
        self.model.eval()

        # [MODIFIED] This logic is now identical to the DiffusionSampler
        if self.model.is_conditional:
            if y_labels is None:
                # If no labels are provided, create a balanced set for grid display
                num_classes = self.model.num_classes
                n_per_class = math.ceil(n_samples / num_classes)
                # Use .repeat() to get [0,1,..,9, 0,1,..,9, ...] for column consistency
                y = torch.arange(num_classes, device=self.device).repeat(n_per_class)
                y = y[:n_samples]  # Ensure exact number of samples
            else:
                # Use the provided labels
                y = y_labels.to(self.device)
        else: # Unconditional case
            y = None
        
        n_total_samples = n_samples

        def ode_func(t, z, **kwargs):
            t_expanded = t.expand(n_total_samples) if t.numel() == 1 else t
            # The adaptive DiT model handles the case where y is None
            return self.model(z, t_expanded, y)

        node = NeuralODE(ode_func, solver="euler")
        z0 = torch.randn(n_total_samples, 4, self.model.input_size, self.model.input_size, device=self.device)
        traj = node.trajectory(z0, t_span=torch.linspace(0, 1, sample_steps, device=self.device))

        z1 = traj[-1]
        images = self.vae.decode(z1 / 0.18215).sample
        images = (images.clamp(-1, 1) + 1) / 2
        self.model.train()
        return images


if __name__ == "__main__":
    main()