# train_fm.py
# Final version with full, flexible support for FP16 and BF16 Automatic Mixed Precision.
# The core training and sampling logic remains IDENTICAL to the original.

import argparse
import copy
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import trange
import wandb

# [NEW] Import GradScaler for fp16 support
from torch.cuda.amp import GradScaler

# --- Core Logic Imports (Unchanged) ---
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

# --- Import user-provided utilities ---
from utils import ema, generate_samples, infiniteloop


# [MODIFIED] Added --fp16 and --bf16 arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Conditional Flow Matching on CIFAR-10")

    # Model settings
    parser.add_argument("--model", type=str, default="otcfm", choices=["otcfm", "icfm", "fm", "si"],
                        help="Flow matching model type")
    parser.add_argument("--output_dir", type=str, default="./results/", help="Output directory")
    parser.add_argument("--num_channel", type=int, default=128, help="Base channel of UNet")

    # Training settings
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--total_steps", type=int, default=400001, help="Total training steps")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")
    parser.add_argument("--parallel", action="store_true", help="Use DataParallel")

    # [NEW] Mixed Precision arguments
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 Automatic Mixed Precision (AMP)")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable BF16 Automatic Mixed Precision (AMP). Recommended for Ampere+ GPUs.")

    # Checkpoint settings
    parser.add_argument("--save_step", type=int, default=10000, help="Checkpoint save interval")

    # Wandb settings
    parser.add_argument("--project", type=str, default="torchcfm-cifar10", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")

    return parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def warmup_lr(step, warmup_steps):
    return min(step, warmup_steps) / warmup_steps


def main():
    args = parse_args()

    # [NEW] Setup Automatic Mixed Precision (AMP)
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if args.bf16 and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 is not supported on this device. Disabling mixed precision.")
        args.bf16 = False
        use_amp = args.fp16  # Fallback to fp16 if it was also enabled
        amp_dtype = torch.float16

    if use_amp:
        print(f"Using Automatic Mixed Precision with dtype: {amp_dtype}")

    # Initialize wandb
    wandb.init(
        project=args.project,
        entity=args.wandb_entity,
        config=args,
        name=f"{args.model}-lr{args.lr}-bs{args.batch_size}"
    )

    print("Training with settings:", args)

    # [NEW] Initialize GradScaler. It's only truly needed for fp16.
    scaler = GradScaler(enabled=args.fp16)

    # Dataset and DataLoader
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    datalooper = infiniteloop(dataloader)

    model = UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=args.num_channel,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions="16", dropout=0.1
    ).to(device)

    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr(step, args.warmup))

    if args.parallel:
        model = torch.nn.DataParallel(model)
        ema_model = torch.nn.DataParallel(ema_model)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f} M")

    # Flow matcher setup
    matcher_map = {
        "otcfm": ExactOptimalTransportConditionalFlowMatcher, "icfm": ConditionalFlowMatcher,
        "fm": TargetConditionalFlowMatcher, "si": VariancePreservingConditionalFlowMatcher
    }
    FM = matcher_map[args.model](sigma=0.0)

    save_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)  # Use set_to_none=True for slightly better performance
            x1, _ = next(datalooper)
            x1 = x1.to(device)
            x0 = torch.randn_like(x1)

            # [MODIFIED] Wrap the forward pass and loss calculation with autocast
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                vt = model(t, xt)
                loss = torch.mean((vt - ut) ** 2)

            # [MODIFIED] Use conditional logic for the backward pass
            if args.fp16:
                # FP16 path with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # FP32 or BF16 path (no scaler needed)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            ema(model, ema_model, args.ema_decay)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=loss.item(), lr=current_lr)
            wandb.log({"loss": loss.item(), "lr": current_lr}, step=step)

            if (args.save_step > 0 and (step + 1) % args.save_step == 0) or step == 500:
                current_step = step + 1

                generate_samples(model, args.parallel, save_dir, current_step, net_="normal")
                generate_samples(ema_model, args.parallel, save_dir, current_step, net_="ema")

                normal_sample_path = os.path.join(save_dir, f"normal_generated_FM_images_step_{current_step}.png")
                ema_sample_path = os.path.join(save_dir, f"ema_generated_FM_images_step_{current_step}.png")

                if os.path.exists(normal_sample_path):
                    wandb.log(
                        {"normal_samples": wandb.Image(normal_sample_path, caption=f"Normal Step {current_step}")},
                        step=step)
                if os.path.exists(ema_sample_path):
                    wandb.log({"ema_samples": wandb.Image(ema_sample_path, caption=f"EMA Step {current_step}")},
                              step=step)

                torch.save({
                    "net_model": model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched": scheduler.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": current_step,
                    "args": args
                }, os.path.join(save_dir, f"{args.model}_cifar10_weights_step_{current_step}.pt"))

    wandb.finish()
    print("Training finished.")


if __name__ == "__main__":
    main()