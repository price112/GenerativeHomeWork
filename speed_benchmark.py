import time
import torch
from thop import profile
from dit_adaptive_speed_only import *

def benchmark(model: torch.nn.Module,
              input_size=(1, 3, 224, 224),
              warmup=10,               # 预热迭代
              reps=100,                # 计时迭代
              device='cuda'):

    assert torch.cuda.is_available()

    model = model.to(device).eval()          # 进入评估模式
    dummy  = torch.randn(*input_size, device=device)

    # 参数量
    mparams = sum(p.numel() for p in model.parameters()) / 1e6

    gflops, _ = profile(model, inputs=(dummy,), verbose=False)
    gflops /= 1e9

    with torch.no_grad():
        for _ in range(warmup):              # 预热，排除首次 CUDA 加载开销
            model(dummy)
        torch.cuda.synchronize()

        tic = time.time()
        for _ in range(reps):
            model(dummy)
        torch.cuda.synchronize()
        toc = time.time()

    latency_ms = (toc - tic) / reps * 1000
    throughput = input_size[0] / (latency_ms / 1000)

    return mparams, gflops, latency_ms, throughput


if __name__ == "__main__":

    for name, net in [("DiT_S_2", DiT_S_2()),
                      ("DiT_S_2_diff", DiT_S_2_diff()),
                      ("DiT_B_2", DiT_B_2()),
                      ("DiT_B_2_diff", DiT_B_2_diff()),
                      ("DiT_L_2", DiT_L_2()),
                      ("DiT_L_2_diff", DiT_L_2_diff()),
                      ("DiT_XL_2", DiT_XL_2()),
                      ("DiT_XL_2_diff", DiT_XL_2_diff())]:
        m, f, l, t = benchmark(net, input_size=(32, 4, 32, 32))
        print(f"{name:10}: {m:.2f} M params | {f:.2f} GFLOPs | "
              f"{l:.2f} ms/batch | {t:.1f} img/s")
