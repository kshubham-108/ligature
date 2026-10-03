"""Time generation with and without the KV-cache on CPU, and on GPU when one is available."""

import argparse
import csv
import platform
import statistics
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ligature.config import load_config
from ligature.model import GPT
from ligature.tokenizer import Tokenizer
from ligature.train import load_checkpoint
from ligature.utils import get_device

FIELDS = [
    "device",
    "device_name",
    "torch_threads",
    "new_tokens",
    "use_cache",
    "median_s",
    "ms_per_token",
    "tokens_per_sec",
    "kv_cache_bytes",
]
SERIES = {True: ("with KV-cache", "#2a78d6"), False: ("without cache", "#eb6834")}
SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def cpu_name() -> str:
    """Best-effort CPU model name; platform.processor() is often vague or empty."""
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Windows":
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown CPU"


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return cpu_name() if device.type == "cpu" else device.type


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_generation(
    model: GPT, start: torch.Tensor, new_tokens: int, use_cache: bool, warmup: int, repeats: int
) -> float:
    """Median wall-clock seconds for one greedy generation, after warm-up runs."""
    times = []
    for i in range(warmup + repeats):
        synchronize(start.device)
        t0 = time.perf_counter()
        model.generate(start, new_tokens, temperature=0, use_cache=use_cache)
        synchronize(start.device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def plot(rows: list[dict], path: Path) -> None:
    devices = list(dict.fromkeys(row["device"] for row in rows))
    fig, axes = plt.subplots(1, len(devices), figsize=(6 * len(devices), 4.5), dpi=150)
    axes = [axes] if len(devices) == 1 else list(axes)
    fig.set_facecolor(SURFACE)
    for ax, device in zip(axes, devices, strict=True):
        name = next(row["device_name"] for row in rows if row["device"] == device)
        for use_cache, (label, colour) in SERIES.items():
            points = [r for r in rows if r["device"] == device and r["use_cache"] == use_cache]
            x = [r["new_tokens"] for r in points]
            y = [r["ms_per_token"] for r in points]
            ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=8, label=label)
            ax.annotate(
                f"{y[-1]:.1f} ms",
                xy=(x[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=9,
            )
        ax.set_title(f"{device}: {name}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("tokens generated", color=INK_SECONDARY)
        ax.set_ylabel("milliseconds per token", color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_ylim(bottom=0)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY)
        legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)
    fig.suptitle("Greedy generation from one start token, batch 1", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, help="trained checkpoint; default: random weights")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--vocab-size", type=int, default=4096, help="for the random model")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 255])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-csv", type=Path, default=Path("results/kv_cache.csv"))
    parser.add_argument("--out-png", type=Path, default=Path("results/kv_cache.png"))
    args = parser.parse_args()

    devices = [torch.device("cpu")]
    if get_device().type != "cpu":
        devices.append(get_device())

    rows = []
    for device in devices:
        if args.ckpt:
            model, _ = load_checkpoint(args.ckpt, device)
            eot_id = Tokenizer.load(args.ckpt.parent / "tokenizer.json").eot_id
        else:
            # Speed does not depend on the weights, so a randomly initialised model will do.
            torch.manual_seed(0)
            model = GPT(load_config(args.config, []).model_config(args.vocab_size)).to(device)
            eot_id = args.vocab_size - 1  # the tokeniser puts <|endoftext|> last
        model.eval()
        start = torch.tensor([[eot_id]], device=device)  # (1, 1)
        caches = model.make_caches(1, device)
        cache_bytes = sum(c.k.nbytes + c.v.nbytes for c in caches)

        for new_tokens in args.lengths:
            for use_cache in (True, False):
                seconds = time_generation(
                    model, start, new_tokens, use_cache, args.warmup, args.repeats
                )
                row = {
                    "device": device.type,
                    "device_name": device_name(device),
                    "torch_threads": torch.get_num_threads(),
                    "new_tokens": new_tokens,
                    "use_cache": use_cache,
                    "median_s": round(seconds, 4),
                    "ms_per_token": round(1000 * seconds / new_tokens, 3),
                    "tokens_per_sec": round(new_tokens / seconds, 1),
                    "kv_cache_bytes": cache_bytes if use_cache else 0,
                }
                rows.append(row)
                print(
                    f"{device.type:>4} | {new_tokens:>3} tokens | cache {str(use_cache):<5} "
                    f"| {row['ms_per_token']:8.2f} ms/token | {row['tokens_per_sec']:8.1f} tok/s"
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.out_png)
    print(f"wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
