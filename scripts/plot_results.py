"""Plot validation curves and the size sweep, and write a summary table, from files in results/."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import NullFormatter

SIZE_SWEEP = ("size_small", "size_medium", "base")

# Categorical hues in a fixed order, checked for colour-blind separation. A colour belongs to a
# model variant (its config apart from the seed), and the seed is shown by line style.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SEED_STYLES = ["-", "--", ":", "-."]
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)


@dataclass
class Run:
    name: str
    config: dict
    info: dict
    rows: list[dict[str, float]]

    @property
    def final(self) -> dict[str, float]:
        return self.rows[-1]

    @property
    def tokens_per_step(self) -> int:
        c = self.config
        return c["batch_size"] * c["grad_accum_steps"] * c["block_size"]

    @property
    def variant(self) -> tuple:
        """The config without its seed, so runs that differ only by seed share a colour."""
        return tuple(sorted((k, v) for k, v in self.config.items() if k != "seed"))


def load_runs(runs_dir: Path) -> list[Run]:
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if (p / "log.csv").exists()):
        with (run_dir / "log.csv").open(newline="", encoding="utf-8") as f:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        info = yaml.safe_load((run_dir / "run_info.yaml").read_text(encoding="utf-8"))
        runs.append(Run(run_dir.name, config, info, rows))
    seeds = list(dict.fromkeys(run.config["seed"] for run in runs))
    return sorted(runs, key=lambda run: (seeds.index(run.config["seed"]), run.name))


def style_axes(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def draw_bigram(ax: plt.Axes, bigram: dict) -> None:
    ax.axhline(bigram["val_loss"], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        f"bigram baseline ({bigram['val_loss']:.3f})",
        xy=(1.0, bigram["val_loss"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=9,
    )


def plot_val_loss(runs: list[Run], bigram: dict, path: Path) -> None:
    variants = list(dict.fromkeys(run.variant for run in runs))
    seeds = list(dict.fromkeys(run.config["seed"] for run in runs))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for run in runs:
        # Step 0 is left out: the initial loss of about ln(vocab_size) would flatten the curves.
        rows = [r for r in run.rows if r["step"] > 0]
        tokens = [r["step"] * run.tokens_per_step / 1e6 for r in rows]
        ax.plot(
            tokens,
            [r["val_loss"] for r in rows],
            color=SERIES_COLOURS[variants.index(run.variant) % len(SERIES_COLOURS)],
            linestyle=SEED_STYLES[seeds.index(run.config["seed"]) % len(SEED_STYLES)],
            linewidth=2,
            label=f"{run.name} (seed {run.config['seed']})",
        )
    draw_bigram(ax, bigram)
    ax.set_title("Validation loss during training", loc="left")
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("validation loss (nats per token)")
    style_axes(fig, ax)
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_scaling(runs: list[Run], bigram: dict, path: Path) -> None:
    sweep = sorted(
        (run for run in runs if run.name in SIZE_SWEEP),
        key=lambda run: run.info["non_embedding_params"],
    )
    if len(sweep) < 2:
        print(f"skipping {path.name}: need at least two of {SIZE_SWEEP}")
        return
    params = [run.info["non_embedding_params"] for run in sweep]
    losses = [run.final["val_loss"] for run in sweep]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(params, losses, color=SERIES_COLOURS[0], linewidth=2, marker="o", markersize=8)
    for run, n, loss in zip(sweep, params, losses, strict=True):
        ax.annotate(
            f"{run.name}\n{loss:.3f}",
            xy=(n, loss),
            # Below-left of each point stays clear of a line that slopes down to the right.
            xytext=(-8, -6),
            textcoords="offset points",
            ha="right",
            va="top",
            color=INK_SECONDARY,
            fontsize=9,
        )
    draw_bigram(ax, bigram)
    ax.set_xscale("log")
    ax.set_xlim(min(params) / 2, max(params) * 2)
    # Tick exactly at the models in the sweep, labelled with their size.
    ax.set_xticks(params, [f"{n / 1e6:.2f}M" for n in params])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Final validation loss against model size", loc="left")
    ax.set_xlabel("non-embedding parameters (log scale)")
    ax.set_ylabel("final validation loss (nats per token)")
    style_axes(fig, ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def gap_pairs(runs: list[Run]) -> list[tuple[Run, Run]]:
    """Pair each learned-position run with the RoPE run whose config is otherwise identical."""

    def key(run: Run) -> tuple:
        return tuple(sorted((k, v) for k, v in run.config.items() if k != "pos_encoding"))

    rope = {key(run): run for run in runs if run.config["pos_encoding"] == "rope"}
    pairs = [
        (rope[key(run)], run)
        for run in runs
        if run.config["pos_encoding"] == "learned" and key(run) in rope
    ]
    return pairs


def write_summary(runs: list[Run], bigram: dict, path: Path) -> None:
    lines = [
        "# Results",
        "",
        "Generated by `scripts/plot_results.py` from `results/runs/*/` and `results/bigram.json`.",
        "Losses are mean cross-entropy in nats per token on the validation split.",
        "",
        "| Run | Positions | Seed | Non-embedding params | Tokens seen | Final val loss "
        "| Final val perplexity | Training time |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        final = run.final
        lines.append(
            f"| {run.name} | {run.config['pos_encoding']} | {run.config['seed']} "
            f"| {run.info['non_embedding_params']:,} "
            f"| {int(final['step']) * run.tokens_per_step:,} "
            f"| {final['val_loss']:.4f} | {final['val_ppl']:.2f} "
            f"| {final['elapsed_s'] / 60:.1f} min |"
        )
    lines.append(
        f"| bigram baseline | - | - | - | {bigram['train_tokens']:,} "
        f"| {bigram['val_loss']:.4f} | {bigram['val_ppl']:.2f} | - |"
    )

    pairs = gap_pairs(runs)
    if pairs:
        lines += [
            "",
            "## RoPE against learned positions",
            "",
            "Gap is learned minus RoPE final validation loss; positive means RoPE did better.",
            "",
            "| Seed | RoPE run | Learned run | RoPE val loss | Learned val loss | Gap |",
            "|---|---|---|---:|---:|---:|",
        ]
        for rope, learned in pairs:
            r, lr = rope.final["val_loss"], learned.final["val_loss"]
            lines.append(
                f"| {rope.config['seed']} | {rope.name} | {learned.name} "
                f"| {r:.4f} | {lr:.4f} | {lr - r:+.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    runs = load_runs(args.results_dir / "runs")
    if not runs:
        raise SystemExit(f"no runs with a log.csv under {args.results_dir / 'runs'}")
    bigram = json.loads((args.results_dir / "bigram.json").read_text(encoding="utf-8"))

    plot_val_loss(runs, bigram, args.results_dir / "val_loss.png")
    plot_scaling(runs, bigram, args.results_dir / "scaling.png")
    write_summary(runs, bigram, args.results_dir / "summary.md")
    print(f"wrote val_loss.png, scaling.png and summary.md for {len(runs)} runs")


if __name__ == "__main__":
    main()
