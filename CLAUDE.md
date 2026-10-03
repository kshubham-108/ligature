# Ligature

A GPT-style language model written from scratch in PyTorch and trained on TinyStories: my own byte-level BPE tokeniser, the transformer, the training loop, a KV-cache for generation, controlled ablations, tests, CI and a small Dockerised API.

This is a learning project as much as a portfolio piece. I need to understand and be able to explain every line, so clear and conventional beats clever.

The name is a nod to typographic ligatures (fi → ﬁ): the tokeniser builds its vocabulary the same way, by fusing frequent pairs of symbols into one.

## Rules
- Write the core pieces by hand. Do not use `nn.Transformer`, `nn.MultiheadAttention`, Hugging Face `transformers` / `tokenizers` / `datasets` / `Trainer`, PyTorch Lightning, tiktoken or sentencepiece. `F.scaled_dot_product_attention` is allowed only behind the `use_sdpa` flag, next to the explicit implementation.
- Allowed dependencies: torch, numpy, regex, pyyaml, tqdm, matplotlib, huggingface_hub (to download the data and upload the checkpoint), fastapi, uvicorn, pydantic, pytest, ruff. Ask before adding anything else.
- Never invent numbers. Every figure in the README or `results/` must come from a file written by a real run.
- Never weaken or delete a test to make it pass. If you think a test is wrong, explain why first.
- Do not run `git commit` or `git push`. I review the diff and commit myself at the end of each phase.
- `configs/tiny.yaml` must always train on CPU in under 60 seconds; CI depends on it.
- Pick the device in this order: CUDA, MPS, CPU.

## How to work with me
- Before building a new component, outline it in at most five bullet points: what it does, the main design choice and why.
- Stay within the phase I ask for and keep diffs small.
- At the end of each phase, give me the files you changed, the exact commands to run, and three questions I should be able to answer about the new code. Do not answer them unless I ask.

## Code style
- Python 3.11, PEP 8, type hints, `ruff` clean, 100-character lines. Small functions, dataclasses for configs, no global state.
- Conventional ML names: `B, T, C`, `n_layer`, `n_head`, `d_model`, `head_dim`, `block_size`.
- Comments explain why, or spell out a non-obvious shape or piece of maths. Never restate what the code already says; most lines need no comment at all.
- Put shape comments such as `# (B, n_head, T, head_dim)` on reshapes and transposes.
- British English in comments, docstrings and docs: normalise, initialise, behaviour, tokeniser, optimiser.
- Docstrings are one line; describe arguments only when they are not obvious.
- Plain, understated tone everywhere, including the README. No emojis, no decorative banners or separator lines, no "Step 1 / Step 2" narration, no exclamation marks. Avoid "This function...", "Here we...", "Let's...", "Note that...", "robust", "seamless", "leverage", "comprehensive", "state-of-the-art".
- `# TODO:` only for specific, real follow-up work.

## Layout
- `src/ligature/`: config.py, tokenizer.py, data.py, model.py, train.py, generate.py, utils.py
- `scripts/`: prepare_data.py, bigram_baseline.py, benchmark_kv_cache.py, plot_results.py, push_to_hub.py
- `configs/`: tiny, base, base_learned, size_small, size_medium
- `tests/`, `app/`, `notebooks/` and `results/` are committed; `data/` and `runs/` are git-ignored

## Phases
0 scaffold, 1 tokeniser and data, 2 model, 3 training, 4 GPU runs and plots, 5 KV-cache and generation, 6 API, Docker and CI, 7 README, 8 Hugging Face Spaces deploy