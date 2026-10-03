"""FastAPI app that serves story generation from a trained Ligature checkpoint."""

import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from ligature.model import GPT
from ligature.tokenizer import Tokenizer
from ligature.train import load_checkpoint
from ligature.utils import get_device

INDEX_HTML = Path(__file__).with_name("index.html")

Loader = Callable[[], tuple[GPT, Tokenizer]]


class GenerateRequest(BaseModel):
    prompt: str = Field("Once upon a time", max_length=2000)
    max_new_tokens: int = Field(200, ge=1, le=1024, description="capped to fit in block_size")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 means greedy")
    top_k: int | None = Field(None, ge=1)


class GenerateResponse(BaseModel):
    text: str
    new_tokens: int
    tokens_per_sec: float


def load_from_env() -> tuple[GPT, Tokenizer]:
    """Load ckpt.pt and tokenizer.json from the HF_REPO_ID repo if set, else from MODEL_DIR."""
    repo_id = os.environ.get("HF_REPO_ID")
    if repo_id:
        ckpt_path = Path(hf_hub_download(repo_id, "ckpt.pt"))
        tokenizer_path = Path(hf_hub_download(repo_id, "tokenizer.json"))
    else:
        model_dir = Path(os.environ.get("MODEL_DIR", "runs/base"))
        ckpt_path, tokenizer_path = model_dir / "ckpt.pt", model_dir / "tokenizer.json"
    model, _ = load_checkpoint(ckpt_path, get_device())
    return model.eval(), Tokenizer.load(tokenizer_path)


def create_app(loader: Loader = load_from_env) -> FastAPI:
    """Build the app; loader runs once at startup, and tests pass one that needs no files."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.tokenizer = loader()
        yield

    app = FastAPI(title="Ligature", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.post("/generate")
    def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        model: GPT = request.app.state.model
        tokenizer: Tokenizer = request.app.state.tokenizer
        # Every training story follows an <|endoftext|>, so the prompt starts after one.
        prompt_ids = [tokenizer.eot_id] + tokenizer.encode(body.prompt)
        room = model.config.block_size - len(prompt_ids)
        if room < 1:
            raise HTTPException(
                status_code=422,
                detail=f"prompt is {len(prompt_ids)} tokens; the limit is "
                f"{model.config.block_size - 1} to leave room for output",
            )

        device = next(model.parameters()).device
        idx = torch.tensor([prompt_ids], device=device)  # (1, T)
        start = time.perf_counter()
        out = model.generate(
            idx, min(body.max_new_tokens, room), temperature=body.temperature, top_k=body.top_k
        )
        seconds = time.perf_counter() - start

        new_tokens = out.size(1) - len(prompt_ids)
        story = out[0, 1:].tolist()  # drop the leading <|endoftext|>
        if tokenizer.eot_id in story:
            story = story[: story.index(tokenizer.eot_id)]  # the story ends at the next one
        return GenerateResponse(
            text=tokenizer.decode(story),
            new_tokens=new_tokens,
            tokens_per_sec=round(new_tokens / seconds, 1),
        )

    return app


app = create_app()
