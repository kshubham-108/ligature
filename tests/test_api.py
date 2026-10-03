from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import create_app, load_from_env
from ligature.model import GPT, GPTConfig
from ligature.tokenizer import Tokenizer

BLOCK_SIZE = 32


@pytest.mark.parametrize("weights_file", ["model.pt", "ckpt.pt"])
def test_load_from_env_reads_either_weights_file(
    weights_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    tokenizer = Tokenizer(merges=[])
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=16, n_layer=1, n_head=2, d_model=16
    )
    model = GPT(config)
    torch.save(
        {"model": model.state_dict(), "model_config": asdict(config)}, tmp_path / weights_file
    )
    tokenizer.save(tmp_path / "tokenizer.json")
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    loaded, loaded_tokenizer = load_from_env()
    assert loaded.config == config
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size


def test_load_from_env_prefers_the_weights_only_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = Tokenizer(merges=[])
    Tokenizer(merges=[]).save(tmp_path / "tokenizer.json")
    for name, d_model in (("model.pt", 16), ("ckpt.pt", 32)):
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, block_size=16, n_layer=1, n_head=2, d_model=d_model
        )
        torch.save(
            {"model": GPT(config).state_dict(), "model_config": asdict(config)}, tmp_path / name
        )
    monkeypatch.delenv("HF_REPO_ID", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert load_from_env()[0].config.d_model == 16


@pytest.fixture
def client() -> Iterator[TestClient]:
    """The app with a tiny random model and a merge-free byte tokeniser: nothing to download."""
    torch.manual_seed(0)
    tokenizer = Tokenizer(merges=[])
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE, n_layer=1, n_head=2, d_model=16
    )
    model = GPT(config).eval()
    with TestClient(create_app(loader=lambda: (model, tokenizer))) as client:
        yield client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<textarea" in response.text


def test_generate_returns_text_and_speed(client: TestClient) -> None:
    response = client.post("/generate", json={"prompt": "Once", "max_new_tokens": 5})
    assert response.status_code == 200
    data = response.json()
    assert data["text"].startswith("Once")
    assert data["new_tokens"] == 5
    assert data["tokens_per_sec"] > 0


def test_max_new_tokens_is_capped_to_block_size(client: TestClient) -> None:
    # The byte tokeniser makes "abc" three tokens, after the leading <|endoftext|>.
    response = client.post("/generate", json={"prompt": "abc", "max_new_tokens": 1000})
    assert response.status_code == 200
    assert response.json()["new_tokens"] == BLOCK_SIZE - 4


def test_greedy_generation_is_repeatable(client: TestClient) -> None:
    request = {"prompt": "The cat", "max_new_tokens": 10, "temperature": 0}
    first = client.post("/generate", json=request).json()["text"]
    second = client.post("/generate", json=request).json()["text"]
    assert first == second


@pytest.mark.parametrize(
    "request_body",
    [
        {"prompt": "x" * BLOCK_SIZE},  # fills block_size, leaving no room for output
        {"prompt": "Once", "max_new_tokens": 0},
        {"prompt": "Once", "temperature": -1},
        {"prompt": "Once", "top_k": 0},
    ],
    ids=["prompt-too-long", "zero-tokens", "negative-temperature", "zero-top-k"],
)
def test_invalid_requests_are_rejected(client: TestClient, request_body: dict) -> None:
    assert client.post("/generate", json=request_body).status_code == 422
