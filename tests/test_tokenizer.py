import pytest

from ligature.tokenizer import EOT, Tokenizer

CORPUS = (
    "Once upon a time, there was a little girl named Lily. She loved to play in the sun.\n"
    'One day, she saw a big red ball in the park. "Can I play with it?" she asked her mum.\n'
    "The café sold crème brûlée, and the naïve dragon wanted some.\n"
) * 20


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=320)


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world! It's 2026 and the cat's 3 toys    are here.\n\tTabs too.",
        "Crème brûlée à la carte, naïve façade, Zürich, smørrebrød.",
        "The dragon \U0001f409 smiled \U0001f642 at the family \U0001f468‍\U0001f469‍\U0001f467.",
        "",
    ],
    ids=["ascii", "accented", "emoji", "empty"],
)
def test_round_trip_is_exact(tokenizer: Tokenizer, text: str) -> None:
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids).encode("utf-8") == text.encode("utf-8")


def test_empty_string_encodes_to_no_ids(tokenizer: Tokenizer) -> None:
    assert tokenizer.encode("") == []


def test_special_token_is_one_id_and_survives_decode(tokenizer: Tokenizer) -> None:
    assert tokenizer.encode(EOT) == [tokenizer.eot_id]
    text = f"The end.{EOT}Once upon a time"
    ids = tokenizer.encode(text)
    assert ids.count(tokenizer.eot_id) == 1
    assert tokenizer.decode(ids) == text


def test_training_is_deterministic() -> None:
    first = Tokenizer.train(CORPUS, vocab_size=300)
    second = Tokenizer.train(CORPUS, vocab_size=300)
    assert list(first.merges) == list(second.merges)


def test_first_merges_match_hand_count() -> None:
    # Chunks: "low", " lower", " lowest". Byte ids: ' '=32, 'e'=101, 'l'=108, 'o'=111, 'w'=119.
    # 1. (l,o)=3 and (o,w)=3 tie; the smaller pair (108, 111) wins -> 256 "lo".
    # 2. (256,w)=3 beats ( ,256)=2 and (w,e)=2 -> 257 "low".
    # 3. ( ,257)=2 and (257,e)=2 tie; the smaller pair (32, 257) wins -> 258 " low".
    tok = Tokenizer.train("low lower lowest", vocab_size=256 + 3 + 1)
    assert list(tok.merges) == [(108, 111), (256, 119), (32, 257)]
    assert [tok.vocab[i] for i in (256, 257, 258)] == [b"lo", b"low", b" low"]


def test_vocab_size_is_respected() -> None:
    tok = Tokenizer.train(CORPUS, vocab_size=300)
    assert tok.vocab_size == 300
    assert tok.eot_id == 299
    assert max(tok.encode(CORPUS + EOT)) < 300


def test_save_and_load_round_trip(tokenizer: Tokenizer, tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    assert loaded.special_tokens == tokenizer.special_tokens
    assert loaded.encode(CORPUS) == tokenizer.encode(CORPUS)
