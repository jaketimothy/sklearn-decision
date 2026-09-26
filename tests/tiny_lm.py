"""A tiny random causal LM and word-level tokenizer, built in memory.

Lets TransformersModel run end to end in CI with no download. The answers are
meaningless; the tests check plumbing, not quality.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


NAME = "tiny/random-lm"
COMMIT = "0123456789abcdef0123456789abcdef01234567"

WORDS = """Text: Based only on the text above, is the following statement true? Answer Yes or No.
Statement: answer question by choosing exactly one option. Reply with option's letter only. Question: Options:
rate it on scale below. level's number Scale, from lowest to highest: yes no""".split()


def build():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = ["[UNK]", "[PAD]", "[EOS]"]
    vocab = {}
    for w in specials + [chr(ord("A") + i) for i in range(26)] + [str(i) for i in range(10)] + WORDS:
        vocab.setdefault(w, len(vocab))
    for i in range(200):  # filler words so states have distinct tokens
        vocab.setdefault(f"w{i}", len(vocab))
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                      pad_token_id=vocab["[PAD]"])
    model = LlamaForCausalLM(cfg).eval()
    return tok, model


def install(calls=None):
    """Register the tiny LM under NAME for every device/dtype, via the loader."""
    tok, model = build()

    def loader(name, revision, device, dtype, trust_remote_code):
        if calls is not None:
            calls.append(name)
        assert name == NAME
        return tok, model, COMMIT

    return loader


def texts(n):
    return [" ".join(f"w{(i * 7 + j) % 200}" for j in range(3 + i % 5)) for i in range(n)]
