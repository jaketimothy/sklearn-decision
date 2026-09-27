"""A frozen open-weights causal LM as a decision model, via logit readout.

The state goes first and the question after it; the answer is read from the
next-token logits, never from generated text:

* noul:   P(true) = softmax over the "Yes" / "No" answer tokens.
* choice: options are listed with letters A-Z; the distribution is a softmax
          over the letter tokens.
* score:  levels are numbered 0-9; the distribution is a softmax over the
          digit tokens.

Each answer string is matched with its common tokenizations (" Yes", "yes",
...), whose logits are combined with logsumexp. Nothing is sampled, so answers
are deterministic, and nothing leaves the machine.

This is the "frozen model + logit readout" recipe (e.g. Gemma 4 12B/31B
instruction-tuned checkpoints). Weights load lazily on the first model call
and are shared by every clone in the process.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .._answers import DecisionModelError, DistAnswer, NoulAnswer, Response
from .._cache import canon
from .base import Capabilities, DecisionModel

__all__ = ["TransformersModel", "DEFAULT_TEMPLATES"]

TEMPLATE_VERSION = 1
LETTERS = [chr(ord("A") + i) for i in range(26)]
DIGITS = [str(i) for i in range(10)]

DEFAULT_TEMPLATES = {
    "noul": (
        "Text:\n{state}\n\n"
        "Based only on the text above, is the following statement true? Answer Yes or No.\n"
        "Statement: {instructions}"
    ),
    "choice": (
        "Text:\n{state}\n\n"
        "Based only on the text above, answer the question by choosing exactly one option. "
        "Reply with the option's letter only.\n"
        "Question: {instructions}\n"
        "Options:\n{options}"
    ),
    "score": (
        "Text:\n{state}\n\n"
        "Based only on the text above, rate it on the scale below. Reply with the level's number only.\n"
        "Question: {instructions}\n"
        "Scale, from lowest to highest:\n{options}"
    ),
}
_PLAIN_SUFFIX = "\nAnswer:"  # for models without a chat template

# (name, revision, device, dtype, trust_remote_code) -> (tokenizer, model, commit)
_LOADED: dict[tuple, tuple[Any, Any, str]] = {}
_LOAD_LOCK = threading.Lock()


def load_pretrained(name: str, revision: str, device: str, dtype: str, trust_remote_code: bool):
    """Load a tokenizer and causal LM from the Hugging Face Hub or a local path.
    Returns (tokenizer, model, commit hash or revision)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("TransformersModel needs torch and transformers: "
                          "pip install 'sklearn-decision[local]'") from e
    tok = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    import transformers

    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in 4.56
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code,
                                                 **{dtype_kw: torch_dtype})
    model.to(device).eval()
    commit = getattr(model.config, "_commit_hash", None) or revision
    return tok, model, commit


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersModel(DecisionModel):
    """A frozen open-weights causal LM, answering by logit readout.

    Parameters
    ----------
    name : str
        Hugging Face repo id or local path of an instruction-tuned causal LM,
        e.g. "google/gemma-4-12b-it". Also reachable as
        ``model="hf:<repo id>[@revision]"``.
    revision : str
        Branch, tag or commit. Pin a commit hash for reproducible features:
        the cache is keyed on this string, and the resolved commit is
        recorded as each answer's version.
    device : str or None
        "cuda", "mps", "cpu"... None picks the best available.
    dtype : str
        "auto" (the checkpoint's dtype), "bfloat16", "float16" or "float32".
    batch_size : int
        Question prompts per forward pass. Rows run one at a time: each
        row's shared prefix is encoded once, then its questions run in
        batches against the cached prefix.
    max_state_tokens : int or None
        Truncate each state to this many tokens (keeping the start) before
        building prompts. Rows whose prompt is still longer than the model's
        context fail individually.
    use_chat_template : {"auto", True, False}
        Wrap the prompt as a user turn with the tokenizer's chat template
        ("auto": when the tokenizer has one). Otherwise "\\nAnswer:" is
        appended to the plain prompt.
    chat_template_kwargs : dict or None
        Extra arguments for ``apply_chat_template``, e.g.
        ``{"enable_thinking": False}`` for templates with a thinking mode.
    answer_prefix : str
        Text placed after the generation prompt, before the answer token,
        e.g. an empty think block for models that insist on one.
    templates : dict or None
        Overrides for the "noul" / "choice" / "score" prompt templates, with
        ``{state}``, ``{instructions}`` and (choice/score) ``{options}``
        fields. They are part of the cache namespace.
    trust_remote_code : bool
        Passed to ``from_pretrained``. Leave False unless you have read the code.

    Notes
    -----
    * Answers have no native confidence, so ``predict_confidence`` is not
      offered; probabilities are raw readouts and should be recalibrated
      (``CalibratedClassifierCV``) before thresholding.
    * Choice questions are limited to 26 options (letters) and score rubrics
      to 10 levels (digits); ``ChoiceEncoder`` blocks larger codebooks.
    * The state is encoded once per row and shared by all its questions
      through the key/value cache; each question then costs only its own
      tokens (the embeddings report's "L_text + M x L_question").
    """

    def __init__(self, name: str = "", *, revision: str = "main", device: str | None = None, dtype: str = "auto",
                 batch_size: int = 8, max_state_tokens: int | None = None, use_chat_template="auto",
                 chat_template_kwargs: Mapping | None = None, answer_prefix: str = "",
                 templates: Mapping[str, str] | None = None, trust_remote_code: bool = False):
        self.name = name
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_state_tokens = max_state_tokens
        self.use_chat_template = use_chat_template
        self.chat_template_kwargs = chat_template_kwargs
        self.answer_prefix = answer_prefix
        self.templates = templates
        self.trust_remote_code = trust_remote_code

    # ---------------- DecisionModel API ----------------

    def resolve(self) -> TransformersModel:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a Hugging Face repo id or a local path")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("batch_size must be an int >= 1")
        if self.max_state_tokens is not None and (not isinstance(self.max_state_tokens, int)
                                                  or self.max_state_tokens < 1):
            raise ValueError("max_state_tokens must be an int >= 1 or None")
        if self.use_chat_template not in ("auto", True, False):
            raise ValueError("use_chat_template must be 'auto', True or False")
        if self.dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be 'auto', 'bfloat16', 'float16' or 'float32', got {self.dtype!r}")
        unknown = set(self.templates or {}) - set(DEFAULT_TEMPLATES)
        if unknown:
            raise ValueError(f"templates has unknown keys {sorted(unknown)}; use noul/choice/score")
        return self

    def cache_namespace(self) -> str:
        prompt = canon({"v": TEMPLATE_VERSION, "t": self._templates(), "chat": self.use_chat_template,
                        "chat_kw": self.chat_template_kwargs, "prefix": self.answer_prefix,
                        "max_state": self.max_state_tokens})
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"hf:{self.name}@{self.revision}:{self.dtype}:{digest}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            max_choice_options=len(LETTERS),
            max_score_levels=len(DIGITS),
            max_questions_per_call=None,
            native_confidence=False,
            deterministic=True,
        )

    def answer(self, items: Sequence[tuple[Any, Mapping[str, dict]]]) -> list[Response | BaseException]:
        if not items:
            return []
        tok, model, commit = self._load()
        version = f"{self.name}@{commit}"
        answer_ids = self._answer_ids(tok)

        # rows run one at a time: the row's shared prefix (chat header + state) is
        # encoded once, then its questions' suffixes run in batches against it
        answers: dict[int, dict] = {}
        errors: dict[int, BaseException] = {}
        for i, (state, questions) in enumerate(items):
            try:
                text = self._state_text(tok, state)
                specs = list(questions.items())
                rows = self._row_logits(tok, model, [self._prompt(tok, text, spec) for _, spec in specs])
                answers[i] = {q: _readout(spec, row, answer_ids) for (q, spec), row in zip(specs, rows)}
            except Exception as e:  # this row fails; the others go on
                errors[i] = e
        out: list[Response | BaseException] = []
        for i in range(len(items)):
            if i in errors:
                err = errors[i]
                out.append(err if isinstance(err, DecisionModelError)
                           else DecisionModelError(f"{type(err).__name__}: {err}"))
            else:
                out.append(Response(answers.get(i, {}), version))
                self.usage["calls"] += 1
                self.usage["answers_fetched"] += len(answers.get(i, {}))
        self.versions_seen.add(version)
        return out

    def estimate_cost(self, states, requests) -> dict:
        n_q = sum(len(r) for r in requests)
        return {"rows": len(states), "requests": len(states) * len(requests),
                "forward_passes": len(states) * n_q, "est_cost_usd": 0.0}

    # ---------------- internals ----------------

    def _templates(self) -> dict[str, str]:
        return {**DEFAULT_TEMPLATES, **dict(self.templates or {})}

    def _load(self):
        device = self.device or _default_device()
        key = (self.name, self.revision, device, self.dtype, self.trust_remote_code)
        with _LOAD_LOCK:
            if key not in _LOADED:
                tok, model, commit = load_pretrained(*key)
                if tok.pad_token_id is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                _LOADED[key] = (tok, model, str(commit))
            return _LOADED[key]

    def _answer_ids(self, tok) -> dict[str, list[int]]:
        """Token ids for every answer string, over its common spellings."""
        out = {}
        for s in ["Yes", "No", *LETTERS, *DIGITS]:
            # lower-case spellings only for yes/no: "a" is a word, not option A
            spellings = {s, " " + s} | ({s.lower(), " " + s.lower()} if s in ("Yes", "No") else set())
            ids = set()
            for v in spellings:
                enc = tok.encode(v, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
            if not ids:
                raise DecisionModelError(f"{self.name}: no single-token encoding for answer {s!r}", fatal=True)
            out[s] = sorted(ids)
        return out

    def _use_chat(self, tok) -> bool:
        if self.use_chat_template == "auto":
            return bool(getattr(tok, "chat_template", None))
        return bool(self.use_chat_template)

    def _state_text(self, tok, state) -> str:
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
        if self.max_state_tokens is not None:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > self.max_state_tokens:
                text = tok.decode(ids[: self.max_state_tokens])
        return text

    def _prompt(self, tok, state_text: str, spec: Mapping) -> str:
        t = spec["type"]
        fields = {"state": state_text, "instructions": spec["instructions"]}
        if t == "choice":
            lines = []
            for letter, (label, desc) in zip(LETTERS, spec["criteria"].items()):
                lines.append(f"{letter}. {label}" + (f": {desc}" if desc else ""))
            fields["options"] = "\n".join(lines)
        elif t == "score":
            fields["options"] = "\n".join(f"{i}. {lv}" for i, lv in enumerate(spec["criteria"]))
        body = self._templates()[t].format(**fields)
        if self._use_chat(tok):
            prompt = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                             add_generation_prompt=True, **dict(self.chat_template_kwargs or {}))
        else:
            prompt = body + _PLAIN_SUFFIX
        return prompt + self.answer_prefix

    def _row_logits(self, tok, model, prompts: list[str]) -> list:
        """Final-position logits for one row's prompts.

        The prompts share everything up to the last line break before they
        diverge (chat header and state, with the default templates). That
        prefix is encoded once and its key/value cache reused for every
        question's suffix, so the state costs one forward pass per row
        instead of one per question.
        """
        import torch

        add_special = not self._use_chat(tok)  # a rendered chat template already has its BOS
        cut = os.path.commonprefix(prompts).rfind("\n") + 1
        while cut > 0 and any(len(p) <= cut for p in prompts):  # every suffix needs a token
            cut = prompts[0].rfind("\n", 0, cut - 1) + 1
        pre = tok.encode(prompts[0][:cut], add_special_tokens=add_special) if cut else []
        sufs = [tok.encode(p[cut:], add_special_tokens=add_special and not cut) for p in prompts]
        limit = getattr(model.config, "max_position_embeddings", None)
        longest = len(pre) + max(len(x) for x in sufs)
        if limit is not None and longest > limit:
            raise DecisionModelError(f"prompt has {longest} tokens, over the model's {limit}; set max_state_tokens")

        device = next(model.parameters()).device
        pad = tok.pad_token_id
        n_pre = len(pre)
        results: list = [None] * len(prompts)
        with torch.inference_mode():
            past = None
            if pre:
                past = model(input_ids=torch.tensor([pre], device=device), use_cache=True).past_key_values
                self.usage["input_tokens"] += n_pre
            order = sorted(range(len(sufs)), key=lambda k: len(sufs[k]))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                b, width = len(batch), max(len(sufs[k]) for k in batch)
                ids = torch.full((b, width), pad, dtype=torch.long)
                mask = torch.zeros((b, n_pre + width), dtype=torch.long)
                mask[:, :n_pre] = 1
                for r, k in enumerate(batch):  # right-padded: real tokens never attend to the padding
                    ids[r, : len(sufs[k])] = torch.tensor(sufs[k], dtype=torch.long)
                    mask[r, n_pre : n_pre + len(sufs[k])] = 1
                pos = (n_pre + torch.arange(width)).expand(b, width)
                kw = {"past_key_values": _repeat_cache(past, b), "use_cache": True} if past is not None else {}
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device), position_ids=pos.to(device),
                            **kw)
                last = out.logits.float().cpu()
                for r, k in enumerate(batch):
                    results[k] = last[r, len(sufs[k]) - 1]
                self.usage["input_tokens"] += int(mask[:, n_pre:].sum())
        return results


def _repeat_cache(past, b: int):
    """A copy of a batch-1 key/value cache, repeated to batch size b."""
    if isinstance(past, tuple):  # legacy tuple-of-tensors format
        return tuple(tuple(t.repeat_interleave(b, dim=0) for t in layer) for layer in past)
    cache = copy.deepcopy(past)
    cache.batch_repeat_interleave(b)
    return cache


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _readout(spec: Mapping, row, answer_ids: dict[str, list[int]]):
    """Answer for one question from its final-position logits."""
    if row is None:
        raise DecisionModelError("no logits for this prompt")
    t = spec["type"]
    if t == "noul":
        yes = _logsumexp([float(row[i]) for i in answer_ids["Yes"]])
        no = _logsumexp([float(row[i]) for i in answer_ids["No"]])
        return NoulAnswer(1 / (1 + math.exp(no - yes)))
    labels = LETTERS[: len(spec["criteria"])] if t == "choice" else DIGITS[: len(spec["criteria"])]
    z = [_logsumexp([float(row[i]) for i in answer_ids[lb]]) for lb in labels]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    return DistAnswer(tuple(v / tot for v in e), None)
