"""1Password secret references for credentials, with a faked `op` CLI."""
import pickle
import subprocess

import httpx
import pytest

import sklearn_decision._secrets as secrets
from sklearn_decision import DecisionModelError, JevModel, QuestionFeaturizer, clear_secret_cache, noul

REF = "op://Private/Typesafe API/credential"
FAKE_KEY = "fake-key-for-tests"


@pytest.fixture
def fake_op(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ["read", "--no-newline"] and cmd[3] == REF:
            return subprocess.CompletedProcess(cmd, 0, stdout=FAKE_KEY, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="[ERROR] could not find item\n")

    monkeypatch.setenv("OP_CLI_PATH", "op")
    monkeypatch.setattr(secrets.subprocess, "run", run)
    clear_secret_cache()
    yield calls
    clear_secret_cache()


def test_plain_values_pass_through(fake_op):
    assert secrets.resolve_secret("literal") == "literal"
    assert secrets.resolve_secret(None) is None
    assert fake_op == []


def test_reference_resolved_once_and_cached(fake_op):
    assert secrets.resolve_secret(REF) == FAKE_KEY
    assert secrets.resolve_secret(REF) == FAKE_KEY
    assert fake_op == [["op", "read", "--no-newline", REF]]


def test_failure_is_fatal_and_names_only_the_reference(fake_op):
    with pytest.raises(DecisionModelError, match="could not find item") as e:
        secrets.resolve_secret("op://Private/Missing/credential")
    assert e.value.fatal


def test_missing_cli_is_explained(monkeypatch):
    monkeypatch.delenv("OP_CLI_PATH", raising=False)
    monkeypatch.setattr(secrets.shutil, "which", lambda name: None)
    clear_secret_cache()
    with pytest.raises(DecisionModelError, match="1Password CLI `op` was not found"):
        secrets.resolve_secret(REF)


@pytest.mark.parametrize("via", ["param", "env"])
def test_jev_model_sends_the_resolved_key(fake_op, monkeypatch, via):
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"noul": 0.5}}})

    if via == "env":
        monkeypatch.setenv("TYPESAFE_API_KEY", REF)
        model = JevModel(transport=httpx.MockTransport(handler))
    else:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        model = JevModel(api_key=REF, transport=httpx.MockTransport(handler))
    f = QuestionFeaturizer({"q": noul("x")}, model=model).fit(["a"])
    f.transform(["a"])
    assert seen == [f"Bearer {FAKE_KEY}"]
    # the secret lives only in the resolver's process cache (the test transport
    # holds a local function, which can't be pickled, so drop it first)
    f.model.transport = f.model_.transport = None
    for blob in (repr(f), repr(f.get_params(deep=True)), pickle.dumps(f)):
        assert FAKE_KEY not in (blob if isinstance(blob, str) else blob.decode("latin-1"))


def test_fit_does_not_resolve(fake_op):
    QuestionFeaturizer({"q": noul("x")}, model=JevModel(api_key=REF)).fit(["a"])
    assert fake_op == []
