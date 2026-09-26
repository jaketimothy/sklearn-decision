"""JevModel against a mock of the documented /v1/systemone wire format."""
import json
import os

import httpx
import numpy as np
import pytest

import sklearn_decision.models.jev as jev_mod
from sklearn_decision import (
    ChoiceClassifier,
    DecisionModelError,
    FakeModel,
    JevAPIError,
    JevModel,
    QuestionFeaturizer,
    choice,
    noul,
    resolve_model,
    score,
)

BANK = {
    "refund": noul("Asks for money back."),
    "product": choice("Which product?", ["app", "api"]),
    "urgency": score("How urgent?", ["low", "mid", "high"]),
}


def wire_answer(spec):
    if spec["type"] == "noul":
        return {"noul": 0.8}
    if spec["type"] == "choice":
        labels = list(spec["criteria"])
        return {"probabilities": {lb: 1 / len(labels) for lb in labels}, "confidence": 0.5}
    return {"probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}, "confidence": 0.7}


class Server:
    """Records requests; ``script`` maps call number -> (status, headers)."""

    def __init__(self, script=None):
        self.requests = []
        self.script = script or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        status, headers = self.script.get(len(self.requests), (200, {}))
        if status != 200:
            return httpx.Response(status, headers=headers, text="nope")
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {q: wire_answer(s) for q, s in body["questions"].items()},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(jev_mod, "_sleep", instant)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)


def mock(server):
    # estimators clone (deep-copy) their model, transport included; a function
    # survives deepcopy by reference, so every copy still reports to `server`
    return httpx.MockTransport(lambda request: server(request))


def model(server, **kw):
    kw.setdefault("api_key", "test-key")
    return JevModel("jev-1.13", transport=mock(server), **kw)


def feat(server, **kw):
    return QuestionFeaturizer(BANK, model=model(server, **kw.pop("model_kw", {})), cache_path=None,
                              score_repr="both", include_confidence=True, **kw)


def test_payload_shape_auth_and_normalization():
    srv = Server()
    f = feat(srv).fit(["hello"])
    X = f.transform(["hello"])
    req, body = srv.requests[0]
    assert req.url == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer test-key"
    assert body == {"model": "jev-1.13", "state": "hello", "questions": BANK}
    names = list(f.get_feature_names_out())
    row = dict(zip(names, X[0]))
    assert row["refund"] == 0.8
    assert row["product__app"] == 0.5 and row["product__confidence"] == 0.5
    assert [row[f"urgency__L{i}"] for i in range(3)] == [0.2, 0.3, 0.5]  # index-keyed levels
    assert row["urgency__ev"] == pytest.approx(1.3)
    assert f.model_.usage["input_tokens"] == 10
    assert f.model_.versions_seen == {"jev-1.13.0"}


def test_chunking_by_max_questions_per_call():
    srv = Server()
    feat(srv, model_kw={"max_questions_per_call": 2}).fit(["a"]).transform(["a", "b"])
    assert sorted(len(b["questions"]) for _, b in srv.requests) == [1, 1, 2, 2]


def test_retries_honour_retry_after(monkeypatch):
    waits = []

    async def record(s):
        waits.append(s)

    monkeypatch.setattr(jev_mod, "_sleep", record)
    srv = Server({1: (429, {"retry-after": "2"}), 2: (503, {})})
    X = feat(srv).fit(["a"]).transform(["a"])
    assert len(srv.requests) == 3 and not np.isnan(X).any()
    assert 2 <= waits[0] <= 2.5 and 1 <= waits[1] <= 1.25  # Retry-After, then 0.5 * 2**1 backoff


def test_client_errors_fail_fast():
    srv = Server({1: (400, {})})
    f = feat(srv).fit(["a"])
    with pytest.raises(JevAPIError) as e:
        f.transform(["a"])
    assert e.value.status == 400 and not e.value.fatal and len(srv.requests) == 1


def test_auth_errors_are_fatal_even_with_on_error_nan():
    srv = Server({1: (401, {})})
    f = feat(srv, on_error="nan").fit(["a"])
    with pytest.raises(JevAPIError, match="401"):
        f.transform(["a"])


def test_retries_exhausted_gives_nan_with_on_error_nan():
    srv = Server({i: (500, {}) for i in range(1, 10)})
    f = feat(srv, on_error="nan", model_kw={"max_retries": 2}).fit(["a"])
    with pytest.warns(UserWarning, match="failed"):
        X = f.transform(["a"])
    assert np.isnan(X).all() and len(srv.requests) == 3


def test_successes_cached_before_error_raised():
    srv = Server({2: (400, {})})
    f = feat(srv, model_kw={"max_concurrency": 1}).fit(["a"])
    with pytest.raises(JevAPIError):
        f.transform(["a", "b"])
    n = len(srv.requests)
    f.transform(["a"])
    assert len(srv.requests) == n


def test_missing_api_key_raises_at_call_time_not_fit(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    srv = Server()
    f = QuestionFeaturizer(BANK, model=JevModel(transport=mock(srv)), cache_path=None,
                           on_error="nan").fit(["a"])
    with pytest.raises(DecisionModelError, match="TYPESAFE_API_KEY"):
        f.transform(["a"])
    assert not srv.requests


def test_base_url_override():
    srv = Server()
    feat(srv, model_kw={"base_url": "https://openrouter.ai/api/"}).fit(["a"]).transform(["a"])
    assert str(srv.requests[0][0].url) == "https://openrouter.ai/api/v1/systemone"


def test_malformed_response_is_a_per_row_error():
    def handler(request):
        return httpx.Response(200, json={"answers": {}})

    f = QuestionFeaturizer(BANK, model=JevModel(api_key="k", transport=httpx.MockTransport(handler)),
                           cache_path=None, on_error="nan").fit(["a"])
    with pytest.warns(UserWarning):
        assert np.isnan(f.transform(["a"])).all()


def test_capabilities_enforced():
    big = choice("x", [str(i) for i in range(256)])
    with pytest.raises(ValueError, match="limit of 255"):
        QuestionFeaturizer({"q": big}, cache_path=None).fit(["a"])


# ---------------- registry ----------------

def test_registry_resolution():
    assert isinstance(resolve_model("jev-1.13"), JevModel)
    assert resolve_model("jev-1.13").name == "jev-1.13"
    assert isinstance(resolve_model("fake-2"), FakeModel)
    with pytest.raises(ValueError, match="Registered prefixes"):
        resolve_model("gpt-9")
    with pytest.raises(TypeError):
        resolve_model(42)
    m = JevModel("jev-1.13", timeout=5)
    r = resolve_model(m)
    assert r is not m and r.timeout == 5


def test_latest_warns():
    with pytest.warns(UserWarning, match="not pinned"):
        resolve_model("jev-latest")


def test_invalid_jev_params():
    with pytest.raises(ValueError, match="max_concurrency"):
        resolve_model(JevModel(max_concurrency=0))


# ---------------- live smoke test ----------------

@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="needs TYPESAFE_API_KEY")
def test_live_smoke(tmp_path):
    texts = ["Refund my order please.", "How do I reset my API key?", "Great service!"]
    f = QuestionFeaturizer(BANK, model="jev-1.13", cache_path=str(tmp_path / "live.sqlite"),
                           score_repr="both", include_confidence=True).fit(texts)
    X = f.transform(texts)
    assert X.shape == (3, len(f.get_feature_names_out())) and not np.isnan(X).any()
    np.testing.assert_allclose(X[:, f.feature_groups_["product"][:2]].sum(axis=1), 1, atol=1e-3)
    clf = ChoiceClassifier("Which product is discussed?", {"app": None, "api": None},
                           featurizer=QuestionFeaturizer(cache_path=str(tmp_path / "live.sqlite"))).fit(texts)
    assert set(clf.predict(texts)) <= {"app", "api"}
