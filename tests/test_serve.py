"""Tests for the serving layer.

No network and no GPU: these exercise the router, the cache and the response
contract, which is everything that can go wrong without a backend attached.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from argmap.schema import Component, Prediction, Relation
from argmap.serve.app import ExtractResponse, ResultCache, app, cache, prediction_to_response


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    cache.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health and metrics
# ---------------------------------------------------------------------------


def test_health_reports_per_route_availability(client: TestClient) -> None:
    """A bare 'ok' would let the UI offer a route that cannot work."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["routes"]) == {"claude", "local", "cascade"}
    assert all(isinstance(v, bool) for v in body["routes"].values())


def test_cascade_needs_both_backends(client: TestClient) -> None:
    routes = client.get("/health").json()["routes"]
    assert routes["cascade"] == (routes["claude"] and routes["local"])


def test_metrics_exposes_cache_stats(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert set(body["cache"]) >= {"entries", "hits", "misses", "hit_rate"}


def test_timing_header_is_present(client: TestClient) -> None:
    response = client.get("/health")
    assert float(response.headers["x-response-time-ms"]) >= 0


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_empty_text_is_rejected(client: TestClient) -> None:
    assert client.post("/extract", json={"text": ""}).status_code == 422


def test_whitespace_only_text_is_rejected(client: TestClient) -> None:
    assert client.post("/extract", json={"text": "   "}).status_code == 400


def test_unknown_route_is_rejected(client: TestClient) -> None:
    body = {"text": "Cars pollute.", "route": "telepathy"}
    assert client.post("/extract", json=body).status_code == 422


def test_unconfigured_route_returns_503(client: TestClient) -> None:
    """An unavailable backend is 503, not a 500 or a silent fallback.

    Falling back to another route would make the latency and quality numbers
    describe a different system than the one requested.
    """
    body = {"text": "Cars pollute. So cycle.", "route": "local"}
    response = client.post("/extract", json=body)
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_oversized_document_is_rejected(client: TestClient) -> None:
    from argmap.serve.app import MAX_DOCUMENT_CHARS

    body = {"text": "x" * (MAX_DOCUMENT_CHARS + 1)}
    assert client.post("/extract", json=body).status_code == 422


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------


def _response(model: str = "m") -> ExtractResponse:
    prediction = Prediction(
        doc_id="d",
        components=[Component(id="c1", type="Claim", start=0, end=5, text="Alpha")],
        relations=[],
    )
    return prediction_to_response(prediction, route="claude", model=model, latency_ms=1.0)


def test_cache_round_trips_and_counts() -> None:
    key = ResultCache.key("hello", "claude", None)
    assert cache.get(key) is None
    cache.put(key, _response())
    assert cache.get(key) is not None
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_key_separates_routes_and_models() -> None:
    """The comparison view asks for the same text on two routes at once."""
    base = ResultCache.key("hello", "claude", None)
    assert ResultCache.key("hello", "local", None) != base
    assert ResultCache.key("hello", "claude", "claude-sonnet-5") != base
    assert ResultCache.key("goodbye", "claude", None) != base


def test_cache_evicts_least_recently_used() -> None:
    small = ResultCache(capacity=2)
    small.put("a", _response())
    small.put("b", _response())
    small.get("a")  # 'a' becomes most recent, so 'b' should go first
    small.put("c", _response())
    assert small.get("b") is None
    assert small.get("a") is not None


def test_cached_responses_are_flagged() -> None:
    """A cached result must not be reported as a fresh measurement."""
    original = _response()
    assert original.cached is False
    assert original.model_copy(update={"cached": True}).cached is True


# ---------------------------------------------------------------------------
# Response contract
# ---------------------------------------------------------------------------


def test_response_carries_the_unified_schema() -> None:
    prediction = Prediction(
        doc_id="d",
        components=[
            Component(id="c1", type="Claim", start=0, end=5, text="Alpha"),
            Component(id="c2", type="Premise", start=6, end=10, text="beta"),
        ],
        relations=[Relation(id="r1", src="c2", tgt="c1", type="supports")],
    )
    response = prediction_to_response(prediction, route="claude", model="m", latency_ms=12.3)
    assert [c.id for c in response.components] == ["c1", "c2"]
    assert response.relations[0].type == "supports"
    assert response.latency_ms == 12.3
    assert response.escalated is None
