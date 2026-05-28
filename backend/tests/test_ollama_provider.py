from uuid import UUID

import pytest

from app.llm import ollama_provider
from app.retrieval.models import RetrievedChunk


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload: dict):
        self._payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, path: str, json: dict) -> _FakeResponse:
        self.calls.append((path, json))
        return _FakeResponse(self._payload)


@pytest.mark.asyncio
async def test_decide_route_parses_json(monkeypatch: pytest.MonkeyPatch):
    fake_client = _FakeAsyncClient(
        {"message": {"content": '{"route":"direct","confidence":0.9,"reasoning":"greeting"}'}}
    )
    monkeypatch.setattr(
        ollama_provider.httpx,
        "AsyncClient",
        lambda **_: fake_client,
    )

    decision = await ollama_provider.decide_route(
        "hello",
        base_url="http://localhost:11434",
        model="llama3.2",
    )
    assert decision.route == "direct"
    assert fake_client.calls[0][0] == "/api/chat"
    assert fake_client.calls[0][1]["format"] == "json"


@pytest.mark.asyncio
async def test_generate_from_context_returns_plain_text(monkeypatch: pytest.MonkeyPatch):
    fake_client = _FakeAsyncClient({"message": {"content": "Grounded answer from Ollama"}})
    monkeypatch.setattr(
        ollama_provider.httpx,
        "AsyncClient",
        lambda **_: fake_client,
    )

    answer = await ollama_provider.generate_from_context(
        query="What is PTO?",
        contexts=["PTO is accrued monthly."],
        route="single_hop_rag",
        base_url="http://localhost:11434",
        model="llama3.2",
    )
    assert answer == "Grounded answer from Ollama"
    assert "format" not in fake_client.calls[0][1]


@pytest.mark.asyncio
async def test_grade_retrieval_parses_structured_json(monkeypatch: pytest.MonkeyPatch):
    fake_client = _FakeAsyncClient(
        {"message": {"content": '{"relevant":true,"score":0.8,"should_abstain":false}'}}
    )
    monkeypatch.setattr(
        ollama_provider.httpx,
        "AsyncClient",
        lambda **_: fake_client,
    )

    grade = await ollama_provider.grade_retrieval(
        query="PTO",
        chunks=[
            RetrievedChunk(
                chunk_id=UUID("29ec4faa-6ff1-45de-b912-f8da99f0f0fd"),
                document_id=UUID("7a23ed96-874d-48d3-a0fb-7f48d87d5f88"),
                content="PTO policy chunk.",
                score=0.9,
            )
        ],
        threshold=0.25,
        base_url="http://localhost:11434",
        model="llama3.2",
    )
    assert grade.relevant is True
    assert grade.should_abstain is False


@pytest.mark.asyncio
async def test_decide_route_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch):
    fake_client = _FakeAsyncClient({"message": {"content": "not-json"}})
    monkeypatch.setattr(
        ollama_provider.httpx,
        "AsyncClient",
        lambda **_: fake_client,
    )

    with pytest.raises(ValueError):
        await ollama_provider.decide_route(
            "hello",
            base_url="http://localhost:11434",
            model="llama3.2",
        )
