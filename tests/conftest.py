import pytest

from sklearn_decision import clear_memory_cache

TEXTS = [
    "Please refund my last invoice, the app crashed all week.",
    "How do I rotate my API key?",
    "Billing charged me twice this month.",
    "Love the new dashboard, no issues at all.",
    "The API returns 500 errors since this morning, urgent!",
    "Can I get an invoice copy for March?",
]


@pytest.fixture(autouse=True)
def fresh_memory_cache():
    """Tests count model calls, so each starts with an empty in-memory cache."""
    clear_memory_cache()
    yield
    clear_memory_cache()


@pytest.fixture
def texts():
    return list(TEXTS)
