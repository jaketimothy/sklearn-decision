import pytest

from sklearn_decision import JevModel, clear_memory_cache

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
    """Tests count model calls and spending, so each starts with an empty
    in-memory cache and no Jev spend."""
    clear_memory_cache()
    JevModel.reset_process_spend()
    yield
    clear_memory_cache()
    JevModel.reset_process_spend()


@pytest.fixture
def texts():
    return list(TEXTS)
