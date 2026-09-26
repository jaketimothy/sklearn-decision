import pytest

TEXTS = [
    "Please refund my last invoice, the app crashed all week.",
    "How do I rotate my API key?",
    "Billing charged me twice this month.",
    "Love the new dashboard, no issues at all.",
    "The API returns 500 errors since this morning, urgent!",
    "Can I get an invoice copy for March?",
]


@pytest.fixture
def texts():
    return list(TEXTS)
