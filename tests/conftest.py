import pytest


@pytest.fixture(autouse=True)
def _allow_missing_dev_knowledge_for_unit_tests(monkeypatch):
    monkeypatch.setenv("KSEARCH_ALLOW_MISSING_DEV_KNOWLEDGE", "1")
