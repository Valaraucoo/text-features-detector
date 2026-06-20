"""Offline tests for retry/backoff helpers in the Pydantic AI judge."""

from text_features_detector.judges import judge


def test_retryable_error_detects_rate_limit():
    assert judge._is_retryable_error(Exception("status_code: 429 rate_limit_exceeded"))


def test_retryable_error_detects_timeout():
    assert judge._is_retryable_error(Exception("request timed out"))


def test_non_retryable_error_not_retried():
    assert not judge._is_retryable_error(Exception("invalid api key"))


def test_backoff_delay_increases(monkeypatch):
    monkeypatch.setattr(judge.random, "uniform", lambda *_: 0.0)
    assert judge._backoff_delay(1) > judge._backoff_delay(0)
