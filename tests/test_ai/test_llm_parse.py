"""LLMAgent._parse_response — the untrusted boundary with an external LLM.

The model can return anything: fenced JSON, out-of-range numbers, wrong-typed
fields, or outright garbage. Every malformed case must fail *safe* to a
non-actionable HOLD (score/confidence 0, error set) — never raise, and never
let a bad response turn into a tradeable signal.
"""
from ai.llm_agent import LLMAgent
from config.constants import Signal


def _agent() -> LLMAgent:
    a = LLMAgent.__new__(LLMAgent)   # skip __init__ (no API key / client needed)
    a.mock = True
    return a


def _parse(resp: str):
    return _agent()._parse_response(resp, "BTC/USDT")


class TestValidResponses:
    def test_plain_json(self):
        a = _parse('{"score": 0.5, "signal": "BUY", "confidence": 0.7}')
        assert a.signal == Signal.BUY and a.score == 0.5 and a.confidence == 0.7
        assert a.error is None and a.mock is False

    def test_markdown_fenced_json(self):
        a = _parse('```json\n{"score": -0.4, "signal": "SELL", "confidence": 0.8}\n```')
        assert a.signal == Signal.SELL and a.score == -0.4

    def test_score_and_confidence_are_clamped(self):
        a = _parse('{"score": 9.9, "signal": "BUY", "confidence": 5.0}')
        assert a.score == 1.0 and a.confidence == 1.0
        b = _parse('{"score": -9.9, "signal": "SELL", "confidence": -1.0}')
        assert b.score == -1.0 and b.confidence == 0.0

    def test_unknown_signal_string_becomes_hold(self):
        a = _parse('{"score": 0.1, "signal": "MOON", "confidence": 0.9}')
        assert a.signal == Signal.HOLD


class TestMalformedFailsSafe:
    def _assert_safe_hold(self, a):
        assert a.signal == Signal.HOLD
        assert a.score == 0.0 and a.confidence == 0.0
        assert a.error is not None          # recorded, not swallowed
        assert a.mock is False              # a clean parse-fail, not a mock signal

    def test_garbage_text(self):
        self._assert_safe_hold(_parse("not json at all"))

    def test_null_score_does_not_raise(self):
        # Regression: {"score": null} → float(None) → TypeError used to escape.
        self._assert_safe_hold(_parse('{"score": null, "signal": "BUY", "confidence": 0.7}'))

    def test_numeric_signal_does_not_raise(self):
        # Regression: {"signal": 5} → int.upper() → AttributeError used to escape.
        self._assert_safe_hold(_parse('{"score": 0.5, "signal": 5, "confidence": 0.7}'))

    def test_empty_object_defaults_to_hold(self):
        a = _parse("{}")
        assert a.signal == Signal.HOLD  # missing fields fall back to safe defaults
