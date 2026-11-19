"""LLM wrapper: answers are cached, and a gateway that keeps failing stops being called."""

from spendcube.llm import LLM


class FlakyLLM(LLM):
    name = "flaky"

    def __init__(self, cache_dir, fail: bool):
        super().__init__({"model": "m", "retries": 0, "max_consecutive_failures": 2}, cache_dir)
        self.fail, self.sent = fail, 0

    def _call(self, prompt, schema):
        self.sent += 1
        if self.fail:
            raise RuntimeError("401 unauthorized")
        return '{"ok": true}', {"input_tokens": 100, "output_tokens": 10}


def test_answers_are_cached(tmp_path):
    llm = FlakyLLM(tmp_path, fail=False)
    assert llm.complete_json("p", {}) == {"ok": True}
    assert llm.complete_json("p", {}) == {"ok": True}
    assert llm.sent == 1 and llm.usage.cache_hits == 1 and llm.usage.input_tokens == 100


def test_circuit_opens_after_repeated_failures(tmp_path):
    FlakyLLM(tmp_path, fail=False).complete_json("cached", {})
    llm = FlakyLLM(tmp_path, fail=True)
    assert llm.complete_json("a", {}) is None
    assert llm.complete_json("b", {}) is None  # second failure in a row: stop calling
    assert llm.complete_json("c", {}) is None
    assert llm.sent == 2 and llm.usage.skipped == 1 and "401" in llm.disabled_reason
    assert llm.complete_json("cached", {}) == {"ok": True}  # the cache still answers
