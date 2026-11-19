"""LLM access with an on-disk cache, retries and usage accounting.

Two backends share one call, `complete_json(prompt, schema) -> dict | None`:

- codex: runs the Codex CLI (`codex exec`). AgentRouter only accepts known coding-agent
  clients, so this is how the tool reaches it. Codex runs read-only in an empty
  folder with its own CODEX_HOME, so it sees nothing but the prompt.
- openai: any OpenAI-compatible endpoint, for whoever has a direct key.

A failed call returns None; the caller falls back to the local classifier for those items.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    skipped: int = 0  # calls not made after the gateway kept failing
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens
    reasoning_output_tokens: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


class LLM:
    def __init__(self, cfg: dict, cache_dir: Path):
        self.cfg = cfg
        self.model = cfg["model"]
        self.cache_dir = cache_dir / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self.disabled_reason: str | None = None

    name = "base"

    def complete_json(self, prompt: str, schema: dict) -> dict | None:
        cache_file = self._cache_file(prompt, schema)
        if cache_file.exists():
            with self._lock:
                self.usage.cache_hits += 1
            cached = json.loads(cache_file.read_text())
            return cached["result"] if "result" in cached and "usage" in cached else cached
        if self.disabled_reason:  # circuit open: don't keep calling a gateway that refuses us
            with self._lock:
                self.usage.skipped += 1
            return None

        for attempt in range(self.cfg["retries"] + 1):
            start = time.time()
            try:
                text, tokens = self._call(prompt, schema)
                result = json.loads(text)
            except Exception as exc:  # network, timeout, malformed JSON: retry, then give up
                error = exc
                time.sleep(2 ** attempt)
                continue
            finally:
                with self._lock:
                    self.usage.seconds += time.time() - start
            with self._lock:
                self.usage.calls += 1
                self._consecutive_failures = 0
                for key, value in tokens.items():
                    setattr(self.usage, key, getattr(self.usage, key) + value)
            cache_file = self._cache_file(prompt, schema, for_write=True)
            cache_file.write_text(json.dumps({"result": result, "usage": tokens}))
            return result
        with self._lock:
            self.usage.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.cfg["max_consecutive_failures"] and not self.disabled_reason:
                self.disabled_reason = str(error)[:300]
                print(f"[llm] {self._consecutive_failures} calls failed in a row; no more LLM calls this run. "
                      f"Last error: {self.disabled_reason}")
        print(f"[llm] call failed after retries: {str(error)[:300]}")
        return None

    def _cache_file(self, prompt: str, schema: dict, for_write: bool = False) -> Path:
        """Cache key covers everything that changes the answer: model, reasoning effort, prompt, schema.

        Answers cached before the effort was part of the key were all made at effort 'low';
        they are still found under their old key.
        """
        effort = self.cfg.get("reasoning_effort")
        key = hashlib.sha256(json.dumps([self.model, effort, prompt, schema], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if not for_write and not path.exists() and effort == "low":
            legacy = hashlib.sha256(json.dumps([self.model, prompt, schema], sort_keys=True).encode()).hexdigest()
            if (self.cache_dir / f"{legacy}.json").exists():
                return self.cache_dir / f"{legacy}.json"
        return path

    def _call(self, prompt: str, schema: dict) -> tuple[str, dict]:
        raise NotImplementedError


class CodexLLM(LLM):
    name = "codex"

    def __init__(self, cfg: dict, cache_dir: Path):
        super().__init__(cfg, cache_dir)
        self.home = REPO_ROOT / ".cache" / "codex_home"
        self.workdir = REPO_ROOT / ".cache" / "codex_work"
        self.home.mkdir(parents=True, exist_ok=True)
        self.workdir.mkdir(parents=True, exist_ok=True)

    def _command(self, schema_file: Path, out_file: Path) -> list[str]:
        provider = (f'{{name="gateway", base_url="{self.cfg["base_url"]}", env_key="{self.cfg["api_key_env"]}", '
                    'wire_api="responses", request_max_retries=0, stream_max_retries=0}')
        cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
               "--sandbox", "read-only", "-C", str(self.workdir),
               "-c", "model_provider=gateway", "-c", f"model_providers.gateway={provider}",
               "-c", f'model="{self.model}"',
               "-c", f'model_reasoning_effort="{self.cfg["reasoning_effort"]}"']
        if self.cfg.get("instructions_file"):
            cmd += ["-c", f'model_instructions_file="{REPO_ROOT / self.cfg["instructions_file"]}"']
        for feature in self.cfg.get("disable_features", []):
            cmd += ["--disable", feature]
        return cmd + ["--output-schema", str(schema_file), "-o", str(out_file), "--json", "-"]

    def _call(self, prompt: str, schema: dict) -> tuple[str, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            schema_file, out_file = Path(tmp) / "schema.json", Path(tmp) / "answer.json"
            schema_file.write_text(json.dumps(schema))
            env = {**os.environ, "CODEX_HOME": str(self.home)}
            proc = subprocess.run(self._command(schema_file, out_file), input=prompt, capture_output=True,
                                  text=True, timeout=self.cfg["timeout_seconds"], env=env)
            usage, error = {}, None
            for line in proc.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                elif event.get("type") == "turn.failed":
                    error = event.get("error", {}).get("message")
            if error or not out_file.exists():
                raise RuntimeError(error or proc.stderr[-500:] or "codex returned no answer")
            tokens = {k: int(usage.get(k, 0)) for k in
                      ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")}
            return out_file.read_text(), tokens


class OpenAICompatibleLLM(LLM):
    name = "openai"

    def __init__(self, cfg: dict, cache_dir: Path):
        super().__init__(cfg, cache_dir)
        from openai import OpenAI

        self.client = OpenAI(base_url=cfg.get("base_url") or None, api_key=os.environ[cfg["api_key_env"]],
                             timeout=cfg["timeout_seconds"], max_retries=0)

    def _call(self, prompt: str, schema: dict) -> tuple[str, dict]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
        )
        usage = response.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        tokens = {"input_tokens": usage.prompt_tokens, "cached_input_tokens": cached,
                  "output_tokens": usage.completion_tokens, "reasoning_output_tokens": reasoning}
        return response.choices[0].message.content, tokens


def make_llm(settings: dict, cache_dir: Path) -> LLM | None:
    """The configured backend, or None (with a notice) when it cannot run here.

    SPENDCUBE_LLM=codex|openai|none and SPENDCUBE_LLM_MODEL override the settings file.
    """
    backend = os.environ.get("SPENDCUBE_LLM", settings["llm"]["backend"]).lower()
    if backend == "none":
        return None
    cfg = {**settings["llm"], **settings["llm"][backend]}
    cfg["model"] = os.environ.get("SPENDCUBE_LLM_MODEL", cfg["model"])
    if not os.environ.get(cfg["api_key_env"]):
        print(f"[llm] {cfg['api_key_env']} is not set: classifying without an LLM")
        return None
    if backend == "codex":
        if shutil.which("codex") is None:
            print("[llm] codex CLI not found on PATH: classifying without an LLM")
            return None
        return CodexLLM(cfg, cache_dir)
    return OpenAICompatibleLLM(cfg, cache_dir)
