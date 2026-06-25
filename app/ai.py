"""AI reasoning layer (Groq — free tier friendly).

Same three responsibilities and the same matching logic as before:
1. parse_items_batch        — structured brand / product / size-form / pack qty /
   variant / MPN + search query from raw Schein descriptions.
2. extract_and_validate_batch — one call per item that BOTH extracts pricing and
   applies the four trust criteria (brand, product name, size/form, pack qty)
   against real scraped competitor data (Issue 4: no defaulted "Exact"/95/75 —
   confidence must vary with evidence).
3. evaluate_equivalency     — exact_equivalent / close_equivalent /
   possible_alternative scoring (Issue 6).

Groq specifics handled here:
- OpenAI-compatible chat completions via the official `groq` SDK.
- JSON mode (`response_format={"type": "json_object"}`) requires a top-level
  OBJECT, so batched prompts return {"results": [...]}.
- Free-tier rate limits (~30 req/min on llama-3.3-70b-versatile): a request
  pacer + exponential backoff on 429/5xx is built in.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import List, Optional

# ── LLM provider selection ──────────────────────────────────────────────────
# Default 'groq' preserves the existing free-tier path EXACTLY. LLM_PROVIDER=gemini
# routes every call through Google Gemini's OpenAI-compatible endpoint instead
# (free-tier token budget ~1M TPM vs Groq's ~6K — the wall that makes runs take
# hours disappears; the binding limit becomes requests/day). 'openai'/'openrouter'
# presets are included for convenience. ONLY the SDK class, base URL, API key, and
# model list change here — all rotation / pacing / extraction logic is
# provider-agnostic because Groq's and OpenAI's SDKs share one interface.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
_PROVIDER_PRESETS = {
    "groq":       {"base_url": None,
                   "key_env": "GROQ_API_KEY",
                   "models": "llama-3.1-8b-instant,openai/gpt-oss-20b,llama-3.3-70b-versatile"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                   "key_env": "GEMINI_API_KEY",
                   "models": "gemini-2.0-flash,gemini-2.0-flash-lite"},
    "openai":     {"base_url": None,
                   "key_env": "OPENAI_API_KEY",
                   "models": "gpt-4o-mini"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY",
                   "models": "meta-llama/llama-3.3-70b-instruct"},
}
_preset = _PROVIDER_PRESETS.get(LLM_PROVIDER, _PROVIDER_PRESETS["groq"])

# Provider-specific kwargs merged into every chat-completion call. Gemini 2.5
# models THINK by default, and those reasoning tokens are billed against the
# completion budget (max_tokens) — so the extraction JSON gets truncated mid-object
# ("Expecting ',' delimiter") and every extract fails to the regex fallback.
# reasoning_effort="none" disables thinking → clean JSON, and faster/cheaper calls.
# Override per provider via LLM_REASONING_EFFORT.
_PROVIDER_EXTRA: dict = {}
if LLM_PROVIDER == "gemini":
    _re = os.environ.get("LLM_REASONING_EFFORT", "none").strip()
    if _re and _re.lower() != "default":
        _PROVIDER_EXTRA["reasoning_effort"] = _re
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or _preset["base_url"]
LLM_KEY_ENV = _preset["key_env"]

# The SDK class + exception types resolve to the active provider. Groq and OpenAI
# SDKs use the same method names AND the same exception names, so binding them
# here lets every `except RateLimitError / APIStatusError / …` block work for both.
if LLM_PROVIDER == "groq":
    from groq import Groq as _LLMClientCls, RateLimitError, APIStatusError
    try:                   # connection/timeout error types (present in modern SDKs)
        from groq import APITimeoutError, APIConnectionError
    except ImportError:    # very old SDK fallback — treat as generic, retryable
        class APITimeoutError(Exception):
            pass
        class APIConnectionError(Exception):
            pass
else:
    try:
        from openai import OpenAI as _LLMClientCls, RateLimitError, APIStatusError
        from openai import APITimeoutError, APIConnectionError
    except ImportError as _e:   # openai SDK not installed
        raise ImportError(
            f"LLM_PROVIDER={LLM_PROVIDER!r} needs the 'openai' package — run "
            f"`pip install openai` (or `pip install -r requirements.txt`).") from _e

from .models import OrderLineItem, PriceCandidate
from . import db
from .jobs import emit

log = logging.getLogger(__name__)

# Model list follows the active provider's preset, overridable by LLM_MODELS
# (universal). The legacy GROQ_MODELS/GROQ_MODEL vars apply ONLY when the provider
# is groq — otherwise a leftover GROQ_MODELS in .env would feed Llama names to a
# Gemini/OpenAI endpoint and every call would 404. Rotation cycles whatever models
# are listed here across their separate per-model rate buckets.
_DEFAULT_MODELS = _preset["models"]
_legacy_models = (os.environ.get("GROQ_MODELS") or os.environ.get("GROQ_MODEL")) \
    if LLM_PROVIDER == "groq" else None
MODELS = [m.strip() for m in
          (os.environ.get("LLM_MODELS") or _legacy_models or _DEFAULT_MODELS).split(",")
          if m.strip()]
MODEL = MODELS[0]
_model_idx = {"i": 0}
# Free-tier strategy: llama-3.3-70b-versatile has only ~100K tokens/day on the
# free plan and is exhausted within a handful of items, after which extraction
# silently drops to the weak models. So 8b-instant (~500K TPD) is now PRIMARY for
# the bulk of pages, and 70b is reserved as an ESCALATION model spent only on the
# few candidates the cheap model was unsure about (low confidence / conflicting
# criteria / unreliable price) — see _escalate() in extract_and_validate_batch.
ESCALATE_MODEL = os.environ.get("GROQ_ESCALATE_MODEL", "llama-3.3-70b-versatile")
ESCALATE_ENABLED = os.environ.get("GROQ_ESCALATE", "1") not in ("0", "false", "False")
ESCALATE_CONF = int(os.environ.get("GROQ_ESCALATE_CONF", "70"))   # below this confidence → re-check on 70b
ESCALATE_MAX = int(os.environ.get("GROQ_ESCALATE_MAX", "3"))      # cap escalated pages per item (protect 70b budget)
# Extraction-result cache: scrapes are already cached (0 Firecrawl credits on a
# re-run); this caches the Groq EXTRACTION too, keyed by (url + ordered variant +
# pack), so a debug re-run of the same order skips Groq entirely for unchanged
# pages — the biggest free-tier token saver for an iterative workflow.
EXTRACT_CACHE_ENABLED = os.environ.get("EXTRACT_CACHE", "1") not in ("0", "false", "False")
EXTRACT_CACHE_HOURS = int(os.environ.get("EXTRACT_CACHE_HOURS", "24"))
# Bump this whenever extraction logic (slicing / prompt) changes, so stale cached
# verdicts are invalidated on the next run instead of masking the new code.
# v2: variant-aware windowing for multi-product pages (safco A3 $140.49→$364.99).
# v3: only window when the variant row is DEEP — early-variant pages were getting
#     their top-of-page price stripped, nulling the candidate (dropped XLight, A3.5).
# v4: deterministic-price-first — single-offer JSON-LD prices are set in search.py
#     (price_structured) and such pages now get a small VALIDATE-only slice, so old
#     full-slice verdicts must be invalidated.
# v5: pack-aware masking — on multi-pack pages the wrong pack's prices are blanked
#     before extraction (Clinpro 50-pk $137 vs 100-pk $272.99), so old verdicts that
#     may have read the wrong pack must be re-extracted.
# v6: MPN-aware matching — MPN added to the validation prompt; old verdicts re-run.
_EXTRACT_LOGIC_VERSION = os.environ.get("EXTRACT_LOGIC_VERSION", "6")
# Token trims (free-tier saver):
#  - Locked pages (net32 / supplyclinic) already have an authoritative price from
#    the seller-table parser; the LLM only needs the page top (title/variant) to
#    judge criteria, so we send a small slice instead of the full page.
#  - Price-window: for a NORMAL page whose price sits PAST the head window, we
#    stitch the page top (title/variant) + the block around the price instead of
#    sending the whole page. When the price is already inside the head we send the
#    head unchanged (no behaviour change / no truncation-bug regression).
EXTRACT_LOCKED_PAGE_CHARS = int(os.environ.get("EXTRACT_LOCKED_PAGE_CHARS", "1500"))
EXTRACT_PRICE_WINDOW = os.environ.get("EXTRACT_PRICE_WINDOW", "1") not in ("0", "false", "False")
EXTRACT_WIN_BEFORE = int(os.environ.get("EXTRACT_WIN_BEFORE", "1500"))
EXTRACT_WIN_AFTER = int(os.environ.get("EXTRACT_WIN_AFTER", "2200"))
# Variant-aware windowing: on a MULTI-PRODUCT page (one URL listing an applier +
# assortments + every shade, e.g. safco's "Fuji II LC capsules"), the row the
# buyer actually ordered (A3 / 48-box = $364.99) sits far below a cheaper,
# unrelated first row (the $140.49 applier). The model otherwise grabs that first
# price and labels it with the page title, so the title-based applier guard never
# fires. When ON, the model's view is re-centred on the ordered variant's row and
# the price-bearing head is dropped, so the applier price is never in front of it.
EXTRACT_VARIANT_ANCHOR = os.environ.get("EXTRACT_VARIANT_ANCHOR", "1") not in ("0", "false", "False")
EXTRACT_HEAD_CHARS = int(os.environ.get("EXTRACT_HEAD_CHARS", "600"))   # price-free title/brand head
EXTRACT_VAR_BEFORE = int(os.environ.get("EXTRACT_VAR_BEFORE", "200"))   # context kept before the variant row
EXTRACT_VAR_PRICE_GAP = int(os.environ.get("EXTRACT_VAR_PRICE_GAP", "300"))  # max chars variant→its price
# Run-level circuit breaker: when the LAST model (no failover left) gets hard
# rate-limited, every subsequent call would grind through MAX_RETRIES×60s of
# backoff — and with split-retry that multiplies into 10-15min freezes on a
# single item. Once tripped, calls fail fast for a short cooldown so the run
# keeps moving (candidates stay unverified rather than the pipeline hanging).
_groq_cb = {"tripped_until": 0.0}
GROQ_CB_COOLDOWN = int(os.environ.get("GROQ_COOLDOWN", "90"))  # seconds to fail-fast after last-model wall
GROQ_MAX_CYCLES = int(os.environ.get("GROQ_MAX_CYCLES", "3"))  # full passes through the model chain before giving up
MIN_INTERVAL = float(os.environ.get("GROQ_MIN_INTERVAL", "2.6"))  # ~23 req/min, gentler on free-tier RPM
MAX_RETRIES = 5
MAX_BACKOFF = int(os.environ.get("GROQ_MAX_BACKOFF", "60"))  # hard cap on any single 429 wait (seconds)
# Per-request HTTP timeout for the Groq SDK. The SDK's default is short enough
# that a large extract+validate batch (several full product pages of markdown)
# can be cut off mid-flight — the request aborts, the whole chunk's extracted
# prices/criteria are lost, and the candidates fall back to raw search-snippet
# prices (which is exactly how a wrong price like $9.95 survives into the
# report). A generous timeout lets slow-but-valid responses finish instead of
# being dropped. Env-tunable; raise further on a slow host.
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "120"))
# HARD wall-clock ceiling for a SINGLE Groq completion. The SDK's own `timeout`
# (GROQ_TIMEOUT above) can fail to fire on a wedged / slowly-trickling TLS read —
# observed in production as one call hanging ~28 minutes and freezing the entire
# run (both worker threads stuck in recv, the per-item cap unable to help). We
# therefore run every completion in a DAEMON helper thread and enforce an
# absolute deadline here, mirroring the Firecrawl scrape guard. On breach the
# stuck thread is abandoned (daemon → it can never block process exit) and
# _GroqHardTimeout is raised, which the retry/failover path treats like a
# transient timeout (fail over to the next model, or fall back to the regex
# extractor) instead of hanging. Set comfortably ABOVE GROQ_TIMEOUT so a
# legitimately-slow-but-valid response still finishes normally via the SDK.
GROQ_HARD_DEADLINE = float(os.environ.get("GROQ_HARD_DEADLINE_SEC", "150"))

_client = None
_lock = threading.Lock()
_last_call = 0.0


class _GroqHardTimeout(Exception):
    """Raised when an LLM completion blows the hard wall-clock ceiling."""


def client():
    global _client
    if _client is None:
        # max_retries=0 — disable the SDK's OWN retry layer so our paced
        # backoff (_pace + _ask_json) is the single source of retry truth.
        # Stacking both layers double-hammers the endpoint and wastes RPM.
        # timeout=GROQ_TIMEOUT — give big batches room to complete so a slow
        # response is never truncated/aborted (which loses the whole chunk's
        # data and lets stale snippet prices through).
        kw = {"max_retries": 0, "timeout": GROQ_TIMEOUT}
        # Groq reads GROQ_API_KEY from env with the SDK default base URL (path
        # unchanged). Other providers take an explicit key + OpenAI-compatible
        # base URL.
        if LLM_PROVIDER != "groq":
            kw["api_key"] = os.environ.get(LLM_KEY_ENV)
            if LLM_BASE_URL:
                kw["base_url"] = LLM_BASE_URL
        _client = _LLMClientCls(**kw)
    return _client


def _create_with_deadline(**kwargs):
    """Run client().chat.completions.create(**kwargs) under GROQ_HARD_DEADLINE.

    The blocking call runs in a daemon thread so a wedged TLS read can never hang
    the worker past the ceiling (the SDK's own timeout has been observed not to
    fire). The SDK's own exceptions (RateLimitError, APIStatusError, …) are
    re-raised UNCHANGED on the caller's thread so all existing retry/failover
    handling is preserved; only a genuine ceiling breach raises _GroqHardTimeout."""
    box: dict = {}
    # explicit kwargs win; provider extras (e.g. Gemini reasoning_effort) fill in
    if _PROVIDER_EXTRA:
        kwargs = {**_PROVIDER_EXTRA, **kwargs}

    def _run():
        try:
            box["resp"] = client().chat.completions.create(**kwargs)
        except Exception as e:        # capture; re-raised on the caller thread
            box["err"] = e

    t = threading.Thread(target=_run, name="groq-call", daemon=True)
    t.start()
    t.join(GROQ_HARD_DEADLINE)
    if t.is_alive():                  # ceiling breached — abandon the daemon thread
        raise _GroqHardTimeout(
            f"Groq call exceeded {GROQ_HARD_DEADLINE:.0f}s hard ceiling — abandoned "
            f"(stuck thread leaked as daemon; run continues via failover/fallback)")
    if "err" in box:
        raise box["err"]
    return box["resp"]


def _pace():
    """Serialize + space requests so parallel item workers respect free-tier RPM."""
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


# --- Optional tokens-per-minute (TPM) pacer ---------------------------------
# The binding free-tier wall is TOKENS per minute (e.g. 12000 TPM on
# llama-3.3-70b), not requests — the run-killing 429s are "rate_limit_exceeded …
# tokens per minute (TPM): Limit 12000", which the RPM-only _pace() above cannot
# prevent. When GROQ_TPM_PACER=1 this token bucket throttles by ESTIMATED
# prompt+completion tokens against a rolling 60s window, holding the run just
# under the ceiling instead of blowing it and cascading to 8b/gpt-oss. OFF by
# default (zero behaviour change); turn on for the free tier, leave off on a
# paid tier. The estimate is reserved before the call and corrected to the real
# usage afterwards, so a few big batches can't silently overshoot.
import itertools as _it
TPM_PACER = os.environ.get("GROQ_TPM_PACER", "0") not in ("0", "false", "False", "")
# Multi-model rotation: each Groq model has its OWN free-tier TPM bucket, so when
# the active model's bucket is drained, route the call to another model whose
# bucket still has room instead of sleeping on the drained one. Round-robining
# all N models multiplies effective free-tier throughput ~Nx. OFF by default
# (the pacer just waits on the active model, unchanged); turn on for the free
# tier. No effect on a paid key (leave the pacer off there entirely).
TPM_ROTATE = os.environ.get("GROQ_TPM_ROTATE", "0") not in ("0", "false", "False", "")
# Per-model TPM ceilings (free tier), with headroom under each published wall.
# The pacer budget now FOLLOWS whichever model is actually active: after a
# failover from 70b (12K TPM wall) down to 8b (6K) or gpt-oss (8K), it stops
# reserving against a 70b-sized budget and overshooting the smaller model — the
# cause of the 429 storm once 70b's daily token quota runs out mid-run.
_DEFAULT_MODEL_TPM = {
    "llama-3.3-70b-versatile": 11000,   # wall 12000
    "llama-3.1-8b-instant":     5500,   # wall  6000
    "openai/gpt-oss-20b":       7500,   # wall  8000
    "openai/gpt-oss-120b":      7500,   # wall  8000
    # Gemini free tier is ~1M TPM — effectively no token wall; the binding limit
    # is requests/min (handled by _pace) and requests/day. Set high so the TPM
    # pacer is a no-op and rotation is driven only by RPM / 429 cooldowns.
    "gemini-2.0-flash":      1000000,
    "gemini-2.0-flash-lite": 1000000,
    "gemini-2.5-flash":       250000,
    "gemini-1.5-flash":      1000000,
}
# An explicit GROQ_TPM_BUDGET (if set) becomes a GLOBAL CEILING applied on top of
# the per-model value: a manual clamp still works, but a stale value can never
# push a small model ABOVE its wall again.
_tpm_budget_env = os.environ.get("GROQ_TPM_BUDGET")
TPM_BUDGET_CEILING = int(_tpm_budget_env) if _tpm_budget_env else None
TPM_BUDGET = TPM_BUDGET_CEILING or 11000   # back-compat for any other reference/log
TPM_CHARS_PER_TOKEN = float(os.environ.get("GROQ_TPM_CHARS_PER_TOKEN", "4.0"))


def _model_tpm(model: str) -> int:
    """Pacer budget for a specific model: per-model env override (e.g.
    GROQ_TPM_LLAMA_3_1_8B_INSTANT) -> built-in table -> safe floor, then clamped
    by any explicit global GROQ_TPM_BUDGET ceiling."""
    envk = "GROQ_TPM_" + re.sub(r"[^0-9A-Z]+", "_", (model or "").upper()).strip("_")
    v = os.environ.get(envk)
    # Unknown-model floor: 5500 is Groq's smallest wall, but on a high-budget
    # provider (Gemini ~1M) an unlisted model must NOT be throttled to 5500.
    _floor = 5500 if LLM_PROVIDER == "groq" else 1000000
    base = int(v) if v else _DEFAULT_MODEL_TPM.get(model, _floor)
    if TPM_BUDGET_CEILING is not None:
        base = min(base, TPM_BUDGET_CEILING)
    return base
_tpm_lock = threading.Lock()
_tpm_events: dict[int, tuple[float, int, str]] = {}   # id -> (ts, tokens, model); per-model 60s window
_tpm_ids = _it.count()


def _est_tokens(prompt: str, max_tokens: int) -> int:
    """Rough token estimate: ~4 chars/token for the prompt + the reserved
    completion cap. Deliberately conservative (overestimating only makes the
    pacer stay further under the wall)."""
    return int(len(prompt) / TPM_CHARS_PER_TOKEN) + int(max_tokens)


# Output-size budget for the extract+validate call. The JSON it returns is only
# ~120-160 tokens PER PAGE, so the old flat max_tokens=3000 bloated every pacer
# estimate (and, on the small free-tier models, pushed even a 2-page call over
# their 6K/8K TPM wall). Scale to the chunk with ~2x headroom + a floor;
# truncation would only drop a chunk to the regex fallback, never a wrong price.
EXTRACT_OUT_BASE     = int(os.environ.get("EXTRACT_OUT_BASE", "500"))
EXTRACT_OUT_PER_PAGE = int(os.environ.get("EXTRACT_OUT_PER_PAGE", "250"))
EXTRACT_OUT_MIN      = int(os.environ.get("EXTRACT_OUT_MIN", "700"))
EXTRACT_OUT_MAX      = int(os.environ.get("EXTRACT_OUT_MAX", "3000"))


def _extract_out_tokens(n_pages: int) -> int:
    """Completion-token cap sized to the number of pages in the chunk."""
    return max(EXTRACT_OUT_MIN,
               min(EXTRACT_OUT_MAX,
                   EXTRACT_OUT_BASE + EXTRACT_OUT_PER_PAGE * max(1, n_pages)))


def _tpm_prune(now: float) -> None:
    cutoff = now - 60.0
    for k in [k for k, (ts, _t, _m) in _tpm_events.items() if ts < cutoff]:
        _tpm_events.pop(k, None)


def _tpm_used(model: str) -> int:
    """Tokens reserved for `model` in the current 60s window (per-model bucket)."""
    return sum(t for _ts, t, m in _tpm_events.values() if m == model)


def _tpm_reserve(est: int, model: str) -> Optional[int]:
    """Block until adding `est` tokens keeps THIS model's rolling 60s usage under
    its own TPM ceiling, then reserve a slot and return its id (None when pacer
    off). Usage is tracked per-model: each Groq model has a separate free-tier
    bucket, so a drained 8b window no longer throttles a call routed to 70b."""
    if not TPM_PACER or est <= 0:
        return None
    budget = _model_tpm(model)
    while True:
        with _tpm_lock:
            now = time.monotonic()
            _tpm_prune(now)
            used = _tpm_used(model)
            mine = [ts for ts, _t, m in _tpm_events.values() if m == model]
            # always admit when this model's window is empty (a single batch may
            # itself exceed the budget — we can't split it, so let it through).
            if not mine or used + est <= budget:
                tok_id = next(_tpm_ids)
                _tpm_events[tok_id] = (now, est, model)
                return tok_id
            wait = max(0.05, (min(mine) + 60.0) - now)
        log.info("TPM pacer: %s %d tok in window + %d est > %d budget — waiting %.1fs",
                 model, used, est, budget, min(wait, 5.0))
        time.sleep(min(wait, 5.0))


def _tpm_reserve_rotate(est: int, models: list) -> tuple:
    """Pick the FIRST model in `models` whose own 60s bucket has room for `est`
    tokens, reserve under it, and return (model, tok_id). Only sleep when EVERY
    model's bucket is full — then wake as soon as the soonest one frees a slot.
    This is what turns 'wait 5s on 8b' into 'use 20b/70b right now', multiplying
    free-tier throughput by the number of models. (models[0], None) when off."""
    if not TPM_PACER or est <= 0:
        return (models[0], None)
    while True:
        with _tpm_lock:
            now = time.monotonic()
            _tpm_prune(now)
            soonest = None
            for m in models:
                mine = [ts for ts, _t, mm in _tpm_events.values() if mm == m]
                if not mine or _tpm_used(m) + est <= _model_tpm(m):
                    tok_id = next(_tpm_ids)
                    _tpm_events[tok_id] = (now, est, m)
                    return (m, tok_id)
                w = (min(mine) + 60.0) - now
                soonest = w if soonest is None else min(soonest, w)
        log.info("TPM pacer: all %d model bucket(s) full — waiting %.1fs",
                 len(models), min(soonest or 5.0, 5.0))
        time.sleep(min(max(soonest or 5.0, 0.05), 5.0))


def _tpm_correct(tok_id: Optional[int], actual: int) -> None:
    """Replace a reservation's estimate with the real total_tokens once known."""
    if tok_id is None or not actual or actual <= 0:
        return
    with _tpm_lock:
        if tok_id in _tpm_events:
            ts, _t, m = _tpm_events[tok_id]
            _tpm_events[tok_id] = (ts, int(actual), m)


def _parse_json_text(text: str) -> dict:
    """Strip code fences and parse the model's JSON reply (with a brace-scan
    fallback). Shared by the standard and rotation paths."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _ask_json_rotate(messages: list, est: int, max_tokens: int) -> dict:
    """Free-tier throughput path (GROQ_TPM_ROTATE=1): round-robin across all
    models by available TPM bucket instead of waiting on one drained model. A
    model that 429s is put on a short cooldown so rotation skips it; the circuit
    breaker trips only when every model is unusable for longer than the backoff
    cap. Result is identical JSON to the standard path — only which model ran it
    differs, and the deterministic guards downstream are model-independent."""
    last_err = None
    cooldown: dict[str, float] = {}   # model -> monotonic deadline to skip until
    spins = max(GROQ_MAX_CYCLES * len(MODELS) * MAX_RETRIES, 6)
    for _ in range(spins):
        if time.monotonic() < _groq_cb["tripped_until"]:
            raise RuntimeError("Groq all-models rate-limited; failing fast to keep run moving")
        now = time.monotonic()
        avail = [m for m in MODELS if cooldown.get(m, 0.0) <= now] or list(MODELS)
        model, _tok = _tpm_reserve_rotate(est, avail)
        _pace()
        try:
            resp = _create_with_deadline(
                model=model, max_tokens=max_tokens, temperature=0,
                response_format={"type": "json_object"}, messages=messages,
            )
            _tpm_correct(_tok, getattr(getattr(resp, "usage", None), "total_tokens", 0))
            return _parse_json_text(resp.choices[0].message.content or "")
        except RateLimitError as e:
            last_err = e
            ra = None
            try:
                ra = float(getattr(e, "response", None).headers.get("retry-after"))
            except Exception:
                ra = None
            # skip this model's bucket for the Retry-After (capped); rotation uses
            # the other models meanwhile. A long RA = daily-quota exhaustion.
            skip = min(ra or 30.0, 300.0)
            cooldown[model] = time.monotonic() + skip
            log.warning("Groq %s rate-limited — rotating to another model "
                        "(skipping it for %ds)", model, round(skip))
        except APIStatusError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 413:
                raise RuntimeError("Groq 413 payload too large") from e
            last_err = e
            cooldown[model] = time.monotonic() + 20.0
        except json.JSONDecodeError as e:
            last_err = e   # transient; try again (possibly another model)
        except (APITimeoutError, APIConnectionError, _GroqHardTimeout) as e:
            last_err = e
            cooldown[model] = time.monotonic() + 8.0
        # every model cooling down for longer than the backoff cap → give up fast
        if MODELS and all(cooldown.get(m, 0.0) > time.monotonic() for m in MODELS):
            if max(cooldown.values()) - time.monotonic() > MAX_BACKOFF:
                _groq_cb["tripped_until"] = time.monotonic() + GROQ_CB_COOLDOWN
                log.error("Groq all models rate-limited (rotate) — tripping circuit "
                          "breaker for %ss", GROQ_CB_COOLDOWN)
                raise RuntimeError("Groq all-models rate-limited; circuit breaker tripped")
    raise RuntimeError(f"All Groq models exhausted (rotate): {last_err}")


def _ask_json(prompt: str, max_tokens: int = 4000, force_model: str | None = None) -> dict:
    """Ask Groq for JSON with retry + automatic model failover. Each model in
    MODELS has its OWN free-tier rate limit; if the current model stays
    rate-limited past a couple of attempts and another model remains, we fail
    over to it (and remember that for the rest of the run). Only when every
    model is exhausted does the call raise.

    force_model: when set, use ONLY that model (no failover chain) — used by the
    escalation pass to spend the scarce 70b budget on a specific hard page."""
    last_err = None
    # circuit breaker: if the last model was just hard-rate-limited, fail fast
    # for a cooldown window instead of grinding 5×60s backoffs on every call
    # (which, with split-retry, multiplied into 10-15min freezes per item).
    if time.monotonic() < _groq_cb["tripped_until"]:
        raise RuntimeError("Groq all-models rate-limited; failing fast to keep run moving")
    messages = [
        {"role": "system",
         "content": "You are a precise dental supply data engine. "
                    "Respond ONLY with valid JSON. No prose, no markdown."},
        {"role": "user", "content": prompt},
    ]
    est = _est_tokens(prompt, max_tokens)   # TPM pacer reservation size (no-op unless enabled)
    # Free-tier throughput path: rotate across models' separate TPM buckets
    # instead of waiting on one. Escalation (force_model) keeps the single-model
    # path so it still spends the scarce 70b budget on exactly the page it chose.
    if TPM_ROTATE and not force_model:
        return _ask_json_rotate(messages, est, max_tokens)
    # Escalation path: single forced model, short retry, NO failover/cycling.
    if force_model:
        for attempt in range(2):
            _pace()
            _tok = _tpm_reserve(est, force_model)
            try:
                resp = _create_with_deadline(
                    model=force_model, max_tokens=max_tokens, temperature=0,
                    response_format={"type": "json_object"}, messages=messages,
                )
                _tpm_correct(_tok, getattr(getattr(resp, "usage", None), "total_tokens", 0))
                text = resp.choices[0].message.content or ""
                text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    m = re.search(r"\{.*\}", text, re.S)
                    if m:
                        return json.loads(m.group(0))
                    raise
            except RateLimitError as e:
                last_err = e
                ra = None
                try:
                    ra = float(getattr(e, "response", None).headers.get("retry-after"))
                except Exception:
                    ra = None
                # escalation model out of quota → don't wait; keep the cheap result
                if ra and ra > MAX_BACKOFF:
                    raise RuntimeError(f"escalation model {force_model} quota exhausted") from e
                time.sleep(min(ra or 6, MAX_BACKOFF))
            except APIStatusError as e:
                if getattr(getattr(e, "response", None), "status_code", None) == 413:
                    raise RuntimeError("Groq 413 payload too large") from e
                last_err = e
                time.sleep(4)
            except (APITimeoutError, APIConnectionError, _GroqHardTimeout, json.JSONDecodeError) as e:
                last_err = e
                time.sleep(3)
        raise RuntimeError(f"escalation model {force_model} failed: {last_err}")
    # Cycle through the whole model chain up to GROQ_MAX_CYCLES times before
    # giving up: after the last model rate-limits, loop BACK to the first (its
    # per-minute window may have reset). Bounded so it can't spin forever.
    for cycle in range(GROQ_MAX_CYCLES):
        start_idx = _model_idx["i"] if cycle == 0 else 0
        for mi in range(start_idx, len(MODELS)):
            model = MODELS[mi]
            rate_limited = False
            for attempt in range(MAX_RETRIES):
                _pace()
                _tok = _tpm_reserve(est, model)
                try:
                    resp = _create_with_deadline(
                        model=model, max_tokens=max_tokens, temperature=0,
                        response_format={"type": "json_object"}, messages=messages,
                    )
                    _tpm_correct(_tok, getattr(getattr(resp, "usage", None), "total_tokens", 0))
                    text = resp.choices[0].message.content or ""
                    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        m = re.search(r"\{.*\}", text, re.S)
                        if m:
                            return json.loads(m.group(0))
                        raise
                except RateLimitError as e:
                    last_err = e
                    rate_limited = True
                    more_models = mi < len(MODELS) - 1
                    more_cycles = cycle < GROQ_MAX_CYCLES - 1
                    # read Retry-After if present (Groq sends a LONG one — 40+ min —
                    # when the DAILY token quota is exhausted, not a brief throttle)
                    ra = None
                    try:
                        ra = float(getattr(e, "response", None).headers.get("retry-after"))
                    except Exception:
                        ra = None
                    # long wait = daily-quota exhaustion: fail over to the next model
                    # (separate quota) immediately rather than sleeping it off
                    if ra and ra > MAX_BACKOFF and more_models:
                        log.warning("Groq %s daily-quota exhausted (Retry-After %ss) — "
                                    "failing over to %s instead of waiting",
                                    model, round(ra), MODELS[mi + 1])
                        break
                    if more_models and attempt >= 1:
                        log.warning("Groq %s rate-limited — failing over to %s",
                                    model, MODELS[mi + 1])
                        break
                    # hard-cap any wait so a huge Retry-After can never block the run
                    backoff = min(ra, MAX_BACKOFF) if ra else min(8 * (attempt + 1), MAX_BACKOFF)
                    if ra and ra > MAX_BACKOFF:
                        log.warning("Groq %s Retry-After %ss exceeds %ss cap — waiting cap "
                                    "then giving up this model", model, round(ra), MAX_BACKOFF)
                    # last model in this cycle, still rate-limited: if more cycles
                    # remain, break out to loop back to model 0; otherwise trip the
                    # circuit breaker so subsequent calls fail fast.
                    if not more_models and attempt >= 1:
                        if more_cycles:
                            log.warning("Groq %s (last model, cycle %d/%d) rate-limited — "
                                        "looping back to %s", model, cycle + 1,
                                        GROQ_MAX_CYCLES, MODELS[0])
                            break
                        _groq_cb["tripped_until"] = time.monotonic() + GROQ_CB_COOLDOWN
                        log.error("Groq exhausted after %d cycles — tripping circuit breaker "
                                  "for %ss; calls fail fast so the run finishes (candidates "
                                  "left for manual extraction this window)", GROQ_MAX_CYCLES,
                                  GROQ_CB_COOLDOWN)
                        raise RuntimeError("Groq all-models rate-limited; circuit breaker tripped")
                    log.warning("Groq %s rate-limited (429) — waiting %ss (attempt %d/%d)",
                                model, round(backoff), attempt + 1, MAX_RETRIES)
                    time.sleep(backoff)
                except APIStatusError as e:
                    last_err = e
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    msg = str(e).lower()
                    # 413 Payload Too Large will NEVER succeed on retry — too big.
                    if status == 413:
                        log.error("Groq %s 413 Payload Too Large — prompt too big to "
                                  "extract; skipping this batch (no retry)", model)
                        raise RuntimeError("Groq 413 payload too large") from e
                    # A decommissioned / invalid model is permanent — drop and fail over.
                    if status == 400 and ("decommission" in msg or "model_decommissioned" in msg
                                          or "no longer supported" in msg or "has been deprecated" in msg):
                        if mi < len(MODELS) - 1:
                            log.error("Groq %s is decommissioned/invalid — failing over "
                                      "to %s (remove it from GROQ_MODELS)", model, MODELS[mi + 1])
                            rate_limited = True
                            break
                        log.error("Groq %s is decommissioned/invalid and no models remain", model)
                        raise RuntimeError(f"Groq model {model} decommissioned") from e
                    backoff = min(2 ** attempt * 2, 30)
                    log.warning("Groq %s %s — retrying in %ss (attempt %d/%d)",
                                model, type(e).__name__, backoff, attempt + 1, MAX_RETRIES)
                    time.sleep(backoff)
                except json.JSONDecodeError as e:
                    last_err = e
                    log.warning("Groq %s unparseable JSON — retrying (attempt %d)",
                                model, attempt + 1)
                except (APITimeoutError, APIConnectionError, _GroqHardTimeout) as e:
                    # A timeout/connection drop (or a hard-deadline breach on a
                    # wedged socket) is transient: the request was cut off before a
                    # (valid) response arrived. Retry with a short backoff instead
                    # of letting the exception bubble up and lose the whole chunk's
                    # extracted data. Fail over to the next model only after a
                    # couple of failures on this one. _GroqHardTimeout is what
                    # converts the old 28-min freeze into a bounded failover.
                    last_err = e
                    if mi < len(MODELS) - 1 and attempt >= 1:
                        log.warning("Groq %s timed out/connection error — failing over to %s",
                                    model, MODELS[mi + 1])
                        rate_limited = True   # reuse failover path below
                        break
                    backoff = min(4 * (attempt + 1), 20)
                    log.warning("Groq %s timeout/connection error — retrying in %ss "
                                "(attempt %d/%d): %s", model, backoff, attempt + 1,
                                MAX_RETRIES, type(e).__name__)
                    time.sleep(backoff)
            if rate_limited and mi < len(MODELS) - 1:
                _model_idx["i"] = mi + 1
                log.info("Groq failover: primary model is now %s for the rest of this run",
                         MODELS[mi + 1])
                continue
        # finished a full cycle without returning; if cycles remain, reset to
        # model 0 and try the whole chain again (quotas may have recovered)
        if cycle < GROQ_MAX_CYCLES - 1:
            _model_idx["i"] = 0
    raise RuntimeError(f"All Groq models exhausted after {GROQ_MAX_CYCLES} cycles "
                       f"({', '.join(MODELS)}): {last_err}")


def _results(data) -> list:
    """Accept {"results": [...]} (JSON mode) or a bare array (defensive)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "items", "candidates", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        # single-object response for a single-element batch
        if data:
            return [data]
    return []


# ---------------------------------------------------------------- parsing ---

PARSE_PROMPT = """You are a dental supply product data expert. For each Henry
Schein order line below, extract structured fields. Use your knowledge of
dental products (Dentsply, Kerr, 3M, GC, Kulzer, DMG, Coltene, Premier, etc.)
to identify the true brand and, where confidently known, the manufacturer part
number (MPN). If the MPN is printed in the description (e.g. "9736H-150",
"PF171-5.0", "841G-012") use it. If you are not confident about the MPN,
return null — never guess.

Return ONLY a JSON object: {{"results": [...]}} with one object per input
line, same order, each with keys:
- sku (string, echo back)
- brand (string|null)
- product_name (string)         # core product line, e.g. "TPH Spectra ST"
- size_form (string|null)       # e.g. "compules", "syringe", "59 ml bottle"
- pack_qty (int|null)           # units per pack, e.g. 20 for "20/Pk"
- variant (string|null)         # shade/color/size variant: "A2", "Green 6.5mm", "Medium"
- mpn (string|null)
- search_query (string)         # best web search query to find this exact product,
                                # include brand, product, variant and pack size
- generic_query (string)        # SECOND query with the brand REMOVED: product
                                # type + spec + variant + pack only, e.g.
                                # "high performance mixing tips 4.2mm yellow 48 pack".
                                # CRITICAL for Henry Schein house brands (Acclean,
                                # Maxima, Criterion, Benzo-Jel, Foamies, Neotray):
                                # no other supplier sells these under the Schein
                                # name, so the generic query finds the equivalents.
                                # The generic query MUST keep the dental/clinical
                                # product category so results stay on-domain (e.g.
                                # "dental impression mixing tips 4.2mm yellow 48",
                                # never just "mixing tips" which returns paint/craft
                                # supplies). Expand cryptic descriptions into the
                                # full product type using your domain knowledge
                                # (e.g. "SILHOUETTE MEDIUM, 12 PK" is a Porter
                                # Silhouette low-profile nasal mask).
- mpn_query (string|null)       # THIRD query using the manufacturer part number
                                # when known (e.g. "Clinpro 7210FF flavorless",
                                # "DMG 110351 mixing tips"). MPNs are the single
                                # most precise way to find the exact variant/pack
                                # across suppliers. Null if no reliable MPN.

Order lines:
{lines}
"""


def parse_items_batch(items: List[OrderLineItem], chunk_size: int = 12) -> List[OrderLineItem]:
    """Batched parsing. Chunked to stay inside free-tier output-token limits."""
    emit("groq", action="start", phase="parse_batch", provider=LLM_PROVIDER,
         items=len(items), message=f"{LLM_PROVIDER.upper()} parsing {len(items)} product descriptions")
    by_sku = {}
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        lines = "\n".join(
            f'{i.schein_sku} | "{i.description}" | UOM {i.uom}' for i in chunk
        )
        emit("groq", action="chunk", phase="parse_batch", provider=LLM_PROVIDER,
             chunk=start // chunk_size + 1, items=len(chunk))
        try:
            data = _ask_json(PARSE_PROMPT.format(lines=lines), max_tokens=4000)
        except Exception:
            log.exception("Groq parse chunk failed (items %d–%d) — falling back to raw descriptions",
                          start + 1, start + len(chunk))
            emit("groq", action="chunk_failed", phase="parse_batch", provider=LLM_PROVIDER)
            continue
        for d in _results(data):
            by_sku[str(d.get("sku"))] = d
    emit("groq", action="complete", phase="parse_batch", provider=LLM_PROVIDER,
         items=len(items), message=f"{LLM_PROVIDER.upper()} enrichment complete")
    for it in items:
        d = by_sku.get(it.schein_sku, {})
        it.brand = d.get("brand")
        it.product_name = d.get("product_name") or it.description
        it.size_form = d.get("size_form")
        it.variant = d.get("variant")
        it.mpn = d.get("mpn")
        if d.get("pack_qty"):
            try:
                it.pack_qty = int(d["pack_qty"])
            except (TypeError, ValueError):
                pass
        it.search_query = d.get("search_query") or it.description
        it.generic_query = d.get("generic_query") or None
        it.mpn_query = d.get("mpn_query") or None
    return items


# ------------------------------------------------------------ equivalency ---

EQUIV_PROMPT = """A dental practice maintains an equivalency table mapping
Henry Schein products to generic/equivalent alternatives. Evaluate this pair:

Original (Schein): "{description}" (brand {brand}, pack {pack_qty})
Claimed equivalent: "{equivalent}" — client note: "{note}"
Best market data found for the equivalent: {market}

Classify the equivalency confidence:
- "exact_equivalent": same product, different brand/channel; substitution straightforward
- "close_equivalent": same category, compatible specs, minor differences
- "possible_alternative": related product, requires client review

Return ONLY a JSON object:
{{"confidence_level": "...", "basis": "<one-sentence reasoning>"}}
"""


def evaluate_equivalency(item: OrderLineItem, equivalent_name: str,
                         note: str, market_summary: str) -> dict:
    try:
        return _ask_json(EQUIV_PROMPT.format(
            description=item.description, brand=item.brand, pack_qty=item.pack_qty,
            equivalent=equivalent_name, note=note or "", market=market_summary or "none",
        ), max_tokens=400)
    except Exception:
        log.exception("Groq equivalency failed for SKU %s", item.schein_sku)
        return {
            "confidence_level": "possible_alternative",
            "basis": "AI equivalency check unavailable — manual review required",
        }


# ----------------------------------------------- batched page extraction ---

EXTRACT_PROMPT = """You are extracting structured data from scraped dental-supply
product pages. The buyer ordered this reference product:

  Description: "{description}"
  Ordered variant (flavor/shade/size/sterility): {variant}
  Ordered pack quantity: {pack_qty}

Below are several scraped pages (markdown), each with an index. For EACH page,
extract the data for the SPECIFIC product/variant that page is about. Critical
rules for accuracy:
- price: the price the buyer would ACTUALLY PAY TODAY for ONE unit of the EXACT
  variant/pack the page is selling. Apply these rules in order:
  1. SALE/DISCOUNT: if the page shows both an original (higher) and a discounted
     (lower) price (e.g. a struck-through original next to a sale price, or "Was
     <high> Now <low>", or "Sale: <low>"), use the CURRENT/SALE/NOW (lower)
     price, never the original. In markdown a struck-through original may appear
     as a higher number next to the lower one — the lower current price is the
     answer. Read the ACTUAL numbers off THIS page; never invent one.
  2. EXACT OPTION: if the page lists multiple variants/pack sizes, use the price
     of the option matching the page's own title/URL — NEVER the default,
     lowest, or per-unit price of a DIFFERENT option.
  3. QUANTITY BREAKS: if the page shows tiered/bulk pricing, use the BASE
     single-pack price, not the bulk-discounted tier — unless the bulk price is
     the only one shown.
  4. TOTAL not per-unit: return the price for the whole pack as sold (the
     pack_quantity below), never the "$/each" unit price of a larger pack.
  null if no public price is shown.
- original_price: if the page shows a higher pre-discount/struck-through price
  alongside the sale price, return that original number here (so we can show
  "was $X, now $Y"). null if there is no separate original price.
- pack_quantity: units per pack/box as sold on this page (integer). null if unclear.
- variant: the shade/color/flavor/size this page is selling.
- product_name: the product title as shown on the page.
- requires_login_for_price: true if the price is hidden behind login/membership.
- in_stock: true/false if shown, else null.
- minimum_order_condition: any case-lot/minimum-qty/bulk-discount note attached
  to the price (e.g. "buy 4 get 1 free", "min order 2"), else null.

Return ONLY a JSON object {{"results": [...]}} with one object per page, each:
{{"idx": <int>, "product_name": <str|null>, "price": <number|null>,
  "original_price": <number|null>, "pack_quantity": <int|null>,
  "variant": <str|null>, "requires_login_for_price": <bool>,
  "in_stock": <bool|null>, "minimum_order_condition": <str|null>}}

Pages:
{pages}
"""


EXTRACT_VALIDATE_PROMPT = """You are analyzing scraped dental-supply product pages
to (1) extract pricing data and (2) judge how well each matches the buyer's order.

The buyer ordered (from Henry Schein):
  Brand: {brand}
  Product: {product_name}
  Size/form: {size_form}
  Ordered variant (flavor/shade/size/sterility): {variant}
  Ordered pack quantity: {pack_qty}
  Manufacturer part # (MPN): {mpn}
  Original description: "{description}"

Below are scraped pages (markdown), each with an index. For EACH page do BOTH:

STEP 1 — EXTRACT (read the page):
- price: the price the buyer would ACTUALLY PAY TODAY for ONE unit of the EXACT
  variant/pack the page sells. Rules in order:
  1. ONLY THIS PRODUCT: price the item named in the page title/buy box. NEVER take
     a price from a "Related Products", "Related Items", "Similar items", "Best
     sellers related", "Customers who bought", "You may also like", or
     "Recommended" section — those are DIFFERENT products. If the only prices you
     can see belong to such sections, return null.
  2. MULTI-SELLER / AGGREGATOR PAGES (e.g. Net32, SupplyClinic and similar that
     list several sellers for the SAME product under "Vendor options",
     "Price + Shipping" / "Total", "Other Sellers", or a "Lowest Price" badge):
     use the LOWEST in-stock seller's TOTAL price INCLUDING shipping — i.e. the
     value in the "Total" column / the row marked "Lowest Price". Do NOT use the
     big headline "$X/ea" at the top: that is just the default seller and often
     EXCLUDES shipping (e.g. headline "$26.93/ea + $9.95 shipping" is beaten by
     another seller's "$26.94 Total, free shipping" — the correct price is the
     $26.94 Total). When a per-each price has separate shipping, the comparable
     number is price + shipping.
  3. SALE/DISCOUNT: if both an original (higher) and a discounted (lower) price
     are shown for THIS product — struck-through original next to a sale price,
     or "Was <high> Now <low>" — use the CURRENT/SALE (lower) price. Only pair a
     "was"/"now" when BOTH numbers clearly belong to this same product; never
     build a fake discount from a related item's price.
  4. MATCH THE ORDERED OPTION: the buyer ordered pack={pack_qty}, variant="{variant}".
     If the page lists SEVERAL pack sizes (e.g. a "50-pack" group and a "100-pack"
     group) or several flavors/shades/sizes, return the price of the row matching
     BOTH the ordered pack size AND the ordered variant. NEVER return a smaller or
     cheaper pack's price, and never a different flavor/shade's price, just because
     it is lower. If the exact ordered pack+variant is NOT on the page, return the
     closest single option's price and set pack_quantity and variant to what you
     ACTUALLY priced (so the mismatch is visible) — do not pretend it matches.
  5. TOTAL not per-unit: the whole-pack price as sold, not a "$/each" of a larger
     pack (unless rule 2's seller Total already is the comparable).
  6. If you cannot find a clear price for THIS product, return null — do NOT guess
     and do NOT copy a number from another product or from these instructions.
  null if no public price is shown.
- original_price: the higher pre-discount price if one is actually shown for THIS
  product, else null.
- pack_quantity: units per pack/box as sold (integer), else null.
- variant: the shade/color/flavor/size this page sells.
- product_name: the product title on the page.
- requires_login_for_price: true if THIS product's price is behind login/membership
  (e.g. "Log in to see price", "Login to view pricing", "Members only"). Set this
  true and price=null even when unrelated prices (promos, related items) show a "$"
  elsewhere on the page.
- in_stock: true/false if shown, else null.
- minimum_order_condition: any case-lot/min-qty/bulk note, else null.

STEP 2 — VALIDATE (judge the match, using what you extracted in step 1):
Decide the four trust criteria strictly (true/false each):
  brand_match, name_match, size_form_match, pack_match
Rules:
- ALL FOUR true => match_type "exact". Any false => "approximate" if same product
  family, else "rejected".
- A different shade/variant (A2 vs A3, Green vs Yellow, S vs M) => name_match=false.
  The extracted page data is authoritative over the search-result title.
- A different FLAVOR, SHADE, SIZE, or COLOR than ordered => name_match=false, even
  when the brand and product line are identical. Examples that are MISMATCHES:
  ordered "No Flavor / Flavorless / Unflavored" but page is "Watermelon" or "Mint";
  ordered shade "A3.5" but page "C2" or "A2"; ordered size "#5" but page "#4".
  Set variant to the value the page actually sells so the mismatch is visible.
- A different pack quantity => pack_match=false even if cheaper.
- If the buyer's product has no identifiable brand (Brand null), set
  brand_match=true when the candidate is the same generic product type.
- Marketing-name variations of the SAME line are NOT a name mismatch
  ("Luxatemp Plus" vs "Luxatemp Automix Plus", "Glide" vs "Glide Pro-Health").
- MPN (manufacturer part number): if an MPN is given above and the page shows a
  CONFLICTING part number, it is a DIFFERENT product even if the name/color look
  identical → name_match=false (e.g. ordered "DCA06-170" but page is "DCA05-110";
  ordered "841G-012" but page "842-012"; ordered "9742M-040" but page "9743M-040").
  A MATCHING MPN strongly confirms the same product. Ignore reseller prefixes
  (a store SKU like "194-9749M-110" still carries the MPN "9749M-110").
- confidence: integer 0-100, vary with evidence quality; do not default.
- CONSISTENCY: match_type may be "exact" ONLY when all four booleans are true.

Return ONLY a JSON object {{"results": [...]}} with one object per page, each:
{{"idx": <int>, "product_name": <str|null>, "price": <number|null>,
  "original_price": <number|null>, "pack_quantity": <int|null>,
  "variant": <str|null>, "requires_login_for_price": <bool>,
  "in_stock": <bool|null>, "minimum_order_condition": <str|null>,
  "match_type": <"exact"|"approximate"|"rejected">, "confidence": <int>,
  "brand_match": <bool>, "name_match": <bool>, "size_form_match": <bool>,
  "pack_match": <bool>, "notes": <str>}}

Pages:
{pages}
"""

# Once 70b's daily quota is hit, stop probing it for the rest of the run so we
# don't waste a failed escalation call on every remaining item.
_esc_state = {"off": False}


def _extract_key(item, c) -> str:
    """Cache key for an extraction result: same (page, ordered variant, pack)
    yields the same key across runs, so a debug re-run reuses the cached verdict
    instead of paying Groq tokens again.

    The key is salted with EXTRACT_LOGIC_VERSION so that whenever the extraction
    logic itself changes (e.g. the variant-aware windowing that fixes safco's
    multi-product page), every cached verdict is invalidated automatically on the
    next run — otherwise a stale verdict for the same URL masks the new code for
    EXTRACT_CACHE_HOURS. Bump the version (or set EXTRACT_LOGIC_VERSION in env)
    whenever _slice / the extract prompt changes."""
    ov = (getattr(c, "_ordered_variant", "") or getattr(item, "variant", "") or "")
    raw = f"{_EXTRACT_LOGIC_VERSION}|{getattr(c, 'url', '')}|{ov.strip().lower()}|{getattr(item, 'pack_qty', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def extract_and_validate_batch(item, candidates: List[PriceCandidate]) -> None:
    """One Groq call per item that BOTH extracts pricing and judges match criteria.
    On a 413 (prompt too big for the current model — e.g. after failover to a
    smaller-context model), the chunk is split in half and retried down to single
    pages, so a size limit NEVER drops a whole item's data. Mutates candidates in
    place; deterministic checks still run AFTER this in matcher."""
    pending = [c for c in candidates
               if getattr(c, "scraped_markdown", None) and c.match_type != "rejected"]
    if not pending:
        return
    # CHUNK = max pages per Groq call; PER_PAGE = max chars per page in the prompt.
    # PER_PAGE was 1500, which on real product pages (Net32, dental-city, etc.)
    # frequently cut off BELOW the price/pack/variant block — Groq then returned
    # price=null and the stale search-snippet price survived into the report
    # (this is the root cause of wrong prices like the $9.95 glove row). 4000
    # chars comfortably covers the buy-box of a real product page while the
    # OVERSIZE reject in search.py still keeps category/listing pages out. The
    # 413 split-retry below is the safety net if a model's context is smaller.
    CHUNK = int(os.environ.get("EXTRACT_CHUNK", "2"))
    PER_PAGE = int(os.environ.get("EXTRACT_PER_PAGE_CHARS", "4000"))

    # ---- extraction cache: reuse prior verdicts, skip Groq for unchanged pages
    todo, hits = [], 0
    for c in pending:
        c._ev_key = _extract_key(item, c) if EXTRACT_CACHE_ENABLED else None
        cached = None
        if c._ev_key:
            try:
                cached = db.get_cached_extract(c._ev_key, EXTRACT_CACHE_HOURS)
            except Exception:
                cached = None
        if cached:
            _apply_ev_result(item, c, cached)
            hits += 1
        else:
            todo.append(c)
    log.info("SKU %s — %d scraped page(s): %d cache hit(s), %d to extract",
             item.schein_sku, len(pending), hits, len(todo))
    if not todo:
        emit("groq", action="cache_only", phase="extract_validate", provider=LLM_PROVIDER,
             sku=item.schein_sku, pages=len(pending))
        return

    emit("groq", action="start", phase="extract_validate", provider=LLM_PROVIDER,
         sku=item.schein_sku, pages=len(todo), cache_hits=hits,
         message=f"{LLM_PROVIDER.upper()} validating {len(todo)} page(s) for SKU {item.schein_sku}")

    # GROQ EXTRACT CAP (speed, accuracy-preserving): validating a page that ALREADY
    # has a deterministic price (aggregator lock or single-offer structured) is a
    # cheap small-slice call; EXTRACTING a price from an un-priced page is the
    # expensive call (full-page slice). The lowest valid price is always among the
    # CHEAPEST candidates, so we always validate every priced page but cap the
    # expensive extractions to the GROQ_EXTRACT_CAP cheapest un-priced pages. The
    # pricier deferred pages keep their snippet/structured price via the regex
    # fallback (flagged UNVERIFIED → they show as lower-priority options, never a
    # confirmed EXACT). Set GROQ_EXTRACT_CAP=0 to disable (validate everything).
    extract_cap = int(os.environ.get("GROQ_EXTRACT_CAP", "4"))
    if extract_cap > 0:
        def _has_det_price(c):
            return getattr(c, "price_locked", False) or getattr(c, "price_structured", False)
        det_priced = [c for c in todo if _has_det_price(c)]
        unpriced = [c for c in todo if not _has_det_price(c)]
        if len(unpriced) > extract_cap:
            unpriced.sort(key=lambda c: c.price if (c.price and c.price > 0) else 1e18)
            deferred = unpriced[extract_cap:]
            todo = det_priced + unpriced[:extract_cap]
            log.info("SKU %s — Groq extract cap: validating %d priced + %d cheapest "
                     "un-priced page(s); %d pricier page(s) deferred to regex fallback "
                     "(set GROQ_EXTRACT_CAP higher to validate more)",
                     item.schein_sku, len(det_priced), extract_cap, len(deferred))

    def _slice(c, per_page):
        """Page text sent to the model, trimmed to save free-tier tokens:
          - price already DETERMINED (price_locked = aggregator seller table, or
            price_structured = single-offer JSON-LD): small head slice. The price
            is authoritative; the LLM only confirms the variant/criteria, so it
            needs the title/variant block, not the whole page. This is the bulk of
            the token saving — every page with a trustworthy deterministic price
            costs a fraction of a full extraction.
          - normal pages: full head UNLESS the price sits past the head window, in
            which case stitch the top (title/variant) + the block around the price
            so a deep price is never truncated away."""
        # PACK-AWARE masking first: blank prices belonging to a DIFFERENT pack size
        # than ordered, so a multi-pack page (Clinpro 100-pk $272.99 vs 50-pk
        # $137.49) can't feed the model the wrong pack's price. No-op on single-pack
        # pages. Applied to every slice path (validation slices included).
        md = _mask_wrong_pack(c.scraped_markdown or "", item)
        if getattr(c, "price_locked", False) or getattr(c, "price_structured", False):
            return md[:EXTRACT_LOCKED_PAGE_CHARS]
        if len(md) <= per_page:
            return md
        if EXTRACT_PRICE_WINDOW:
            # (1) MULTI-PRODUCT page where the ORDERED variant's row sits DEEP (past
            # the head), below a cheaper unrelated row — e.g. safco's $364.99 A3 refill
            # below the $140.49 applier. Centre on that row with a price-FREE head so
            # the applier/"$99 free-shipping" prices are never shown to the model.
            # ONLY when the anchor is past the head: if the variant is near the top
            # (a normal single-product page — title shade + price beside it), the
            # price-free head would BLANK the real price and the window would start
            # AFTER it, leaving the model no price at all → null → the candidate (and
            # sometimes the whole item) drops out. So for an early variant, fall
            # through to the plain slice, which shows that top-of-page price intact.
            anchor = _variant_anchor(md, item)
            if anchor is not None and anchor >= EXTRACT_HEAD_CHARS:
                head = _PAGE_PRICE_RE.sub("$—", md[:EXTRACT_HEAD_CHARS])
                start = max(EXTRACT_HEAD_CHARS, anchor - EXTRACT_VAR_BEFORE)
                end = anchor + EXTRACT_WIN_AFTER
                return (head + "\n…\n" + md[start:end])[:per_page]
            # (2) otherwise: keep the head, but if the price sits past it, stitch in
            # the block around the first price so a deep price isn't truncated away.
            m = _PAGE_PRICE_RE.search(md)
            if m and m.start() > per_page - 400:
                head = md[: per_page // 3]
                start = max(len(head), m.start() - EXTRACT_WIN_BEFORE)
                end = m.start() + EXTRACT_WIN_AFTER
                return (head + "\n…\n" + md[start:end])[:per_page]
        return md[:per_page]

    def _save(c, d):
        if EXTRACT_CACHE_ENABLED and getattr(c, "_ev_key", None):
            try:
                db.save_extract(c._ev_key, d)
            except Exception:
                pass

    def _run(chunk, per_page):
        """Send one chunk; on 413 (too big for the current model) split in half
        and retry, down to single pages, so a size limit NEVER drops data."""
        if not chunk:
            return
        pages = "\n\n".join(
            f'--- PAGE idx={i} (url: {c.url}) ---\n{_slice(c, per_page)}'
            for i, c in enumerate(chunk))
        try:
            data = _ask_json(EXTRACT_VALIDATE_PROMPT.format(
                brand=item.brand, product_name=item.product_name,
                size_form=item.size_form, variant=item.variant,
                pack_qty=item.pack_qty,
                mpn=(item.mpn if (getattr(item, "mpn_verified", False) and item.mpn) else "(not specified)"),
                description=item.description,
                pages=pages), max_tokens=_extract_out_tokens(len(chunk)))
        except RuntimeError as e:
            if "413" in str(e) or "payload too large" in str(e).lower():
                if len(chunk) > 1:
                    mid = len(chunk) // 2
                    log.warning("SKU %s — 413 on %d pages; splitting into %d+%d and retrying",
                                item.schein_sku, len(chunk), mid, len(chunk) - mid)
                    _run(chunk[:mid], per_page)
                    _run(chunk[mid:], per_page)
                    return
                # single page still too big: shrink its text and try once more
                if per_page > 800:
                    log.warning("SKU %s — 413 on a single page; shrinking text %d→800 and retrying",
                                item.schein_sku, per_page)
                    _run(chunk, 800)
                    return
                log.error("SKU %s — single page still 413 at 800 chars; skipping just this page (%s)",
                          item.schein_sku, chunk[0].url[:60])
                return
            log.error("Groq extract+validate failed for SKU %s: %s", item.schein_sku, e)
            return
        except Exception as e:
            log.error("Groq extract+validate failed for SKU %s: %s", item.schein_sku, e)
            return
        for d in _results(data):
            try:
                c = chunk[int(d["idx"])]
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            _apply_ev_result(item, c, d)
            _save(c, d)

    def _escalate(cands):
        """Re-check only the low-confidence / unreliable freshly-extracted pages on
        the scarce 70b model (escalation-only — 8b did the bulk). Skips price_locked
        pages (price already authoritative) and stops for the run once 70b's daily
        quota is hit so we don't waste a probe on every remaining item."""
        if not (ESCALATE_ENABLED and cands) or _esc_state["off"]:
            return
        picks = [c for c in cands
                 if not getattr(c, "price_locked", False)
                 and (getattr(c, "confidence", 0) < ESCALATE_CONF
                      or getattr(c, "price_unreliable", False)
                      or (getattr(c, "criteria", None)
                          and c.match_type == "exact"
                          and not all(c.criteria.values())))]
        if not picks:
            return
        for c in picks[:ESCALATE_MAX]:
            try:
                data = _ask_json(EXTRACT_VALIDATE_PROMPT.format(
                    brand=item.brand, product_name=item.product_name,
                    size_form=item.size_form, variant=item.variant,
                    pack_qty=item.pack_qty,
                mpn=(item.mpn if (getattr(item, "mpn_verified", False) and item.mpn) else "(not specified)"),
                    description=item.description,
                    pages=f'--- PAGE idx=0 (url: {c.url}) ---\n{_slice(c, PER_PAGE)}'),
                    max_tokens=_extract_out_tokens(1), force_model=ESCALATE_MODEL)
            except Exception as e:
                _esc_state["off"] = True
                log.info("SKU %s — escalation to %s unavailable (%s); keeping 8b results",
                         item.schein_sku, ESCALATE_MODEL, e)
                return
            for d in _results(data):
                _apply_ev_result(item, c, d)
                _save(c, d)

    for start in range(0, len(todo), CHUNK):
        _run(todo[start:start + CHUNK], PER_PAGE)
    _escalate(todo)
    emit("groq", action="complete", phase="extract_validate", provider=LLM_PROVIDER,
         sku=item.schein_sku, pages=len(todo),
         message=f"{LLM_PROVIDER.upper()} validation done for SKU {item.schein_sku}")
    return


_PAGE_PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
# Login-flavored rejection wording (used to spare aggregator pages that carry a
# public price from a login-based LLM rejection — Change 1).
_LOGIN_NOTE_RE = re.compile(r"log\s?in|login|sign\s?in|members?\b|account|register|sign[\s\-]?up", re.I)

# Pack-size tokens on a product page ("100-pack", "100/Pk", "50 pack", "4/Bx").
EXTRACT_PACK_MASK = os.environ.get("EXTRACT_PACK_MASK", "1") not in ("0", "false", "False", "")
_PACK_TOKEN_RE = re.compile(
    r"(?<![\d.$])(\d{1,4})\s*[-/ ]?\s*"
    r"(?:pk|pks|pack|packs|bx|box|boxes|ct|cnt|count|cs|case)\b", re.I)


def _mask_wrong_pack(md: str, item) -> str:
    """PACK-AWARE extraction guard. On a page that lists MULTIPLE pack sizes (e.g.
    safco's Clinpro: a 100-pack at $272.99 AND a 50-pack at $137.49), blank the
    prices that belong to a pack size DIFFERENT from the order, so the model can't
    grab the wrong pack's price for a 100/Pk order. The nearest pack-size token
    before each price decides which pack it belongs to.

    No-op unless the page has ≥2 distinct pack sizes AND the ordered size is
    present — so normal single-pack pages are untouched. Order-independent (masks a
    wrong-pack section whether it sits above or below the ordered one)."""
    pq = getattr(item, "pack_qty", None)
    if not (EXTRACT_PACK_MASK and pq) or not md:
        return md
    pqs = str(int(pq))
    packs = [(m.start(), m.group(1)) for m in _PACK_TOKEN_RE.finditer(md)]
    sizes = {s for _, s in packs}
    if pqs not in sizes or len(sizes) < 2:
        return md
    lookback = int(os.environ.get("PACK_MASK_LOOKBACK", "180"))

    def _owner(pos):
        owner = None
        for p, s in packs:
            if p <= pos and pos - p <= lookback and (owner is None or p > owner[0]):
                owner = (p, s)
        return owner[1] if owner else None

    out, last, masked = [], 0, 0
    for m in _PAGE_PRICE_RE.finditer(md):
        own = _owner(m.start())
        if own is not None and own != pqs:
            out.append(md[last:m.start()]); out.append("$—"); last = m.end(); masked += 1
    if not masked:
        return md
    out.append(md[last:])
    return "".join(out)


def _variant_anchor(md: str, item) -> Optional[int]:
    """Offset of the ordered-variant row on a multi-product page, or None.

    Finds the buyer's exact variant token (e.g. "A3") as a standalone token —
    NOT matching a longer sibling like "A3.5" — that has a price within
    EXTRACT_VAR_PRICE_GAP chars (i.e. a real priced row, not a breadcrumb or
    spec mention). Returns the first such offset so the extractor can window
    there instead of on the page's first/cheapest price. None when there's no
    ordered variant or no priced variant row (caller keeps its normal slice)."""
    if not EXTRACT_VARIANT_ANCHOR:
        return None
    v = (getattr(item, "variant", "") or "").strip()
    # need a discriminating token; 1-char variants ("A") are too noisy to anchor
    if len(v) < 2:
        return None
    # standalone token: not glued to a letter/digit on the left, and on the right
    # not followed by a digit or a dot (so "A3" won't match "A3.5" or "A30")
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![0-9.])", re.I)
    for m in pat.finditer(md):
        if _PAGE_PRICE_RE.search(md[m.start(): m.start() + EXTRACT_VAR_PRICE_GAP]):
            return m.start()
    return None


def _price_on_page(md: str, price: float, tol: float = 0.02) -> bool:
    """True if `price` (or a near value) actually appears as a $ token in the
    scraped markdown. Used to catch prices the model invented or pulled from a
    related item. Deliberately lenient — returns True when it can't check — so
    it only ever flags a price, never hides a real one:
      - no markdown / no price → can't check → True
      - page has no parseable $ tokens (JS-only price, odd formatting) → True
      - superscript-cents mangling ("$18¹⁵" → "$18") → whole-dollar match accepted
    """
    if not md or not price or price <= 0:
        return True
    toks = []
    for m in _PAGE_PRICE_RE.finditer(md):
        try:
            toks.append(float(m.group(1).replace(",", "")))
        except ValueError:
            continue
    if not toks:
        return True
    has_fraction = abs(price - round(price)) >= 0.01
    for t in toks:
        if abs(t - price) <= max(0.01, tol * price):
            return True
        # cents dropped by markdown (superscript) → accept whole-dollar token
        if has_fraction and abs(t - round(price)) < 0.01:
            return True
    return False


def _apply_ev_result(item, c, d):
    """Apply one extract+validate JSON result object to a candidate."""
    # ---- extraction half ----
    if d.get("requires_login_for_price"):
        # CHANGE 1: aggregator marketplaces (net32 / supplyclinic) PUBLISH public
        # seller prices — their "log in for account pricing" prompt must not cause
        # the LLM to drop the page (it was rejecting genuinely-priceable supplyclinic
        # listings). For those domains, ignore the login flag and keep the page so
        # its locked / structured / regex price stands. Every OTHER domain still
        # gets rejected when the price is truly behind a login wall.
        from . import search as _search
        if not _search._aggregator_domain(getattr(c, "url", "") or ""):
            c.match_type = "rejected"
            c.rejected_reason = "login required for pricing (public pricing only)"
            return
    c.scraped_product_name = d.get("product_name") or c.title
    c.scraped_variant = d.get("variant")
    ov = (getattr(c, "_ordered_variant", "") or item.variant or "").strip().lower()
    if ov:
        hay = f"{d.get('product_name','')} {d.get('variant','')}".lower()
        toks = [w for w in re.split(r"[^a-z0-9]+", ov) if len(w) > 2]
        if toks and not any(w in hay for w in toks):
            c.variant_unverified = True
    price = d.get("price")
    # A price is already DETERMINED when it came from the aggregator seller table
    # (price_locked) OR a single-offer JSON-LD/og price (price_structured). In both
    # cases the number is authoritative and the LLM — fed only a small validation
    # slice — must NOT overwrite it; it just judges the variant/criteria.
    det_price = bool(getattr(c, "price_locked", False) or getattr(c, "price_structured", False))
    if det_price:
        vp = c.price
    else:
        if isinstance(price, (int, float)):
            vp = float(price)
        elif isinstance(price, str):
            cleaned = price.replace(",", "").replace("$", "").strip()
            _m = re.search(r"\d+(?:\.\d+)?", cleaned)
            vp = float(_m.group(0)) if _m else None
        else:
            vp = None
    if vp and not det_price:
        # Sanity cross-check: the search-snippet price (c.price, from discovery)
        # is an INDEPENDENT signal. If the AI's page price diverges wildly from
        # it, ONE of them is wrong (AI grabbed a banner/related-item price, or
        # the snippet was stale) — we can't tell which, so we keep the page price
        # for display but mark the candidate UNRELIABLE so it can't headline as a
        # trusted "best price" and is clearly flagged for manual verification.
        snippet = c.price if (c.price and c.price > 0) else None
        c.price = vp
        # GROUNDING: the extracted price must actually appear on the scraped page.
        # If it doesn't, the model likely grabbed a number from a related item,
        # banner, or these instructions — flag it (it still shows, with a ⚠, for
        # manual review) so it can't be presented as a confirmed exact match.
        md = getattr(c, "scraped_markdown", None)
        grounded = bool(md and _price_on_page(md, vp))
        if md and not grounded:
            c.price_unreliable = True
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                f"PRICE UNRELIABLE — extracted ${vp:.2f} does not appear on the "
                f"scraped page; likely a related-item or mis-read price, VERIFY MANUALLY")
        # Snippet-divergence is only a red flag when the page price is NOT grounded.
        # A price that DOES appear on the scraped page is page-verified truth — the
        # divergent number is the (stale/unrelated) discovery SNIPPET, not the page
        # price — so trust the grounded page price and don't flag it (this is what
        # wrongly flagged correct net32/aggregator prices like Mity $10.65).
        if snippet and not grounded:
            ratio = vp / snippet
            if ratio > 1.8 or ratio < 0.55:
                c.price_unreliable = True
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"PRICE UNRELIABLE — AI page price ${vp:.2f} vs listing "
                    f"${snippet:.2f} ({ratio:.1f}×); one is wrong, VERIFY MANUALLY")
    elif not det_price:
        # Groq could not read a price off THIS page. If a search-snippet price is
        # still attached (set during discovery), it is NOT page-verified — keep it
        # for display but flag it so it can never be treated as a confirmed exact
        # match. This is the exact failure mode behind the wrong $9.95 glove row:
        # the real page price sat past the old truncation window, Groq returned
        # null, and the unverified snippet price slipped through looking clean.
        if c.price and c.price > 0 and getattr(c, "scraped_markdown", None):
            c.price_unreliable = True
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                "PRICE UNVERIFIED — no price could be read off the product page; "
                "value shown is from the search listing only, VERIFY MANUALLY")
    op = d.get("original_price")
    if isinstance(op, str):
        _m = re.search(r"\d+(?:\.\d+)?", op.replace(",", "").replace("$", ""))
        op = float(_m.group(0)) if _m else None
    if isinstance(op, (int, float)) and op > 0 and not getattr(c, "price_locked", False):
        c.original_price = float(op)
        if vp and op > vp:
            pct = round((op - vp) / op * 100)
            note = f"on sale: was ${op:.2f}, now ${vp:.2f} ({pct}% off)"
            c.pack_condition = (c.pack_condition + " · " + note) if c.pack_condition else note
    if d.get("pack_quantity") is not None:
        try:
            c.pack_qty = int(float(d["pack_quantity"]))
        except (TypeError, ValueError):
            pass
    if d.get("in_stock") is not None and not getattr(c, "price_locked", False):
        c.in_stock = bool(d["in_stock"])
    if d.get("minimum_order_condition"):
        c.pack_condition = d["minimum_order_condition"]
    # ---- validation half ----
    c.match_type = d.get("match_type", "rejected")
    try:
        c.confidence = int(float(d.get("confidence", 0)))
    except (TypeError, ValueError):
        c.confidence = 0
    c.criteria = {k: bool(d.get(k)) for k in
                  ("brand_match", "name_match", "size_form_match", "pack_match")}
    if d.get("notes"):
        c.notes = ((c.notes + " · ") if c.notes else "") + d["notes"]
    if c.match_type == "rejected":
        c.rejected_reason = d.get("notes") or c.rejected_reason
        # CHANGE 1 (completed): the LLM also drops aggregator pages by returning
        # match_type="rejected" with a LOGIN reason (not the boolean flag handled
        # at the top). On net32/supplyclinic that wrongly discards a page carrying
        # a public price. So for an aggregator domain, when the rejection reason is
        # LOGIN-flavored AND a price is present, demote to "approximate" (keep it as
        # an option) instead of rejecting. Product/variant rejections (no login
        # wording) are NOT overridden, so a wrong-product net32 lock still drops.
        if (c.price and c.price > 0
                and _LOGIN_NOTE_RE.search(c.rejected_reason or "")):
            from . import search as _search
            if _search._aggregator_domain(getattr(c, "url", "") or ""):
                c.match_type = "approximate"
                c.rejected_reason = None
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    "kept despite a login note — aggregator shows a public price; "
                    "VERIFY the in-cart total")


# --------------------------------------------- manual (no-Groq) fallback ---
# Last-resort price extraction from scraped markdown WITHOUT any AI, used when
# Groq is fully rate-limited (circuit breaker tripped). It mirrors the Groq
# pricing rules as far as regex can: skip struck-through originals, prefer the
# current/sale/"Now" price, take the price nearest the product title. Every
# price it sets is flagged unverified so it can NEVER be treated as an exact
# match — it only ensures the report shows *a* price instead of nothing.

_PRICE_TOKEN = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")
# struck-through markdown: ~~$280~~  → the original/was price, must be skipped
_STRIKE = re.compile(r"~~\s*\$\s?\d[\d,]*(?:\.\d{2})?\s*~~")
# words that mark a price as the OLD/original one (skip these)
_WAS_NEAR = re.compile(r"(was|reg(?:ular)?|orig(?:inal)?|list|msrp|compare)\b", re.I)
# words that mark the CURRENT price to prefer
_NOW_NEAR = re.compile(r"(now|sale|today|your price|our price|special)\b", re.I)


def _manual_price_from_markdown(md: str) -> tuple[float | None, float | None, str]:
    """Return (current_price, original_price, note) from markdown without AI.
    Heuristics mirror the Groq prompt rules; result is always flagged unverified
    by the caller. Returns (None, None, ...) if no usable price is found."""
    if not md:
        return None, None, "no content"
    # 1) collect struck-through prices — these are originals, never the answer
    struck = set()
    for m in _STRIKE.finditer(md):
        mm = _PRICE_TOKEN.search(m.group(0))
        if mm:
            struck.add(mm.group(1))
    # 2) gather all price tokens with their position + nearby words
    candidates = []   # (price_float, raw_str, score)
    for m in _PRICE_TOKEN.finditer(md):
        raw = m.group(1)
        if raw in struck:
            continue                       # skip struck-through originals
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if val <= 0:
            continue
        # The label that owns a price PRECEDES it ("Was $99", "Your price: $79"),
        # so weight the text immediately BEFORE the price far more than text
        # after it. Stop the preceding window at any INTERVENING price token, so
        # a previous price's label (e.g. "MSRP $50") can't bleed onto this one.
        before = md[max(0, m.start() - 24):m.start()]
        prev_price = _PRICE_TOKEN.search(before)
        if prev_price:
            before = before[prev_price.end():]   # only text AFTER the prior price
        after = md[m.end():min(len(md), m.end() + 12)]
        score = 0.0
        if _NOW_NEAR.search(before):
            score += 5                      # "now/sale/your price" right before → strong prefer
        elif _NOW_NEAR.search(after):
            score += 1
        if _WAS_NEAR.search(before):
            score -= 5                      # "was/reg/msrp" right before → strong avoid
        elif _WAS_NEAR.search(after):
            score -= 1
        # earlier in the page = nearer the product title/buy box = more likely
        # the main price; small weight so a label always outranks position
        score += max(0.0, 1.5 - (m.start() / 800))
        candidates.append((val, raw, score))
    if not candidates:
        return None, None, "no price token found"
    # pick the highest-scoring; tie-break toward the EARLIER price (nearer title)
    best = max(candidates, key=lambda t: t[2])
    # an original price, if we saw a struck-through one, for the "was/now" note
    original = None
    if struck:
        try:
            original = max(float(s.replace(",", "")) for s in struck)
        except ValueError:
            original = None
    note = "price auto-extracted from page text (AI unavailable) — UNVERIFIED, confirm manually"
    if original and original > best[0]:
        note += f"; appears on sale (was ${original:.2f})"
    return best[0], original, note


def manual_extract_fallback(item, candidates: List[PriceCandidate]) -> int:
    """No-AI fallback: fill price from scraped markdown for candidates that have
    markdown but never got AI-extracted (Groq exhausted). Everything is flagged
    unverified and forced to 'approximate' so it can't be reported as exact.
    Returns how many candidates were given a price this way."""
    n = 0
    for c in candidates:
        # 1) STRUCTURED-DATA price (JSON-LD / og-meta captured at scrape time):
        #    rescues nav-bloated store pages whose price never reached the
        #    markdown. Only fills a gap — never overrides a Groq or aggregator
        #    price — and is flagged unverified so it can't be reported as exact.
        sp = getattr(c, "structured_price", None)
        if (c.price is None and sp and not getattr(c, "price_locked", False)
                and c.match_type != "rejected"):
            c.price = sp
            if not c.scraped_product_name and getattr(c, "structured_name", None):
                c.scraped_product_name = c.structured_name
            c.variant_unverified = True
            if c.match_type in (None, "", "exact"):
                c.match_type = "approximate"   # structured price isn't variant-verified
            c.criteria = c.criteria or {"brand_match": False, "name_match": False,
                                        "size_form_match": False, "pack_match": False}
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                "price $%.2f from page structured data (JSON-LD/og-meta) — "
                "markdown had no usable price" % sp)
            n += 1
            continue
        md = getattr(c, "scraped_markdown", None)
        # only fill ones the AI didn't already handle (no scraped_product_name) and
        # whose price is NOT already authoritative — never let the regex guess
        # overwrite an aggregator-locked or single-offer structured price.
        if (not md or c.scraped_product_name is not None or c.match_type == "rejected"
                or getattr(c, "price_locked", False) or getattr(c, "price_structured", False)):
            continue
        price, original, note = _manual_price_from_markdown(md)
        if price is None:
            continue
        c.price = price
        if original and original > price:
            c.original_price = original
        c.variant_unverified = True        # never exact — variant not AI-confirmed
        c.match_type = "approximate"
        c.confidence = 0
        c.criteria = c.criteria or {"brand_match": False, "name_match": False,
                                    "size_form_match": False, "pack_match": False}
        c.notes = ((c.notes + " · ") if c.notes else "") + note
        n += 1
    if n:
        log.warning("SKU %s — manual fallback: auto-extracted %d price(s) from page "
                    "text (Groq unavailable); all flagged UNVERIFIED", item.schein_sku, n)
    return n