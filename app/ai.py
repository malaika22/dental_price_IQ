"""AI reasoning layer (Groq — free tier friendly).

Same three responsibilities and the same matching logic as before:
1. parse_items_batch   — structured brand / product / size-form / pack qty /
   variant / MPN + search query from raw Schein descriptions.
2. validate_candidates — the four trust criteria (brand, product name,
   size/form, pack qty) applied against real scraped competitor data
   (Issue 4: no defaulted "Exact"/95/75 — confidence must vary with evidence).
3. evaluate_equivalency — exact_equivalent / close_equivalent /
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
from typing import List

from groq import Groq, RateLimitError, APIStatusError
try:                       # connection/timeout error types (present in modern SDKs)
    from groq import APITimeoutError, APIConnectionError
except ImportError:        # very old SDK fallback — treat as generic, retryable
    class APITimeoutError(Exception):
        pass
    class APIConnectionError(Exception):
        pass

from .models import OrderLineItem, PriceCandidate
from . import db

log = logging.getLogger(__name__)

_DEFAULT_MODELS = "llama-3.1-8b-instant,openai/gpt-oss-20b,llama-3.3-70b-versatile"
MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS", os.environ.get("GROQ_MODEL", _DEFAULT_MODELS)).split(",") if m.strip()]
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
# Run-level circuit breaker: when the LAST model (no failover left) gets hard
# rate-limited, every subsequent call would grind through MAX_RETRIES×60s of
# backoff — and with split-retry that multiplies into 10-15min freezes on a
# single item. Once tripped, calls fail fast for a short cooldown so the run
# keeps moving (candidates stay unverified rather than the pipeline hanging).
_groq_cb = {"tripped_until": 0.0}
GROQ_CB_COOLDOWN = int(os.environ.get("GROQ_COOLDOWN", "90"))  # seconds to fail-fast after last-model wall
GROQ_MAX_CYCLES = int(os.environ.get("GROQ_MAX_CYCLES", "3"))  # full passes through the model chain before giving up
# Run-level circuit breaker: once set, the last model has been hard rate-limited
# and every call grinding through 5×60s backoffs would freeze the run for many
# minutes. After it trips, calls fail fast (short single wait, then raise) so the
# pipeline finishes promptly with unverified candidates instead of hanging.
_groq_exhausted = {"v": False, "until": 0.0}
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

_client: Groq | None = None
_lock = threading.Lock()
_last_call = 0.0


def client() -> Groq:
    global _client
    if _client is None:
        # max_retries=0 — disable the SDK's OWN retry layer so our paced
        # backoff (_pace + _ask_json) is the single source of retry truth.
        # Stacking both layers double-hammers the endpoint and wastes RPM.
        # timeout=GROQ_TIMEOUT — give big batches room to complete so a slow
        # response is never truncated/aborted (which loses the whole chunk's
        # data and lets stale snippet prices through).
        _client = Groq(max_retries=0, timeout=GROQ_TIMEOUT)  # uses GROQ_API_KEY env var
    return _client


def _pace():
    """Serialize + space requests so parallel item workers respect free-tier RPM."""
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


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
    # Escalation path: single forced model, short retry, NO failover/cycling.
    if force_model:
        for attempt in range(2):
            _pace()
            try:
                resp = client().chat.completions.create(
                    model=force_model, max_tokens=max_tokens, temperature=0,
                    response_format={"type": "json_object"}, messages=messages,
                )
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
            except (APITimeoutError, APIConnectionError, json.JSONDecodeError) as e:
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
                try:
                    resp = client().chat.completions.create(
                        model=model, max_tokens=max_tokens, temperature=0,
                        response_format={"type": "json_object"}, messages=messages,
                    )
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
                except (APITimeoutError, APIConnectionError) as e:
                    # A timeout/connection drop is transient: the request was cut
                    # off before a (valid) response arrived. Retry with a short
                    # backoff instead of letting the exception bubble up and lose
                    # the whole chunk's extracted data. Fail over to the next model
                    # only after a couple of failures on this one.
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
    by_sku = {}
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        lines = "\n".join(
            f'{i.schein_sku} | "{i.description}" | UOM {i.uom}' for i in chunk
        )
        try:
            data = _ask_json(PARSE_PROMPT.format(lines=lines), max_tokens=4000)
        except Exception:
            log.exception("Groq parse chunk failed (items %d–%d) — falling back to raw descriptions",
                          start + 1, start + len(chunk))
            continue
        for d in _results(data):
            by_sku[str(d.get("sku"))] = d
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


# ------------------------------------------------------------- validation ---

VALIDATE_PROMPT = """You are validating competitor product matches for a dental
supply price comparison. The reference product (ordered from Henry Schein) is:

  Brand: {brand}
  Product: {product_name}
  Size/form: {size_form}
  Variant (shade/color/size): {variant}
  Pack quantity: {pack_qty}
  Original description: "{description}"

Below are scraped competitor candidates. For EACH candidate decide the four
trust criteria strictly:
  brand_match, name_match, size_form_match, pack_match (true/false each).

Rules:
- ALL FOUR true => match_type "exact". Any false => "approximate" if same
  product family, else "rejected".
- A different shade/variant (e.g. A2 vs A3, Green vs Yellow tips, S vs M
  gloves) means name_match=false. The page content (scraped_product_name /
  scraped_variant) is authoritative over the search-result title when they
  disagree — note such disagreement.
- A different pack quantity is pack_match=false even if cheaper. If the price
  is for a larger pack/case, set pack_condition text accordingly.
- confidence: integer 0-100 reflecting how certain YOU are in the verdict,
  based on evidence quality. Do not default; vary it with the evidence.
- If the reference product has no identifiable brand (generic/house item,
  Brand is null), set brand_match=true when the candidate is the same generic
  product type.
- Marketing-name variations of the SAME product line are NOT a name mismatch
  (e.g. "Luxatemp Plus" vs "Luxatemp Automix Plus", "Glide" vs "Glide
  Pro-Health") when the MPN, specs, or variant confirm the same product.
- CONSISTENCY: match_type may be "exact" ONLY when all four booleans are true.

Return ONLY a JSON object: {{"results": [...]}}, same order as candidates,
objects with keys: idx (int, echo), match_type, confidence, brand_match,
name_match, size_form_match, pack_match, pack_condition (string|null),
notes (string).

Candidates:
{candidates}
"""


def validate_candidates(item: OrderLineItem, cands: List[PriceCandidate]) -> List[PriceCandidate]:
    if not cands:
        return cands
    blob = "\n".join(
        json.dumps({
            "idx": i, "title": c.title, "url": c.url, "site": c.source_site,
            "price": c.price, "scraped_product_name": c.scraped_product_name,
            "scraped_variant": c.scraped_variant, "pack_qty": c.pack_qty,
        }) for i, c in enumerate(cands)
    )
    try:
        data = _ask_json(VALIDATE_PROMPT.format(
            brand=item.brand, product_name=item.product_name,
            size_form=item.size_form, variant=item.variant,
            pack_qty=item.pack_qty, description=item.description,
            candidates=blob,
        ), max_tokens=3000)
    except Exception:
        log.exception("Groq validate failed for SKU %s — candidates left unvalidated", item.schein_sku)
        return cands
    for d in _results(data):
        try:
            c = cands[int(d["idx"])]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        c.match_type = d.get("match_type", "rejected")
        try:
            c.confidence = int(float(d.get("confidence", 0)))
        except (TypeError, ValueError):
            c.confidence = 0
        c.criteria = {k: bool(d.get(k)) for k in
                      ("brand_match", "name_match", "size_form_match", "pack_match")}
        c.pack_condition = d.get("pack_condition") or c.pack_condition
        c.notes = d.get("notes")
        if c.match_type == "rejected":
            c.rejected_reason = c.notes
    return cands


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
    instead of paying Groq tokens again."""
    ov = (getattr(c, "_ordered_variant", "") or getattr(item, "variant", "") or "")
    raw = f"{getattr(c, 'url', '')}|{ov.strip().lower()}|{getattr(item, 'pack_qty', '')}"
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
        return

    def _slice(c, per_page):
        """Page text sent to the model, trimmed to save free-tier tokens:
          - price_locked pages: small head slice (price is authoritative from the
            seller-table parser; the LLM only confirms variant/criteria).
          - normal pages: full head UNLESS the price sits past the head window, in
            which case stitch the top (title/variant) + the block around the price
            so a deep price is never truncated away."""
        md = c.scraped_markdown or ""
        if getattr(c, "price_locked", False):
            return md[:EXTRACT_LOCKED_PAGE_CHARS]
        if len(md) <= per_page:
            return md
        if EXTRACT_PRICE_WINDOW:
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
                pack_qty=item.pack_qty, description=item.description,
                pages=pages), max_tokens=3000)
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
                    pack_qty=item.pack_qty, description=item.description,
                    pages=f'--- PAGE idx=0 (url: {c.url}) ---\n{_slice(c, PER_PAGE)}'),
                    max_tokens=3000, force_model=ESCALATE_MODEL)
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
    return


_PAGE_PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)")


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
    if getattr(c, "price_locked", False):
        # Price was set deterministically from the aggregator seller table
        # (net32 / supplyclinic) in search.py. The LLM still judges variant /
        # criteria, but MUST NOT overwrite the locked price or its grounding —
        # the table parser is authoritative here.
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
    if vp and not getattr(c, "price_locked", False):
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
        if md and not _price_on_page(md, vp):
            c.price_unreliable = True
            c.notes = ((c.notes + " · ") if c.notes else "") + (
                f"PRICE UNRELIABLE — extracted ${vp:.2f} does not appear on the "
                f"scraped page; likely a related-item or mis-read price, VERIFY MANUALLY")
        if snippet:
            ratio = vp / snippet
            if ratio > 1.8 or ratio < 0.55:
                c.price_unreliable = True
                c.notes = ((c.notes + " · ") if c.notes else "") + (
                    f"PRICE UNRELIABLE — AI page price ${vp:.2f} vs listing "
                    f"${snippet:.2f} ({ratio:.1f}×); one is wrong, VERIFY MANUALLY")
    elif not getattr(c, "price_locked", False):
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
        md = getattr(c, "scraped_markdown", None)
        # only fill ones the AI didn't already handle (no scraped_product_name)
        if not md or c.scraped_product_name is not None or c.match_type == "rejected":
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