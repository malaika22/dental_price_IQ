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
import json
import logging
import os
import re
import threading
import time
from typing import List

from groq import Groq, RateLimitError, APIStatusError

from .models import OrderLineItem, PriceCandidate

log = logging.getLogger(__name__)

MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MIN_INTERVAL = float(os.environ.get("GROQ_MIN_INTERVAL", "2.1"))  # ~28 req/min
MAX_RETRIES = 5

_client: Groq | None = None
_lock = threading.Lock()
_last_call = 0.0


def client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()  # uses GROQ_API_KEY env var
    return _client


def _pace():
    """Serialize + space requests so parallel item workers respect free-tier RPM."""
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _ask_json(prompt: str, max_tokens: int = 4000, label: str = "Groq") -> dict:
    last_err = None
    for attempt in range(MAX_RETRIES):
        _pace()
        try:
            log.debug("%s request (attempt %d/%d, max_tokens=%d)", label, attempt + 1, MAX_RETRIES, max_tokens)
            resp = client().chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": "You are a precise dental supply data engine. "
                                "Respond ONLY with valid JSON. No prose, no markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content or ""
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            try:
                parsed = json.loads(text)
                log.debug("%s call succeeded (%d chars response)", label, len(text))
                return parsed
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.S)   # salvage outermost object
                if m:
                    parsed = json.loads(m.group(0))
                    log.debug("%s call succeeded (salvaged JSON from response)", label)
                    return parsed
                raise
        except (RateLimitError, APIStatusError) as e:
            last_err = e
            backoff = min(2 ** attempt * 2, 30)
            log.warning("%s %s — retrying in %ss (attempt %d/%d)",
                        label, type(e).__name__, backoff, attempt + 1, MAX_RETRIES)
            time.sleep(backoff)
        except json.JSONDecodeError as e:
            last_err = e
            log.warning("%s returned unparseable JSON — retrying (attempt %d)", label, attempt + 1)
        except Exception as e:
            last_err = e
            log.error("%s unexpected error: %s (attempt %d/%d)", label, e, attempt + 1, MAX_RETRIES)
            backoff = min(2 ** attempt * 2, 30)
            time.sleep(backoff)
    log.error("%s FAILED after %d attempts: %s", label, MAX_RETRIES, last_err)
    raise RuntimeError(f"{label} call failed after {MAX_RETRIES} attempts: {last_err}")


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

Order lines:
{lines}
"""


def parse_items_batch(items: List[OrderLineItem], chunk_size: int = 12) -> List[OrderLineItem]:
    """Batched parsing. Chunked to stay inside free-tier output-token limits."""
    total_chunks = (len(items) + chunk_size - 1) // chunk_size
    log.info("Groq parse: %d items in %d chunk(s) (model=%s)", len(items), total_chunks, MODEL)
    by_sku = {}
    failed_chunks = 0
    for chunk_idx, start in enumerate(range(0, len(items), chunk_size), start=1):
        chunk = items[start:start + chunk_size]
        skus = [i.schein_sku for i in chunk]
        log.info(
            "Groq parse chunk %d/%d — SKUs %s …",
            chunk_idx, total_chunks, ", ".join(skus),
        )
        lines = "\n".join(
            f'{i.schein_sku} | "{i.description}" | UOM {i.uom}' for i in chunk
        )
        try:
            data = _ask_json(
                PARSE_PROMPT.format(lines=lines), max_tokens=4000,
                label=f"Groq-parse chunk {chunk_idx}/{total_chunks}",
            )
            parsed = _results(data)
            for d in parsed:
                by_sku[str(d.get("sku"))] = d
            log.info(
                "Groq parse chunk %d/%d SUCCESS — parsed %d/%d items",
                chunk_idx, total_chunks, len(parsed), len(chunk),
            )
        except Exception:
            failed_chunks += 1
            log.exception(
                "Groq parse chunk %d/%d FAILED — SKUs %s",
                chunk_idx, total_chunks, ", ".join(skus),
            )

    enriched = 0
    for it in items:
        d = by_sku.get(it.schein_sku, {})
        if not d:
            log.warning("Groq parse: no data returned for SKU %s — using raw description", it.schein_sku)
            it.search_query = it.description
            continue
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
        enriched += 1
        log.debug(
            "SKU %s enriched — brand=%s query=%r",
            it.schein_sku, it.brand, it.search_query,
        )

    if failed_chunks:
        log.error("Groq parse finished with %d/%d chunk(s) FAILED (%d/%d items enriched)",
                  failed_chunks, total_chunks, enriched, len(items))
    else:
        log.info("Groq parse finished OK — %d/%d items enriched", enriched, len(items))
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

Return ONLY a JSON object: {{"results": [...]}}, same order as candidates,
objects with keys: idx (int, echo), match_type, confidence, brand_match,
name_match, size_form_match, pack_match, pack_condition (string|null),
notes (string).

Candidates:
{candidates}
"""


def validate_candidates(item: OrderLineItem, cands: List[PriceCandidate]) -> List[PriceCandidate]:
    if not cands:
        log.info("SKU %s — Groq validate: no candidates to validate", item.schein_sku)
        return cands
    log.info(
        "SKU %s — Groq validate: %d candidate(s) for %r",
        item.schein_sku, len(cands), item.product_name or item.description[:40],
    )
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
        ), max_tokens=3000, label=f"Groq-validate SKU {item.schein_sku}")
    except Exception:
        log.exception("SKU %s — Groq validate FAILED", item.schein_sku)
        return cands

    validated = 0
    for d in _results(data):
        try:
            c = cands[int(d["idx"])]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        c.match_type = d.get("match_type", "rejected")
        c.confidence = int(d.get("confidence", 0))
        c.criteria = {k: bool(d.get(k)) for k in
                      ("brand_match", "name_match", "size_form_match", "pack_match")}
        c.pack_condition = d.get("pack_condition") or c.pack_condition
        c.notes = d.get("notes")
        if c.match_type == "rejected":
            c.rejected_reason = c.notes
        validated += 1
        log.info(
            "SKU %s — candidate %s: %s (confidence=%d) %s",
            item.schein_sku, c.source_site, c.match_type, c.confidence, c.url[:80],
        )
    log.info("SKU %s — Groq validate SUCCESS (%d/%d verdicts)", item.schein_sku, validated, len(cands))
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
    log.info(
        "SKU %s — Groq equivalency: %r vs %r",
        item.schein_sku, item.description[:40], equivalent_name[:40],
    )
    try:
        result = _ask_json(EQUIV_PROMPT.format(
            description=item.description, brand=item.brand, pack_qty=item.pack_qty,
            equivalent=equivalent_name, note=note or "", market=market_summary or "none",
        ), max_tokens=400, label=f"Groq-equiv SKU {item.schein_sku}")
        log.info(
            "SKU %s — Groq equivalency SUCCESS: %s",
            item.schein_sku, result.get("confidence_level", "?"),
        )
        return result
    except Exception:
        log.exception("SKU %s — Groq equivalency FAILED", item.schein_sku)
        return {"confidence_level": "possible_alternative", "basis": "Groq evaluation failed — manual review required"}
