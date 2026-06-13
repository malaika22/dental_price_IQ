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

_DEFAULT_MODELS = "llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it"
MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS", os.environ.get("GROQ_MODEL", _DEFAULT_MODELS)).split(",") if m.strip()]
MODEL = MODELS[0]
_model_idx = {"i": 0}
MIN_INTERVAL = float(os.environ.get("GROQ_MIN_INTERVAL", "2.1"))  # ~28 req/min
MAX_RETRIES = 5

_client: Groq | None = None
_lock = threading.Lock()
_last_call = 0.0


def client() -> Groq:
    global _client
    if _client is None:
        # max_retries=0 — disable the SDK's OWN retry layer so our paced
        # backoff (_pace + _ask_json) is the single source of retry truth.
        # Stacking both layers double-hammers the endpoint and wastes RPM.
        _client = Groq(max_retries=0)  # uses GROQ_API_KEY env var
    return _client


def _pace():
    """Serialize + space requests so parallel item workers respect free-tier RPM."""
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _ask_json(prompt: str, max_tokens: int = 4000) -> dict:
    """Ask Groq for JSON with retry + automatic model failover. Each model in
    MODELS has its OWN free-tier rate limit; if the current model stays
    rate-limited past a couple of attempts and another model remains, we fail
    over to it (and remember that for the rest of the run). Only when every
    model is exhausted does the call raise."""
    last_err = None
    messages = [
        {"role": "system",
         "content": "You are a precise dental supply data engine. "
                    "Respond ONLY with valid JSON. No prose, no markdown."},
        {"role": "user", "content": prompt},
    ]
    for mi in range(_model_idx["i"], len(MODELS)):
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
                if more_models and attempt >= 1:
                    log.warning("Groq %s rate-limited — failing over to %s",
                                model, MODELS[mi + 1])
                    break
                ra = None
                try:
                    ra = float(getattr(e, "response", None).headers.get("retry-after"))
                except Exception:
                    ra = None
                backoff = ra if ra else min(8 * (attempt + 1), 60)
                log.warning("Groq %s rate-limited (429) — waiting %ss (attempt %d/%d)",
                            model, round(backoff), attempt + 1, MAX_RETRIES)
                time.sleep(backoff)
            except APIStatusError as e:
                last_err = e
                backoff = min(2 ** attempt * 2, 30)
                log.warning("Groq %s %s — retrying in %ss (attempt %d/%d)",
                            model, type(e).__name__, backoff, attempt + 1, MAX_RETRIES)
                time.sleep(backoff)
            except json.JSONDecodeError as e:
                last_err = e
                log.warning("Groq %s unparseable JSON — retrying (attempt %d)",
                            model, attempt + 1)
        if rate_limited and mi < len(MODELS) - 1:
            _model_idx["i"] = mi + 1
            log.info("Groq failover: primary model is now %s for the rest of this run",
                     MODELS[mi + 1])
            continue
    raise RuntimeError(f"All Groq models exhausted ({', '.join(MODELS)}): {last_err}")


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
        data = _ask_json(PARSE_PROMPT.format(lines=lines), max_tokens=4000)
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
    data = _ask_json(VALIDATE_PROMPT.format(
        brand=item.brand, product_name=item.product_name,
        size_form=item.size_form, variant=item.variant,
        pack_qty=item.pack_qty, description=item.description,
        candidates=blob,
    ), max_tokens=3000)
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
    return _ask_json(EQUIV_PROMPT.format(
        description=item.description, brand=item.brand, pack_qty=item.pack_qty,
        equivalent=equivalent_name, note=note or "", market=market_summary or "none",
    ), max_tokens=400)