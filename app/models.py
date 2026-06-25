"""Data models for the Dental Supply Price Intelligence System (Stages 1+2)."""
from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class OrderLineItem(BaseModel):
    qty: int
    schein_sku: str
    description: str
    uom: str
    unit_price: float
    extended_price: float
    # Enriched by AI parsing layer
    brand: Optional[str] = None
    product_name: Optional[str] = None
    size_form: Optional[str] = None          # e.g. "compules", "59ml bottle"
    pack_qty: Optional[int] = None           # e.g. 20 (from "20/Pk")
    pack_unit: Optional[str] = None          # e.g. "Pk"
    variant: Optional[str] = None            # shade/color/size e.g. "A2", "Green 6.5mm"
    mpn: Optional[str] = None                # manufacturer part number
    mpn_verified: bool = False               # MPN confirmed (internal-consistency + page-consensus) → safe to REJECT conflicting listings; unverified = discovery hint only
    search_query: Optional[str] = None
    generic_query: Optional[str] = None   # brand-stripped query for house-brand items
    mpn_query: Optional[str] = None       # MPN-based precision query


class ParsedOrder(BaseModel):
    source_file: str
    reference: Optional[str] = None
    ship_to_name: Optional[str] = None
    order_date: Optional[str] = None
    total_price: Optional[float] = None
    items: List[OrderLineItem] = Field(default_factory=list)

    @property
    def computed_total(self) -> float:
        return round(sum(i.extended_price for i in self.items), 2)


class PriceCandidate(BaseModel):
    title: str
    url: str
    source_site: str
    price: Optional[float] = None
    pack_qty: Optional[int] = None
    pack_condition: Optional[str] = None     # e.g. "6-pack price", "case of 4 required"
    scraped_product_name: Optional[str] = None
    scraped_variant: Optional[str] = None
    scraped_markdown: Optional[str] = None  # raw page markdown for batched Groq extraction
    structured_price: Optional[float] = None  # price from JSON-LD/og-meta (rescues nav-bloated store pages)
    structured_name: Optional[str] = None     # product name from JSON-LD/og-meta when markdown had none
    price_structured: bool = False        # price set from a SINGLE-offer JSON-LD/og price (deterministic + accurate); AI validates but must not overwrite it
    out_of_stock: bool = False            # listing is out of stock / no longer available — not a buyable price, must never headline as the recommended best
    original_price: Optional[float] = None   # pre-discount/struck-through price if page was on sale
    variant_unverified: bool = False     # page didn't confirm the ordered variant
    variant_conflict: bool = False       # page CONFLICTS with ordered variant (wrong color/shade/size/flavor) — hard-exclude from price_match, route to alternate
    pack_conflict: bool = False          # page sells a DIFFERENT pack size than ordered (10-count vs a 300/Bx order) — price not comparable, hard-exclude from price_match, route to alternate
    mpn_confirmed: bool = False          # the order's MPN appears in this candidate's URL/name — definitive same-product identity; price_match-eligible regardless of the LLM's criteria
    price_unreliable: bool = False        # AI price wildly diverges from listing — verify
    price_locked: bool = False            # price set deterministically (aggregator table parser); AI must not overwrite it
    is_generic_equivalent: bool = False  # same product type, different (or no) brand
    in_stock: Optional[bool] = None
    # Lowest-priced sellers EXCLUDED from the lock because they are backordered /
    # out-of-stock (net32 flips a seller between "Backordered" and "Long Handling
    # Time" by scrape location). Captured so the report can surface them as a
    # clearly labelled secondary row instead of silently dropping the price.
    # Each entry: {"price": float, "status": str, "vendor": str}.
    backorder_options: List[dict] = Field(default_factory=list)
    # Verdict — assigned by real validation, never defaulted
    match_type: str = "unverified"           # exact | approximate | rejected | unverified
    confidence: int = 0
    criteria: dict = Field(default_factory=dict)   # brand/name/size/pack booleans
    notes: Optional[str] = None
    rejected_reason: Optional[str] = None


class ItemResult(BaseModel):
    item: OrderLineItem
    candidates: List[PriceCandidate] = Field(default_factory=list)
    best_exact: Optional[PriceCandidate] = None
    flagged_sites: List[str] = Field(default_factory=list)
    routed_to_alternate: bool = False


class EquivalencyEntry(BaseModel):
    schein_sku: str
    equivalent_name: str
    equivalent_brand: Optional[str] = None
    notes: Optional[str] = None


class EquivalencyFinding(BaseModel):
    item: OrderLineItem
    equivalent_name: str
    confidence_level: str                    # exact_equivalent | close_equivalent | possible_alternative
    basis: str
    supplier: Optional[str] = None
    url: Optional[str] = None
    price: Optional[float] = None
    pack_condition: Optional[str] = None
    est_savings_total: Optional[float] = None