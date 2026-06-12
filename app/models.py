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
    search_query: Optional[str] = None


class ParsedOrder(BaseModel):
    source_file: str
    reference: Optional[str] = None
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
    in_stock: Optional[bool] = None
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
