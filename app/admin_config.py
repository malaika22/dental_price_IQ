"""Admin configuration: API keys + supplier sources (dashboard-managed)."""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", str(ROOT / "config")))
ENV_PATH = ROOT / ".env"
SOURCES_PATH = CONFIG_DIR / "supplier_sources.json"
SUPPLIER_SITES_PATH = CONFIG_DIR / "supplier_sites.txt"
EXCLUDED_PATH = CONFIG_DIR / "excluded_domains.txt"

API_KEY_DEFS = [
    {"id": "gemini", "label": "Gemini", "env": "GEMINI_API_KEY", "hint": "Google AI Studio"},
    {"id": "groq", "label": "Groq", "env": "GROQ_API_KEY", "hint": "console.groq.com"},
    {"id": "serpapi", "label": "SerpAPI", "env": "SERPAPI_API_KEY", "hint": "serpapi.com"},
    {"id": "firecrawl", "label": "Firecrawl", "env": "FIRECRAWL_API_KEY", "hint": "firecrawl.dev"},
]

SOURCE_TYPES = ("dental_supplier", "marketplace", "aggregator", "other")

DEFAULT_MARKETPLACES = [
    {"domain": "amazon.com", "enabled": True, "type": "marketplace", "priority": 900, "label": "Amazon"},
    {"domain": "walmart.com", "enabled": True, "type": "marketplace", "priority": 910, "label": "Walmart"},
]

AGGREGATOR_DOMAINS = {"net32.com", "supplyclinic.com"}


def _mask(value: str | None) -> str:
    if not value:
        return ""
    v = value.strip()
    if len(v) <= 8:
        return "••••••••" if v else ""
    return f"{v[:4]}…{v[-4:]}"


def _read_env_file() -> dict[str, str]:
    data: dict[str, str] = {}
    if not ENV_PATH.exists():
        return data
    for line in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def _write_env_key(key: str, value: str) -> None:
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    found = False
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for line in lines:
        if pattern.match(line):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


def get_api_keys_public() -> dict[str, Any]:
    env_file = _read_env_file()
    rows = []
    for d in API_KEY_DEFS:
        raw = os.environ.get(d["env"]) or env_file.get(d["env"]) or ""
        rows.append({
            "id": d["id"],
            "label": d["label"],
            "env": d["env"],
            "hint": d["hint"],
            "configured": bool(raw.strip()),
            "masked": _mask(raw) if raw.strip() else "",
        })
    provider = os.environ.get("LLM_PROVIDER") or env_file.get("LLM_PROVIDER") or "groq"
    return {"keys": rows, "llm_provider": provider}


def update_api_keys(payload: dict[str, Any]) -> dict[str, Any]:
    keys = payload.get("keys") or {}
    for d in API_KEY_DEFS:
        val = keys.get(d["id"])
        if val is None:
            continue
        val = str(val).strip()
        if not val or set(val) <= {"•", ".", "…"} or "…" in val:
            continue
        _write_env_key(d["env"], val)
        if d["env"] == "SERPAPI_API_KEY":
            try:
                from . import search as search_mod
                search_mod.SERPAPI_KEY = val
            except Exception:
                pass
        if d["env"] == "FIRECRAWL_API_KEY":
            try:
                from . import search as search_mod
                search_mod.FIRECRAWL_KEY = val
            except Exception:
                pass

    provider = payload.get("llm_provider")
    if provider in ("groq", "gemini", "openai", "openrouter"):
        _write_env_key("LLM_PROVIDER", provider)

    return get_api_keys_public()


def test_api_key(provider: str) -> dict[str, Any]:
    env_file = _read_env_file()

    def key_for(env_name: str) -> str:
        return (os.environ.get(env_name) or env_file.get(env_name) or "").strip()

    try:
        if provider == "gemini":
            key = key_for("GEMINI_API_KEY")
            if not key:
                return {"ok": False, "message": "GEMINI_API_KEY is not set."}
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                ok = 200 <= resp.status < 300
            return {"ok": ok, "message": "Gemini API reachable." if ok else f"HTTP {resp.status}"}

        if provider == "groq":
            key = key_for("GROQ_API_KEY")
            if not key:
                return {"ok": False, "message": "GROQ_API_KEY is not set."}
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                ok = 200 <= resp.status < 300
            return {"ok": ok, "message": "Groq API reachable." if ok else f"HTTP {resp.status}"}

        if provider == "serpapi":
            key = key_for("SERPAPI_API_KEY")
            if not key:
                return {"ok": False, "message": "SERPAPI_API_KEY is not set."}
            url = f"https://serpapi.com/account.json?api_key={key}"
            with urllib.request.urlopen(url, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return {"ok": True, "message": f"SerpAPI OK — plan={body.get('plan') or 'unknown'}"}

        if provider == "firecrawl":
            key = key_for("FIRECRAWL_API_KEY")
            if not key:
                return {"ok": False, "message": "FIRECRAWL_API_KEY is not set."}
            req = urllib.request.Request(
                "https://api.firecrawl.dev/v1/team/credit-usage",
                headers={"Authorization": f"Bearer {key}"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    return {"ok": True, "message": "Firecrawl API reachable."}
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    return {"ok": False, "message": "Firecrawl rejected the API key (401)."}
                if e.code in (402, 403, 404, 429):
                    return {"ok": True, "message": f"Firecrawl key accepted (HTTP {e.code})."}
                return {"ok": False, "message": f"Firecrawl HTTP {e.code}."}

        return {"ok": False, "message": f"Unknown provider: {provider}"}
    except Exception as e:
        return {"ok": False, "message": str(e)[:240]}


def _parse_sites_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s.lower().removeprefix("www."))
    return out


def _bootstrap_sources() -> list[dict[str, Any]]:
    sites = _parse_sites_file(SUPPLIER_SITES_PATH)
    rows: list[dict[str, Any]] = []
    for i, domain in enumerate(sites):
        stype = "aggregator" if domain in AGGREGATOR_DOMAINS else "dental_supplier"
        rows.append({
            "id": domain,
            "domain": domain,
            "label": domain,
            "enabled": True,
            "type": stype,
            "priority": i + 1,
        })
    for m in DEFAULT_MARKETPLACES:
        if not any(r["domain"] == m["domain"] for r in rows):
            rows.append({
                "id": m["domain"],
                "domain": m["domain"],
                "label": m["label"],
                "enabled": m["enabled"],
                "type": m["type"],
                "priority": m["priority"],
            })
    return rows


def load_sources() -> list[dict[str, Any]]:
    if SOURCES_PATH.exists():
        try:
            data = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                domains = {r.get("domain") for r in data}
                for m in DEFAULT_MARKETPLACES:
                    if m["domain"] not in domains:
                        data.append({
                            "id": m["domain"],
                            "domain": m["domain"],
                            "label": m["label"],
                            "enabled": m["enabled"],
                            "type": m["type"],
                            "priority": m["priority"],
                        })
                return data
        except Exception:
            log.exception("Failed to read supplier_sources.json — rebuilding")
    rows = _bootstrap_sources()
    save_sources(rows)
    return rows


def save_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for i, raw in enumerate(sources):
        domain = str(raw.get("domain") or "").strip().lower().removeprefix("www.")
        if not domain or "/" in domain or " " in domain:
            continue
        stype = raw.get("type") if raw.get("type") in SOURCE_TYPES else "other"
        cleaned.append({
            "id": domain,
            "domain": domain,
            "label": str(raw.get("label") or domain).strip() or domain,
            "enabled": bool(raw.get("enabled", True)),
            "type": stype,
            "priority": int(raw.get("priority") or (i + 1)),
        })
    cleaned.sort(key=lambda r: (r["priority"], r["domain"]))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
    _sync_legacy_files(cleaned)
    _reload_search_globals()
    return cleaned


def _sync_legacy_files(sources: list[dict[str, Any]]) -> None:
    dental = [
        s["domain"] for s in sources
        if s["enabled"] and s["type"] in ("dental_supplier", "aggregator")
    ]
    header = (
        "# Auto-synced from Admin → Supplier Sources. Edit via the dashboard.\n"
    )
    SUPPLIER_SITES_PATH.write_text(header + "\n".join(dental) + "\n", encoding="utf-8")

    mkt = {s["domain"]: s["enabled"] for s in sources if s["type"] == "marketplace"}
    existing: list[str] = []
    if EXCLUDED_PATH.exists():
        for line in EXCLUDED_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                existing.append(line)
                continue
            if s.startswith("#"):
                existing.append(line)
                continue
            dom = s.lower().removeprefix("www.")
            if dom in mkt and mkt[dom]:
                continue
            existing.append(line)

    have = {
        ln.strip().lower().removeprefix("www.")
        for ln in existing
        if ln.strip() and not ln.strip().startswith("#")
    }
    for dom, enabled in mkt.items():
        if not enabled and dom not in have:
            existing.append(dom)

    if "henryschein.com" not in have:
        existing.insert(0, "henryschein.com")

    EXCLUDED_PATH.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")


def _reload_search_globals() -> None:
    try:
        from . import search as search_mod
        search_mod.EXCLUDED = search_mod.load_excluded_domains()
        search_mod.SUPPLIER_SITES = search_mod.load_supplier_sites()
        sources = load_sources()
        any_mkt = any(s["enabled"] and s["type"] == "marketplace" for s in sources)
        search_mod.MARKETPLACE_ENABLED = any_mkt
        enabled_mkt = [
            s["domain"] for s in sources
            if s["enabled"] and s["type"] == "marketplace"
        ]
        search_mod.MARKETPLACE_DOMAINS = enabled_mkt or ["amazon.com", "walmart.com"]
        log.info(
            "Reloaded supplier sources — %d sites, marketplaces=%s",
            len(search_mod.SUPPLIER_SITES),
            search_mod.MARKETPLACE_DOMAINS,
        )
    except Exception:
        log.exception("Failed to reload search globals after supplier save")


def classify_source_type(
    domain_or_url: str | None,
    *,
    marketplace: str | None = None,
    is_generic: bool = False,
) -> str:
    if is_generic:
        return "Generic/Equivalent"
    if marketplace:
        return "Marketplace"
    raw = (domain_or_url or "").strip().lower()
    if "://" in raw:
        try:
            from urllib.parse import urlparse
            raw = urlparse(raw).hostname or raw
        except Exception:
            pass
    domain = raw.removeprefix("www.")
    if not domain or domain in ("—", "-", "n/a"):
        return "Other"
    if any(domain == d or domain.endswith("." + d) for d in ("amazon.com", "walmart.com", "ebay.com")):
        return "Marketplace"
    if any(domain == d or domain.endswith("." + d) for d in AGGREGATOR_DOMAINS):
        return "Aggregator"
    try:
        from . import search as search_mod
        if search_mod.is_trusted_supplier(f"https://{domain}/"):
            return "Dental Supplier"
        if "dent" in domain:
            return "Dental Supplier"
    except Exception:
        if "dent" in domain:
            return "Dental Supplier"
    return "Other"
