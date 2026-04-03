"""
jumia_scraper.py
────────────────
Scrapes missing product data from Jumia for post-QC enrichment.

Supported countries : KE · UG · NG · GH · MA
Scraped fields      : COLOR, COUNT_VARIATIONS, PRODUCT_WARRANTY,
                      WARRANTY_DURATION, MAIN_IMAGE, PRICE, DISCOUNT,
                      RATING, REVIEW_COUNT, STOCK_STATUS,
                      DESCRIPTION, KEY_FEATURES, WHATS_IN_BOX,
                      MODEL, GTIN, WEIGHT, CATEGORY_PATH, IS_OFFICIAL_STORE
"""

from __future__ import annotations

import logging
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Country base URLs ─────────────────────────────────────────────────────────
COUNTRY_URLS: dict[str, str] = {
    "KE": "https://www.jumia.co.ke",
    "UG": "https://www.jumia.ug",
    "NG": "https://www.jumia.com.ng",
    "GH": "https://www.jumia.com.gh",
    "MA": "https://www.jumia.ma",
}

# ── All fields this module can fill ──────────────────────────────────────────
SCRAPABLE_FIELDS: list[str] = [
    "PRICE", "DISCOUNT", "STOCK_STATUS",
    "MODEL", "GTIN", "BRAND", "CATEGORY_PATH", "IS_OFFICIAL_STORE",
    "MAIN_IMAGE", "WEIGHT",
    "COLOR", "COLOR_IN_TITLE", "COUNT_VARIATIONS", "SIZES_AVAILABLE",
    "PRODUCT_WARRANTY", "WARRANTY_DURATION",
    "RATING", "REVIEW_COUNT",
    "SELLER_NAME",
    "DESCRIPTION", "KEY_FEATURES", "KEY_SPECS", "SPECIFICATIONS", "WHATS_IN_BOX",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)

_SKU_CANDIDATES = [
    "PRODUCT_SET_SID", "SellerSku", "SELLER_SKU", "seller_sku",
    "SKU", "Sku", "sku",
]

# ── Spec-label filter ─────────────────────────────────────────────────────────
# Jumia renders product attribute rows (Colour, Size, Material, Brand…) inside
# the same  ul.-pvs  element used for variation pills.  On single-SKU products
# these attribute rows are the ONLY items in that list.  BeautifulSoup's
# get_text() strips the span boundary between label and value, producing
# concatenated strings like "ColourRed" or "Size750ml" with no ": " separator,
# so the old colon-space filter silently lets them through.
#
# This regex matches any text whose first word (with or without following text)
# is a known product-attribute label, catching both:
#   • "Colour Red"  (spaced)
#   • "ColourRed"   (concatenated, no separator)
_SPEC_LABEL_RE = re.compile(
    r"^(colour|color|size|material|main\s+material|weight|model|gtin|"
    r"sku|barcode|dimension|brand|style|gender|age\s+group|"
    r"compatible\s+with|connectivity|power\s+source|voltage|wattage|"
    r"capacity|storage|memory|screen|resolution|battery|warranty|"
    r"type|quantity|number\s+of|set\s+includes|net\s+weight|"
    r"unit\s+count|flavor|scent|finish|texture|pattern|fit\s+type|"
    r"closure\s+type|sleeve\s+length|collar\s+type|outer\s+material|"
    r"sole\s+material|upper\s+material|heel\s+type|toe\s+style|"
    r"production|country|manufacturer|origin|made\s+in|from\s+the|"
    r"item\s+weight|package|box\s+contents|box\s+content|included|"
    r"features?|specification|certified|certification|standard)",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_sku(sku: str) -> str:
    return sku.split("-")[0].strip()


def _find_sku_column(df: pd.DataFrame) -> str | None:
    for col in _SKU_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _needs_fill(df: pd.DataFrame, col: str, threshold: float = 0.5) -> bool:
    if col not in df.columns:
        return True
    filled = (
        df[col].astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .sum()
    )
    return (filled / max(len(df), 1)) < threshold


def _row_is_empty(val) -> bool:
    return not str(val).strip() or str(val).strip().lower() in ("nan", "none", "")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – find product URL via search
# ─────────────────────────────────────────────────────────────────────────────

def _search_product_url(sku: str, base_url: str) -> str | None:
    search_url = f"{base_url}/catalog/?q={_base_sku(sku)}"
    try:
        resp = _SESSION.get(search_url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Search request failed for %s: %s", sku, exc)
        return None

    text = resp.text
    if any(p in text.lower() for p in ("no results", "there are no results", "aucun résultat")):
        return None

    soup = BeautifulSoup(text, "html.parser")
    card = soup.select_one("article.prd a.core")
    if card:
        href = card.get("href", "")
        return (base_url + href) if href.startswith("/") else href or None

    for a in soup.select("a[href*='.html']"):
        href = a.get("href", "")
        if href and "/catalog/" not in href and "/c/" not in href:
            return (base_url + href) if href.startswith("/") else href

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – scrape data from the product page
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_product_page(url: str, base_url: str) -> dict[str, str]:
    import json as _json

    result: dict[str, str] = {}
    try:
        resp = _SESSION.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("Product page request failed for %s: %s", url, exc)
        return result

    soup = BeautifulSoup(resp.text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # ── STRATEGY 1: JSON-LD ───────────────────────────────────────────────────
    ld_data: dict = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            blob = _json.loads(script.string or "")
            if isinstance(blob, list):
                for item in blob:
                    if isinstance(item, dict) and item.get("@type") == "Product":
                        ld_data = item
                        break
            elif isinstance(blob, dict) and blob.get("@type") == "Product":
                ld_data = blob
            if ld_data:
                break
        except Exception:
            continue

    if ld_data:
        img = ld_data.get("image")
        if isinstance(img, list): img = img[0]
        if img: result["MAIN_IMAGE"] = str(img)

        desc = ld_data.get("description", "")
        if desc and len(desc) > 20: result["DESCRIPTION"] = desc.strip()

        brand = ld_data.get("brand", {})
        if isinstance(brand, dict): brand = brand.get("name", "")
        if brand: result["BRAND"] = str(brand).strip()

        offers = ld_data.get("offers", {})
        if isinstance(offers, list): offers = offers[0] if offers else {}
        if isinstance(offers, dict):
            price = offers.get("price") or offers.get("lowPrice", "")
            if price: result["_ld_price_raw"] = str(price).replace(",", "")
            avail = offers.get("availability", "")
            if "InStock" in avail: result["_ld_stock"] = "In Stock"
            elif "OutOfStock" in avail: result["_ld_stock"] = "Out of Stock"

        agg = ld_data.get("aggregateRating", {})
        if isinstance(agg, dict):
            rv = agg.get("ratingValue", "")
            rc = agg.get("reviewCount", "") or agg.get("ratingCount", "")
            if rv: result["RATING"] = str(rv)
            if rc: result["REVIEW_COUNT"] = str(rc)

    # ── STRATEGY 2: embedded window JSON ─────────────────────────────────────
    for script in soup.find_all("script"):
        raw = script.string or ""
        if not raw or len(raw) < 50: continue

        if "dataLayer" in raw and "price" in raw.lower():
            m = re.search(r"dataLayer\s*=\s*(\[.*?\])\s*;", raw, re.DOTALL)
            if not m: m = re.search(r"dataLayer\.push\s*\((\{.*?\})\s*\)", raw, re.DOTALL)
            if m:
                try:
                    dl = _json.loads(m.group(1))
                    if isinstance(dl, list): dl = dl[0] if dl else {}
                    dl = dl.get("ecommerce", dl)
                    dl = dl.get("detail", dl)
                    products = dl.get("products", [dl])
                    p = products[0] if products else dl
                    if not result.get("PRICE") and p.get("price"):
                        result["PRICE"] = str(p["price"]).replace(",", "")
                    if not result.get("BRAND") and p.get("brand"):
                        result["BRAND"] = str(p["brand"])
                    if not result.get("CATEGORY_PATH") and p.get("category"):
                        result["CATEGORY_PATH"] = str(p["category"])
                except Exception: pass

        if "__STORES__" in raw or "window.store" in raw.lower():
            m = re.search(r"__STORES__\s*=\s*(\{.+?\})\s*;", raw, re.DOTALL)
            if m:
                try:
                    store = _json.loads(m.group(1))
                    pdp = store.get("pdp", store)
                    prod = pdp.get("product", pdp)
                    if not result.get("PRICE") and prod.get("price"):
                        result["PRICE"] = str(prod["price"]).replace(",", "")
                    if not result.get("MAIN_IMAGE") and prod.get("image"):
                        result["MAIN_IMAGE"] = str(prod["image"])
                except Exception: pass

    # ── STRATEGY 3: HTML / CSS parsing ───────────────────────────────────────

    # Main image
    if not result.get("MAIN_IMAGE"):
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            result["MAIN_IMAGE"] = og["content"]
        else:
            for img in soup.select("img.-fw.-fh, img[data-src*='jumia']"):
                src = img.get("data-src") or img.get("src", "")
                if src and ("product" in src or "unsafe" in src or "jumia.is" in src):
                    result["MAIN_IMAGE"] = src if src.startswith("http") else base_url + src
                    break

    # Price
    _html_price_found = False
    for sel in ["span.-b.-ltr.-tal.-fs24", "span.prc", "span[data-price]",
                "span.-b.-ltr.-tal", "div.-prc-w span"]:
        el = soup.select_one(sel)
        if el:
            raw_p = re.sub(r"[^\d,.]", "", el.get_text())
            if raw_p:
                try:
                    price_val = float(raw_p.replace(",", ""))
                    if price_val >= 100:
                        result["PRICE"] = f"KSh {int(price_val):,}" if price_val == int(price_val) else f"KSh {price_val:,.2f}"
                        _html_price_found = True
                        break
                except ValueError: pass
    if not _html_price_found and result.pop("_ld_price_raw", None):
        result["PRICE"] = result.get("PRICE", "")

    # Discount
    for sel in ["span.bdg._dsct._sm", "span._dsct", "[class*='discount']"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if "%" in txt:
                result["DISCOUNT"] = txt
                break

    # Rating
    if not result.get("RATING"):
        for sel in ["div.stars._s", "[class*='stars'] span", "div[data-rating]"]:
            el = soup.select_one(sel)
            if el:
                rating_text = el.get("data-rating") or el.get_text(strip=True)
                m = re.search(r"(\d+\.?\d*)", rating_text)
                if m:
                    result["RATING"] = m.group(1)
                    break
    if not result.get("RATING"):
        if re.search(r"no\s+ratings?\s+(?:yet|available)", full_text, re.IGNORECASE):
            result["RATING"] = "0"

    # Review count
    if not result.get("REVIEW_COUNT"):
        for sel in ["a[href*='#reviews']", "span[class*='revw']", "[class*='review'] span"]:
            el = soup.select_one(sel)
            if el:
                m = re.search(r"(\d+)", el.get_text())
                if m:
                    result["REVIEW_COUNT"] = m.group(1)
                    break
        if not result.get("REVIEW_COUNT"):
            m = re.search(r"(\d+)\s+verified\s+ratings?", full_text, re.IGNORECASE)
            if m: result["REVIEW_COUNT"] = m.group(1)
    if not result.get("REVIEW_COUNT"):
        if re.search(r"no\s+ratings?\s+(?:yet|available)", full_text, re.IGNORECASE):
            result["REVIEW_COUNT"] = "0"

    # Stock status
    _stock_text = ""
    for sel in ["span[class*='stock']", "div[class*='stock']",
                "span[class*='avail']", "div[class*='avail']", "p[class*='stock']"]:
        el = soup.select_one(sel)
        if el:
            _t = el.get_text(strip=True)
            if _t and len(_t) < 80:
                _stock_text = _t
                break
    if not _stock_text:
        _low_m = re.search(
            r"((?:some\s+)?variations?\s+with\s+low\s+stock"
            r"|low\s+stock|only\s+\d+\s+left"
            r"|hurry[,!]?\s+(?:only\s+)?\d+"
            r"|in\s+stock|out\s+of\s+stock|sold\s+out)",
            full_text, re.IGNORECASE,
        )
        if _low_m: _stock_text = _low_m.group(1).strip()
    if _stock_text:
        result["STOCK_STATUS"] = _stock_text.capitalize()
    else:
        result["STOCK_STATUS"] = result.pop("_ld_stock", "In Stock")
        for sel in ["div.out-of-stock", "[class*='sold-out']",
                    "[class*='out-of-stock']", "button[disabled]"]:
            if soup.select_one(sel):
                result["STOCK_STATUS"] = "Out of Stock"
                break
    result.pop("_ld_stock", None)

    # ── Colors / Sizes / Variations ──────────────────────────────────────────
    colors: list[str] = []
    sizes:  list[str] = []

    _SIZE_RE = re.compile(
        r"""
        ^(
            (EU|UK|US|EU/UK)\s*\d{1,3}(\.\d)?
          | \d{1,3}(\.\d)?
          | (XS|S|M|L|XL|XXL|XXXL|3XL|4XL|5XL)
          | (ONE\s*SIZE|FREE\s*SIZE)
          | \d{1,3}\s*(cm|mm|in|inch|inches)
        )$
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Words found in Jumia page furniture (share buttons, social CTAs, promo
    # text) that must never be treated as variation pill values.
    _PILL_BLOCKLIST: set[str] = {
        "share", "this", "product", "add", "cart", "wishlist", "compare",
        "buy", "now", "view", "more", "details", "description", "reviews",
        "rating", "sold", "by", "seller", "store", "official", "report",
        "incorrect", "information", "see", "all", "promotions", "delivery",
        "returns", "policy", "similar", "products", "from", "brand",
        "print", "email", "copy", "link", "tweet", "pin", "like",
        "follow", "followers", "score", "shipping", "express", "fulfilled",
        "jumia", "verified", "in", "on", "at", "and", "or",
        "the", "a", "an", "with", "for", "of", "to", "is", "are",
        "get", "set", "go", "new", "top", "hot", "sale", "free",
    }

    def _is_size(val: str) -> bool:
        return bool(_SIZE_RE.match(val.strip()))

    def _collect_pills(selector: str) -> list[str]:
        """
        Return text values of non-empty pill/option elements.
        Filters out spec-attribute rows, page-furniture words, and comma-
        separated spec strings whether they appear as single or split tokens.
        """
        vals: list[str] = []
        for el in soup.select(selector):
            inp = el.select_one("input")
            if inp:
                v = inp.get("value") or inp.get("data-value", "")
            else:
                # Read only the first direct child <span> to avoid concatenating
                # sibling spans e.g. "EU 40" + "Selected" → "EU 40Selected"
                first_span = el.find("span", recursive=False)
                if first_span:
                    v = first_span.get_text(strip=True)
                else:
                    v = el.get_text(strip=True)
            v = v.strip()
            if not v or len(v) >= 50:
                continue
            # Filter 1: explicit colon-space separator  e.g. "SKU: XYZ123"
            if ": " in v:
                continue
            # Filter 2: contains a comma — spec strings, never real pills
            if "," in v:
                continue
            # Filter 3: starts with a known spec-attribute label
            if _SPEC_LABEL_RE.match(v):
                continue
            # Filter 4: page-furniture words (share buttons, CTAs, nav links)
            if v.lower() in _PILL_BLOCKLIST:
                continue
            vals.append(v)
        return vals

    # Strategy A
    pill_vals = _collect_pills("ul.-pvs.-mvs li, ul.-pvs li, ul[class*='var'] li")

    # Strategy B
    if not pill_vals:
        pill_vals = _collect_pills(
            "[data-type='color'] span, span.color-selector, "
            "li[class*='color'] span, label[class*='color'], "
            "[data-type='size'] span, li[class*='size'] span"
        )

    # Strategy C: scoped select dropdowns
    if not pill_vals:
        product_form = (
            soup.select_one("form#product_addtocart_form")
            or soup.select_one("form[action*='checkout']")
            or soup.select_one("section.-pvs")
        )
        if product_form:
            scoped_opts: list[str] = []
            for sel_el in product_form.select("select option:not([value='']):not([disabled])"):
                v = sel_el.get_text(strip=True)
                if v and len(v) < 50:
                    scoped_opts.append(v)
            if 0 < len(scoped_opts) <= 30:
                pill_vals = scoped_opts

    # Strategy D: div/button pills
    if not pill_vals:
        pill_vals = _collect_pills(
            "div[class*='var'] button, div[class*='size'] button, "
            "div[class*='color'] button"
        )

    if pill_vals:
        unique_vals = list(dict.fromkeys(v for v in pill_vals if v))
        for v in unique_vals:
            if _is_size(v): sizes.append(v)
            else: colors.append(v)

    def _tokenise_variant_string(raw: str) -> list[str]:
        raw = raw.strip()
        if re.search(r"[,|/]", raw):
            parts = re.split(r"[,|/]", raw)
        elif " " in raw:
            parts = raw.split()
        else:
            parts = re.findall(
                r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+(?:\.[0-9]+)?",
                raw,
            )
        return [p.strip() for p in parts if p.strip() and len(p.strip()) <= 30]

    # Strategy E2: single-token split
    if colors and len(colors) == 1 and not sizes:
        tokens = _tokenise_variant_string(colors[0])
        if len(tokens) >= 2:
            colors = []
            for tok in tokens:
                if _is_size(tok): sizes.append(tok)
                else: colors.append(tok)

    # Strategy F: "Variation available" block
    if not colors and not sizes:
        var_m = re.search(r"Variation\s+available\s*\n([^\n]{2,80})", full_text, re.IGNORECASE)
        if var_m:
            candidate = var_m.group(1).strip()
            tokens = _tokenise_variant_string(candidate)
            for tok in tokens:
                if _is_size(tok): sizes.append(tok)
                else: colors.append(tok)

    # Strategy G: .-pvs section text
    # Only runs when pill detection returned nothing.
    # Aborts if ANY token in the candidate text matches _SPEC_LABEL_RE —
    # that means the section is a spec-attributes display, not a variation
    # display, so there are no real variations to extract.
    if not colors and not sizes:
        pvs = soup.select_one("section.-pvs, div.-pvs, ul.-pvs")
        if pvs:
            candidate_text = pvs.get_text(" ", strip=True)
            candidate_text = re.sub(
                r"(Variation\s+available|Add\s+to\s+cart|Some\s+variations|low\s+stock)",
                "", candidate_text, flags=re.IGNORECASE,
            ).strip()
            if candidate_text and len(candidate_text) <= 120:
                tokens = _tokenise_variant_string(candidate_text)
                # Abort if this looks like a spec section, not a variation section
                if not any(_SPEC_LABEL_RE.match(t) for t in tokens):
                    for tok in tokens:
                        if _is_size(tok): sizes.append(tok)
                        else: colors.append(tok)

    # Strategy E: title colour extraction (only when all above gave nothing)
    if not colors and not sizes:
        title_el = soup.select_one("h1, title")
        if title_el:
            COLOR_WORDS = re.compile(
                r"\b(black|white|silver|gold|blue|red|green|purple|pink|"
                r"grey|gray|titanium|midnight|starlight|ivory|champagne|"
                r"rose|copper|yellow|orange|violet|navy|cream|brown|coral|"
                r"aqua|cyan|teal|lilac|maroon|beige|olive|turquoise)\b",
                re.IGNORECASE,
            )
            found = COLOR_WORDS.findall(title_el.get_text())
            if found:
                colors = list(dict.fromkeys(c.title() for c in found))

    # Write variation results
    if sizes and colors:
        result["SIZES_AVAILABLE"]  = ", ".join(sizes)
        result["COLOR"]            = ", ".join(colors)
        result["COUNT_VARIATIONS"] = str(len(sizes) * len(colors))
    elif sizes:
        result["SIZES_AVAILABLE"]  = ", ".join(sizes)
        result["COUNT_VARIATIONS"] = str(len(sizes))
    elif colors:
        result["COLOR"]            = ", ".join(colors)
        result["COUNT_VARIATIONS"] = str(len(colors))

    # COLOR_IN_TITLE
    _COLOR_WORDS = re.compile(
        r"\b("
        r"titanium\s+silver|titanium\s+gold|midnight\s+black|midnight\s+blue|"
        r"space\s+gray|space\s+grey|rose\s+gold|sky\s+blue|navy\s+blue|"
        r"forest\s+green|hot\s+pink|light\s+blue|dark\s+blue|dark\s+green|"
        r"light\s+green|deep\s+purple|bright\s+red|pearl\s+white|"
        r"starlight\s+white|starlight\s+silver|"
        r"black|white|silver|gold|blue|red|green|purple|pink|"
        r"grey|gray|titanium|midnight|starlight|ivory|champagne|"
        r"rose|copper|yellow|orange|violet|navy|cream|brown|coral|"
        r"aqua|cyan|teal|lilac|maroon|beige|olive|turquoise"
        r")\b",
        re.IGNORECASE,
    )
    h1 = soup.select_one("h1")
    title_text = h1.get_text(" ", strip=True) if h1 else ""
    if not title_text:
        title_tag = soup.find("title")
        title_text = title_tag.get_text(" ", strip=True) if title_tag else ""
    if title_text:
        found_title_colors = _COLOR_WORDS.findall(title_text)
        if found_title_colors:
            result["COLOR_IN_TITLE"] = ", ".join(
                list(dict.fromkeys(c.title() for c in found_title_colors))
            )
        else:
            result["COLOR_IN_TITLE"] = "None"
    else:
        result["COLOR_IN_TITLE"] = "None"

    # Specs table
    spec_map: dict[str, str] = {}
    for section in soup.select("div[class*='spec'], section[class*='spec'], ul[class*='spec']"):
        items = section.find_all("li") or section.find_all("div", recursive=False)
        for item in items:
            txt = item.get_text(" ", strip=True)
            if ":" in txt:
                parts = txt.split(":", 1)
            else:
                parts = re.split(r"\s{2,}", txt, 1)
            if len(parts) == 2:
                k = parts[0].strip().lower().replace(" ", "_")
                v = parts[1].strip()
                if k and v: spec_map[k] = v
    if not spec_map:
        spec_section_m = re.search(
            r"Specifications?\s*\n(.*?)(?:Customer Feedback|Related results|\Z)",
            full_text, re.IGNORECASE | re.DOTALL,
        )
        if spec_section_m:
            for line in spec_section_m.group(1).splitlines():
                line = line.strip()
                if ":" in line:
                    k, _, v = line.partition(":")
                    k = k.strip().lower().replace(" ", "_")
                    v = v.strip()
                    if k and v: spec_map[k] = v

    # MODEL
    if not result.get("MODEL"):
        model_val = ld_data.get("model", "") if ld_data else ""
        if model_val: result["MODEL"] = str(model_val).strip()
    if not result.get("MODEL"):
        for key in ("model", "model_name", "model_number"):
            if spec_map.get(key):
                result["MODEL"] = spec_map[key]
                break
    if not result.get("MODEL"):
        m_mod = re.search(r"\bModel\s*[:\-]\s*([^\n,]+)", full_text, re.IGNORECASE)
        if m_mod: result["MODEL"] = m_mod.group(1).strip()

    # GTIN
    if not result.get("GTIN"):
        for gtin_key in ("gtin13", "gtin12", "gtin8", "gtin", "barcode"):
            val = ld_data.get(gtin_key, "") if ld_data else ""
            if val:
                result["GTIN"] = str(val).strip()
                break
    if not result.get("GTIN"):
        for key in ("gtin_barcode", "gtin", "barcode", "ean", "upc"):
            if spec_map.get(key):
                result["GTIN"] = spec_map[key]
                break
    if not result.get("GTIN"):
        m_gtin = re.search(r"GTIN\s*(?:Barcode)?\s*[:\-]\s*(\d{8,14})", full_text, re.IGNORECASE)
        if m_gtin: result["GTIN"] = m_gtin.group(1).strip()

    # WEIGHT
    if not result.get("WEIGHT"):
        for key in ("weight_(kg)", "weight_kg", "weight_(g)", "weight"):
            if spec_map.get(key):
                result["WEIGHT"] = spec_map[key]
                break
    if not result.get("WEIGHT"):
        wt = ld_data.get("weight", {}) if ld_data else {}
        if isinstance(wt, dict) and wt.get("value"):
            result["WEIGHT"] = f"{wt['value']} {wt.get('unitCode', 'kg')}"
        elif isinstance(wt, str) and wt:
            result["WEIGHT"] = wt
    if not result.get("WEIGHT"):
        m_wt = re.search(r"Weight\s*(?:\(kg\))?\s*[:\-]\s*([\d\.]+\s*(?:kg|g)?)", full_text, re.IGNORECASE)
        if m_wt: result["WEIGHT"] = m_wt.group(1).strip()

    # CATEGORY_PATH
    if not result.get("CATEGORY_PATH"):
        crumbs: list[str] = []
        for sel in ["ol.breadcrumb a", "nav[aria-label*='breadcrumb'] a",
                    "div.-bre a", "a[class*='crumb']", "div[class*='breadcrumb'] a"]:
            links = soup.select(sel)
            if links:
                crumbs = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                break
        if not crumbs:
            for nav in soup.find_all(["nav", "ol", "ul"]):
                links = nav.find_all("a")
                texts = [a.get_text(strip=True) for a in links if a.get_text(strip=True)]
                if 3 <= len(texts) <= 7 and len(set(texts)) == len(texts) and all(len(t) < 50 for t in texts):
                    crumbs = texts
                    break
        if crumbs:
            if crumbs[0].lower() in ("home", "jumia"): crumbs = crumbs[1:]
            if crumbs and len(crumbs[-1]) > 60: crumbs = crumbs[:-1]
            if crumbs: result["CATEGORY_PATH"] = " > ".join(crumbs)

    # IS_OFFICIAL_STORE
    if not result.get("IS_OFFICIAL_STORE"):
        _official_re = re.compile(r"\bofficial\s+store\b", re.IGNORECASE)
        body_copy = soup.find("body")
        for nav_el in (body_copy.find_all(["header", "nav"]) if body_copy else []):
            nav_el.decompose()
        jmall_badge   = soup.select_one("a[href*='JMALL'], a[tag*='JMALL']")
        official_text = body_copy.find(string=_official_re) if body_copy else None
        official_img  = soup.find("img", alt=_official_re)
        result["IS_OFFICIAL_STORE"] = "Yes" if any([jmall_badge, official_text, official_img]) else "No"

    # SELLER_NAME
    # Jumia renders the seller name in a dedicated seller-info section.
    # We use a strict allowlist approach: only trust elements clearly
    # scoped to seller identity, never general nav/promo links.
    if not result.get("SELLER_NAME"):
        _seller_name = ""

        # Strategy 1: dedicated seller-info / seller-details block
        # Jumia renders a seller card below the add-to-cart button.
        # Try explicit class selectors first, then look for a "Seller
        # Information" heading and grab the first link beneath it.
        for _sel in [
            "div.seller-info a[href]",
            "div[class*='seller-info'] a[href]",
            "section[class*='seller'] a[href]",
            "div[class*='seller-detail'] a[href]",
            "a[class*='seller-name']",
            "a[class*='store-name']",
        ]:
            _el = soup.select_one(_sel)
            if _el:
                _txt = _el.get_text(strip=True)
                if _txt and 2 <= len(_txt) <= 60:
                    _lower = _txt.lower()
                    if not any(w in _lower for w in (
                        "seller", "store", "view", "follow", "score",
                        "performance", "information", "rating", "jumia",
                    )):
                        _seller_name = _txt
                        break

        # Try the "Seller Information" section link as part of strategy 1
        if not _seller_name:
            for _h in soup.find_all(["h2", "h3", "h4", "b", "strong", "div", "section"]):
                _ht = _h.get_text(strip=True)
                if not re.search(r"seller\s+information", _ht, re.IGNORECASE):
                    continue
                # Found the seller section — grab first <a> link inside it
                for _a in _h.find_all_next("a", href=True, limit=5):
                    _txt = _a.get_text(strip=True)
                    if _txt and 2 <= len(_txt) <= 60:
                        _lower = _txt.lower()
                        if not any(w in _lower for w in (
                            "follow", "score", "performance", "rating",
                            "jumia", "sell on", "contact",
                        )):
                            _seller_name = _txt
                            break
                if _seller_name:
                    break

        # Strategy 2: seller slug link — ONLY search within a scoped
        # container near the seller section, not all <a> tags in the doc.
        # Jumia seller URLs end in -cod/ (third-party) or are a plain slug.
        # We require the link to appear after the price/add-to-cart area.
        if not _seller_name:
            # Comprehensive set of Jumia category/nav slug words (singular + plural)
            _nav_words = {
                "home", "phone", "phones", "tablet", "tablets",
                "fashion", "compute", "computing", "gaming", "games",
                "supermarket", "groceries", "grocery",
                "health", "beauty", "appliance", "appliances",
                "office", "baby", "jumia", "sell", "selling",
                "help", "warranty", "warranties", "privacy",
                "terms", "conditions", "credit", "store", "official",
                "catalog", "cart", "wishlist", "account", "orders",
                "order", "return", "returns", "refund", "refunds",
                "delivery", "payment", "payments", "sp", "mlp",
                "category", "categories", "electronics", "electronic",
                "television", "televisions", "video", "audio",
                "refrigerator", "refrigerators", "washing",
                "smartphone", "smartphones", "laptop", "laptops",
                "accessories", "accessory", "clothing", "shoes",
                "sports", "sport", "outdoor", "garden", "toys",
                "books", "music", "movies", "automotive", "tools",
                "industrial", "furniture", "kitchen", "dining",
                "bedroom", "bathroom", "lighting", "security",
                "networking", "software", "printers", "cameras",
                "smart", "wearables", "wearable", "watches",
            }
            _seller_slug_re = re.compile(
                r"^/([a-z0-9][a-z0-9\-]{3,40})(?:-cod)?/$"
            )
            # Only scan <a> tags that appear after the price section
            # by finding the price element and walking siblings/children
            _price_anchor = (
                soup.select_one("span.-b.-ltr.-tal.-fs24")
                or soup.select_one("span.prc")
                or soup.select_one("div.-prc-w")
            )
            _search_root = _price_anchor.parent if _price_anchor else soup
            for _a in _search_root.find_all_next("a", href=True, limit=60):
                _href = str(_a.get("href", ""))
                _m = _seller_slug_re.match(_href)
                if not _m:
                    continue
                _slug_parts = set(_m.group(1).replace("-cod", "").split("-"))
                # Skip if any slug segment is a nav/category word
                if _slug_parts & _nav_words:
                    continue
                _txt = _a.get_text(strip=True)
                if not _txt or len(_txt) > 60:
                    continue
                _lower = _txt.lower()
                if any(w in _lower for w in (
                    "terms", "conditions", "credit", "policy", "warranty",
                    "help", "support", "service", "sell on", "official store",
                    "jumia express", "fulfilled", "promotions", "delivery",
                    "returns", "similar", "from", "brand", "follow",
                )):
                    continue
                _seller_name = _txt
                break

        # Strategy 3: JSON-LD — only use the seller object, never brand
        # (brand is the product brand, not the marketplace seller)
        if not _seller_name and ld_data:
            _seller_obj = ld_data.get("seller")
            if isinstance(_seller_obj, dict):
                _s = str(_seller_obj.get("name", "")).strip()
                if _s and len(_s) <= 60:
                    _seller_name = _s
            elif isinstance(_seller_obj, str) and len(_seller_obj.strip()) <= 60:
                _seller_name = _seller_obj.strip()

        # Strategy 4: "Seller Information" heading → next sibling text
        # Safer than regex: scope to the actual seller-info section
        if not _seller_name:
            for _heading in soup.find_all(["h2", "h3", "h4", "b", "strong"]):
                if re.search(r"seller\s+information", _heading.get_text(), re.IGNORECASE):
                    _sib = _heading.find_next_sibling()
                    while _sib:
                        _txt = _sib.get_text(strip=True)
                        if _txt and 2 <= len(_txt) <= 60:
                            _seller_name = _txt
                            break
                        _sib = _sib.find_next_sibling()
                    break

        if _seller_name:
            result["SELLER_NAME"] = _seller_name

    # Warranty
    warranty_pattern = re.compile(
        r"(\d+[\s\-]?(?:year|month|yr|mo)[s]?\s+(?:warranty|guarantee))", re.IGNORECASE
    )
    m = warranty_pattern.search(full_text)
    if m:
        result["PRODUCT_WARRANTY"]  = "Yes"
        result["WARRANTY_DURATION"] = m.group(1).strip()
    else:
        for row in soup.select("ul li, table tr, div.-fs14"):
            txt = row.get_text(" ", strip=True)
            if re.search(r"warranty", txt, re.IGNORECASE):
                val = re.sub(r"[Ww]arranty\s*[:\-]?\s*", "", txt).strip()
                if val and len(val) < 80:
                    result["PRODUCT_WARRANTY"] = "Yes"
                    result.setdefault("WARRANTY_DURATION", val)
                    break
        m2 = re.search(r"Warranty\s+Address\s*[:\-]?\s*(.+?)(?:\n|$)", full_text, re.IGNORECASE)
        if m2:
            result["PRODUCT_WARRANTY"] = "Yes"
            result.setdefault("WARRANTY_DURATION", m2.group(1).strip())

    # SPECIFICATIONS
    if not result.get("SPECIFICATIONS"):
        spec_lines: list[str] = []
        for section in soup.select("div[class*='spec'], section[class*='spec'], "
                                   "ul[class*='spec'], div.-atr, section.-atr"):
            for item in (section.find_all("li") or section.find_all("div", recursive=False)):
                txt = item.get_text(" ", strip=True)
                if ":" in txt and len(txt) < 200:
                    spec_lines.append(txt.strip())
        if not spec_lines:
            m_spec = re.search(
                r"Specifications?\s*\n(.*?)(?:Customer Feedback|Related results|\Z)",
                full_text, re.IGNORECASE | re.DOTALL,
            )
            if m_spec:
                for line in m_spec.group(1).splitlines():
                    line = line.strip()
                    if line and len(line) < 200: spec_lines.append(line)
        known = [
            ("SKU",             result.get("MODEL") or ""),
            ("Weight (kg)",     result.get("WEIGHT") or ""),
            ("Main Material",   spec_map.get("main_material", "")),
            ("Size (L x W H)", spec_map.get("size_(l_x_w_x_h_cm)", "")),
        ]
        for label, val in known:
            if val and not any(label.lower() in l.lower() for l in spec_lines):
                spec_lines.append(f"{label}: {val}")
        if spec_lines:
            result["SPECIFICATIONS"] = " | ".join(spec_lines)

    # KEY_SPECS
    if not result.get("KEY_SPECS"):
        ks_parts: list[str] = []
        for label, key in [("SKU","sku"),("Size","size_(l_x_w_x_h_cm)"),
                            ("Weight","weight_(kg)"),("Material","main_material"),
                            ("GTIN","gtin_barcode")]:
            val = spec_map.get(key, "")
            if val: ks_parts.append(f"{label}: {val}")
        if not any("SKU" in p for p in ks_parts) and result.get("MODEL"):
            ks_parts.insert(0, f"SKU: {result['MODEL']}")
        if not any("Weight" in p for p in ks_parts) and result.get("WEIGHT"):
            ks_parts.append(f"Weight: {result['WEIGHT']} kg")
        if ks_parts:
            result["KEY_SPECS"] = " | ".join(ks_parts)

    # DESCRIPTION
    if not result.get("DESCRIPTION"):
        STOP_HEADS = re.compile(
            r"^(specifications?|customer\s+feedback|related\s+results|seller\s+info|"
            r"promotions?|delivery|return\s+policy|seller\s+information)",
            re.IGNORECASE,
        )

        def _resolve_imgs(container) -> None:
            for img in container.find_all("img"):
                real_src = (img.get("data-src") or img.get("data-original")
                            or img.get("data-lazy") or img.get("src", ""))
                if not real_src or real_src.startswith("data:"):
                    img.decompose()
                    continue
                if real_src.startswith("//"): real_src = "https:" + real_src
                elif real_src.startswith("/"): real_src = base_url + real_src
                img["src"] = real_src
                for attr in ("data-src","data-original","data-lazy","data-srcset","srcset","loading"):
                    if img.has_attr(attr): del img[attr]
                if not img.get("alt"): img["alt"] = ""

        def _html(tag) -> str:
            return str(tag)

        desc_html_parts: list[str] = []
        collecting = False
        for tag in soup.find_all(["h2","h3","h4","p","ul","ol","div","img","figure"], limit=500):
            txt = tag.get_text(" ", strip=True)
            if tag.name in ("h2","h3","h4") and re.search(r"product\s+details?", txt, re.IGNORECASE):
                collecting = True
                continue
            if not collecting: continue
            if tag.name in ("h2","h3","h4") and STOP_HEADS.search(txt): break
            if tag.name == "div":
                cls = " ".join(tag.get("class", []))
                if any(s in cls.lower() for s in ("nav","bread","cart","seller","rate","review","footer","header","sidebar","related","promo","banner")):
                    continue
                if not bool(tag.find("img")) and len(txt) <= 30: continue
            _resolve_imgs(tag)
            serialised = _html(tag).strip()
            if serialised and serialised not in desc_html_parts:
                desc_html_parts.append(serialised)

        if not desc_html_parts:
            for sel in ["div.-prd-desc","div[class*='description']","section[class*='description']","div[class*='product-desc']","div.-fs14"]:
                container = soup.select_one(sel)
                if container:
                    _resolve_imgs(container)
                    desc_html_parts = [_html(container)]
                    break

        if not desc_html_parts:
            for img in soup.find_all("img"):
                src = (img.get("data-src") or img.get("data-original") or img.get("data-lazy") or img.get("src",""))
                if src and not src.startswith("data:") and ("product" in src or "unsafe" in src or "jumia.is" in src):
                    if src.startswith("//"): src = "https:" + src
                    elif src.startswith("/"): src = base_url + src
                    desc_html_parts.append(f'<img src="{src}" alt="" style="max-width:100%;height:auto;display:block;margin:8px 0;">')

        if not desc_html_parts and result.get("DESCRIPTION"):
            plain = result["DESCRIPTION"]
            desc_html_parts = [f"<p>{para.strip()}</p>" for para in plain.split("\n\n") if para.strip()]

        if desc_html_parts:
            seen_keys: set[str] = set()
            unique_parts: list[str] = []
            for part in desc_html_parts:
                key = re.sub(r"\s+", " ", part)[:120]
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_parts.append(part)
            text_parts = [p for p in unique_parts if not re.match(r"^<img", p.strip())]
            if not text_parts:
                img_urls = []
                for part in unique_parts:
                    m_src = re.search(r'src="([^"]+)"', part)
                    if m_src:
                        url = m_src.group(1)
                        if url not in img_urls: img_urls.append(url)
                result["DESCRIPTION"] = "\n".join(img_urls) if img_urls else ""
            else:
                result["DESCRIPTION"] = (
                    '<div class="jm-desc" style="font-family:sans-serif;line-height:1.6;color:#313133;">\n'
                    + "\n".join(text_parts) + "\n</div>"
                )

    # KEY_FEATURES
    key_features: list[str] = []
    kf_pattern = re.compile(r"key\s+features?", re.IGNORECASE)
    for heading in soup.find_all(["h2","h3","h4","b","strong"]):
        if not kf_pattern.search(heading.get_text()): continue
        node = heading.find_next_sibling()
        while node:
            if node.name in ("h2","h3","h4"): break
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt: key_features.append(txt)
                break
            if node.name == "div":
                inner_ul = node.find("ul")
                if inner_ul:
                    for li in inner_ul.find_all("li", recursive=False):
                        txt = li.get_text(" ", strip=True)
                        if txt: key_features.append(txt)
                    if key_features: break
            node = node.find_next_sibling()
        if key_features: break
    if not key_features:
        for ul in soup.find_all("ul"):
            prev_h = ul.find_previous(["h2","h3","h4","b","strong"])
            if prev_h and kf_pattern.search(prev_h.get_text()):
                for li in ul.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt: key_features.append(txt)
                break
    if key_features:
        result["KEY_FEATURES"] = " | ".join(key_features)

    # WHATS_IN_BOX
    in_box: list[str] = []
    box_pattern = re.compile(r"what.{0,5}s?\s+in\s+the\s+box", re.IGNORECASE)
    for heading in soup.find_all(["h2","h3","h4","b","strong"]):
        if not box_pattern.search(heading.get_text()): continue
        node = heading.find_next_sibling()
        while node:
            if node.name in ("h2","h3","h4"): break
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    txt = li.get_text(" ", strip=True)
                    if txt: in_box.append(txt)
                break
            if node.name == "div":
                inner_ul = node.find("ul")
                if inner_ul:
                    for li in inner_ul.find_all("li", recursive=False):
                        txt = li.get_text(" ", strip=True)
                        if txt: in_box.append(txt)
                    if in_box: break
            node = node.find_next_sibling()
        if in_box: break
    if not in_box:
        for ul in soup.find_all("ul"):
            prev_h = ul.find_previous(["h2","h3","h4","b","strong"])
            if prev_h and box_pattern.search(prev_h.get_text()):
                for li in ul.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt: in_box.append(txt)
                break
    if not in_box:
        m_box = re.search(
            r"what.{0,5}s?\s+in\s+the\s+box[:\s]+(.*?)(?=\n\s*\n|##|\bSpecifications?\b|\bCustomer\b|\bRelated\b|\Z)",
            full_text, re.IGNORECASE | re.DOTALL,
        )
        if m_box:
            raw_box = m_box.group(1).strip()
            if len(raw_box) < 300:
                items = re.split(r"[•\n,;]+", raw_box)
                in_box = [i.strip() for i in items if i.strip() and len(i.strip()) > 2]
    if in_box:
        result["WHATS_IN_BOX"] = " | ".join(in_box)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def scrape_single_sku(sku: str, country_code: str = "KE") -> dict[str, str]:
    base_url = COUNTRY_URLS.get(country_code, COUNTRY_URLS["KE"])
    url = _search_product_url(sku, base_url)
    if not url:
        return {}
    return _scrape_product_page(url, base_url)


def enrich_post_qc_df(
    df: pd.DataFrame,
    country_code: str = "KE",
    progress_callback=None,
    delay: float = 0.6,
) -> pd.DataFrame:
    base_url = COUNTRY_URLS.get(country_code, COUNTRY_URLS["KE"])
    cols_needed = [c for c in SCRAPABLE_FIELDS if _needs_fill(df, c)]
    if not cols_needed:
        logger.info("All scrapable columns already populated — skipping enrichment.")
        return df

    sku_col = _find_sku_column(df)
    if sku_col is None:
        logger.warning("No SKU column found in dataframe — cannot enrich.")
        return df

    df = df.copy()
    for col in cols_needed:
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    logger.info("Enriching %d rows | country=%s | fields=%s", total, country_code, cols_needed)

    for seq, (row_idx, row) in enumerate(df.iterrows()):
        sku = str(row.get(sku_col, "")).strip()
        if _row_is_empty(sku):
            if progress_callback: progress_callback(seq + 1, total, "—")
            continue
        row_needs = any(_row_is_empty(row.get(col, "")) for col in cols_needed)
        if not row_needs:
            if progress_callback: progress_callback(seq + 1, total, sku)
            continue
        try:
            product_url = _search_product_url(sku, base_url)
            if product_url:
                scraped = _scrape_product_page(product_url, base_url)
                for col, val in scraped.items():
                    if col in cols_needed and _row_is_empty(df.at[row_idx, col]):
                        df.at[row_idx, col] = val
            time.sleep(delay)
        except Exception as exc:
            logger.warning("Enrichment failed for SKU %s: %s", sku, exc)
        if progress_callback:
            progress_callback(seq + 1, total, sku)

    return df
