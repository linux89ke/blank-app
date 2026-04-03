"""
postqc.py
─────────
Post-QC validation for products already live on Jumia.

Data sources:
  1. 🔍 Keyword search  → scrape Jumia search results for a term
  2. 🔗 Category URL    → scrape a category listing page-by-page
  3. 📁 Upload file     → load a previously-scraped CSV / XLSX

Report format (matches Jumia QC report standard):
  SKU | Name | Brand | Category | Price | Old Price | Discount | Rating |
  Total Ratings | Seller | Jumia Express | Shop Global | Image URL |
  Product URL | Stock | Tags | Country | Quality Score |
  Critical Issues | High Issues | Medium Issues | Low Issues | Total Issues |
  Top Issues | No. of Images | Few Images (<5) | Non-White Background |
  Low Resolution Images | Empty Description | Thin Description |
  Muddled Description | Repeated Description | Missing Images in Desc |
  Naming Issue | Wrong Category | Prohibited Item | Blacklisted Keyword |
  Restricted Brand | NG Seller Restriction | Counterfeit Flag | Suspicious Price

Scoring:
  Quality Score = 100 − (Critical×25) − (High×12) − (Medium×6) − (Low×2)

Issue severity mapping:
  Critical : Prohibited Item, Counterfeit Flag
  High     : Wrong Category, Blacklisted Keyword, Restricted Brand,
             NG Seller Restriction, Suspicious Price
  Medium   : Few Images (<5), Missing Images in Desc, Naming Issue,
             Non-White Background, Low Resolution Images,
             Empty/Thin/Muddled Description
  Low      : Repeated Description
"""

from __future__ import annotations

import logging
import re
import time
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
POST_QC_SIGNATURE_COLS = {'sku', 'name', 'brand', 'category', 'price', 'seller'}

COUNTRY_BASE_URLS: Dict[str, str] = {
    "KE": "https://www.jumia.co.ke",
    "UG": "https://www.jumia.ug",
    "NG": "https://www.jumia.com.ng",
    "GH": "https://www.jumia.com.gh",
    "MA": "https://www.jumia.ma",
}
COUNTRY_CODE_TO_NAME: Dict[str, str] = {
    "KE": "Kenya", "UG": "Uganda", "NG": "Nigeria", "GH": "Ghana", "MA": "Morocco",
}
COUNTRY_NAME_TO_CODE: Dict[str, str] = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}

JUMIA_COLORS = {
    'primary_orange':   '#F68B1E',
    'secondary_orange': '#FF9933',
    'jumia_red':        '#E73C17',
    'dark_gray':        '#313133',
    'medium_gray':      '#5A5A5C',
    'light_gray':       '#F5F5F5',
    'success_green':    '#4CAF50',
    'warning_yellow':   '#FFC107',
}

# Issue severity weights for quality score
_W = {'critical': 25, 'high': 12, 'medium': 6, 'low': 2}

# Final report column order (matches sample file exactly)
REPORT_COLS = [
    'SKU', 'Name', 'Brand', 'Category', 'Price', 'Old Price', 'Discount',
    'Rating', 'Total Ratings', 'Seller', 'Jumia Express', 'Shop Global',
    'Image URL', 'Product URL', 'Stock', 'Tags', 'Country',
    'Quality Score', 'Critical Issues', 'High Issues', 'Medium Issues',
    'Low Issues', 'Total Issues', 'Top Issues',
    'No. of Images', 'Few Images (<5)', 'Non-White Background',
    'Low Resolution Images', 'Empty Description', 'Thin Description',
    'Muddled Description', 'Repeated Description', 'Missing Images in Desc',
    'Naming Issue', 'Wrong Category', 'Prohibited Item', 'Blacklisted Keyword',
    'Restricted Brand', 'NG Seller Restriction', 'Counterfeit Flag',
    'Suspicious Price',
]

_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_SESSION = requests.Session()
_SESSION.headers.update(_SCRAPE_HEADERS)

# ─────────────────────────────────────────────────────────────────────────────
# FILE DETECTION & CATEGORY MAP
# ─────────────────────────────────────────────────────────────────────────────

def detect_file_type(df: pd.DataFrame) -> str:
    cols_lower = set(df.columns.str.strip().str.lower())
    if POST_QC_SIGNATURE_COLS.issubset(cols_lower):
        return 'post_qc'
    return 'pre_qc'


def load_category_map(filename: str = "category_map.xlsx") -> Dict[str, str]:
    import os
    if not os.path.exists(filename):
        csv_path = filename.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            filename = csv_path
        else:
            return {}
    try:
        df = (pd.read_csv(filename, dtype=str) if filename.endswith('.csv')
              else pd.read_excel(filename, engine='openpyxl', dtype=str))
        df.columns = df.columns.str.strip()
        name_col = next((c for c in df.columns if 'name' in c.lower()), None)
        code_col = next((c for c in df.columns if 'code' in c.lower()), None)
        path_col = next((c for c in df.columns if 'path' in c.lower()), None)
        if not name_col or not code_col:
            return {}
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            code = str(row[code_col]).strip().split('.')[0]
            if name and code and name.lower() != 'nan' and code.lower() != 'nan':
                mapping[name.lower()] = code
            if path_col:
                path = str(row.get(path_col, '')).strip()
                if path and path.lower() != 'nan':
                    last = path.split('/')[-1].strip().lower()
                    if last and last not in mapping:
                        mapping[last] = code
        return mapping
    except Exception as e:
        logger.warning(f"load_category_map: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPER — shared card parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_listing_page(html: str, base_url: str) -> List[Dict]:
    """Parse ~40 product cards from one Jumia listing / search results page."""
    soup     = BeautifulSoup(html, "html.parser")
    products: List[Dict] = []

    for article in soup.select("article.prd"):
        try:
            a_tag    = article.select_one("a.core")
            href     = a_tag.get("href", "") if a_tag else ""
            prod_url = (base_url + href) if href.startswith("/") else href

            name_el  = article.select_one("h3.name, .name")
            name     = name_el.get_text(strip=True) if name_el else ""

            brand_el = article.select_one(".brand")
            brand    = brand_el.get_text(strip=True) if brand_el else ""

            price_el  = article.select_one(".prc")
            price_raw = price_el.get_text(strip=True) if price_el else ""
            price     = re.sub(r'[^\d.]', '', price_raw.replace(',', ''))

            old_el    = article.select_one(".old")
            old_price = re.sub(r'[^\d.]', '', old_el.get_text(strip=True).replace(',', '')) if old_el else ""

            disc_el  = article.select_one(".-dsc, .bdg._dsct")
            discount = disc_el.get_text(strip=True) if disc_el else ""

            # Rating decoded from CSS width% on star bar (100% = 5.0 stars)
            rating = ""
            rating_el = article.select_one(".stars._s, .-rtg")
            if rating_el:
                style = rating_el.get("style", "")
                m = re.search(r'width:\s*([\d.]+)%', style)
                if m:
                    rating = str(round(float(m.group(1)) / 20, 1))
                else:
                    rating = rating_el.get_text(strip=True)

            rev_el  = article.select_one(".rev")
            reviews = re.sub(r'[^\d]', '', rev_el.get_text(strip=True)) if rev_el else ""

            img_el = article.select_one("img")
            image  = ""
            if img_el:
                image = (img_el.get("data-src") or img_el.get("data-original")
                         or img_el.get("src") or "")
                if image.startswith("//"): image = "https:" + image
                elif image.startswith("/"): image = base_url + image

            # SKU from data attribute or URL pattern
            sku = article.get("data-id", "") or article.get("data-sku", "")
            if not sku and prod_url:
                m = re.search(r'-(\d{6,12})\.html', prod_url)
                if m: sku = m.group(1)
            if not sku:
                sku = f"UNK-{len(products)}"

            seller = article.get("data-seller", "")

            # Express / Global badges
            express = "Yes" if article.select_one(".-jump, .xpr, [class*='express']") else "No"
            global_ = "Yes" if article.select_one(".-glb, [class*='global']") else "No"

            tags_el = article.select_one(".tags, [class*='tag']")
            tags    = tags_el.get_text(" ", strip=True) if tags_el else ""

            stock_el = article.select_one(".-avl, [class*='stock']")
            stock    = stock_el.get_text(strip=True) if stock_el else "In Stock"

            products.append({
                "SKU":           sku,
                "Name":          name,
                "Brand":         brand,
                "Category":      "",
                "Price":         price,
                "Old Price":     old_price,
                "Discount":      discount,
                "Rating":        rating,
                "Total Ratings": reviews,
                "Seller":        seller,
                "Jumia Express": express,
                "Shop Global":   global_,
                "Image URL":     image,
                "Product URL":   prod_url,
                "Stock":         stock,
                "Tags":          tags,
                "Country":       "",
                # Internal fields for validation
                "PRODUCT_SET_SID": sku,
                "NAME":            name,
                "BRAND":           brand,
                "CATEGORY":        "",
                "CATEGORY_CODE":   "",
                "SELLER_NAME":     seller,
                "GLOBAL_PRICE":    price,
                "OLD_PRICE":       old_price,
                "GLOBAL_SALE_PRICE": price,
                "MAIN_IMAGE":      image,
                "PRODUCT_URL":     prod_url,
                "STOCK_STATUS":    "online",
                "COUNT_VARIATIONS": "1",
                "COLOR":           "",
                "COLOR_FAMILY":    "",
                "PARENTSKU":       sku,
            })
        except Exception as exc:
            logger.debug(f"_parse_listing_page card error: {exc}")

    return products


def _get_total_pages(html: str) -> int:
    soup  = BeautifulSoup(html, "html.parser")
    pages = []
    for el in soup.select("[data-page]"):
        try:
            pages.append(int(el["data-page"]))
        except (ValueError, KeyError):
            pass
    for a in soup.select("a[href*='page=']"):
        m = re.search(r'page=(\d+)', a.get("href", ""))
        if m:
            pages.append(int(m.group(1)))
    return max(pages) if pages else 1


def _extract_breadcrumb(html: str) -> str:
    soup   = BeautifulSoup(html, "html.parser")
    crumbs = soup.select("nav.breadcrumb span, ol.breadcrumb li, .-fdr span")
    if crumbs:
        texts = [c.get_text(strip=True) for c in crumbs if c.get_text(strip=True)]
        for t in reversed(texts):
            if t.lower() not in ('home', 'accueil', ''): return t
    return ""


def _build_page_url(base: str, page: int) -> str:
    base = re.sub(r'[?&]page=\d+', '', base).rstrip('/').rstrip('#')
    base = re.sub(r'#.*$', '', base).rstrip('/')
    sep  = '&' if '?' in base else '?'
    return base if page == 1 else f"{base}{sep}page={page}#catalog-listing"


def _scrape_pages(
    start_url: str,
    country_code: str,
    label: str,
    max_pages: int,
    progress_callback=None,
) -> Tuple[pd.DataFrame, int]:
    """Generic multi-page scraper used by both keyword and category scrapers."""
    base_url     = COUNTRY_BASE_URLS.get(country_code, "https://www.jumia.co.ke")
    all_products: List[Dict] = []
    category_name = label

    # Page 1
    url_p1 = _build_page_url(start_url, 1)
    try:
        resp = _SESSION.get(url_p1, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        logger.error(f"Scrape page 1 failed: {exc}")
        return pd.DataFrame(), 0

    total_pages   = min(_get_total_pages(resp.text), max_pages)
    if not category_name:
        category_name = _extract_breadcrumb(resp.text)
    products_p1   = _parse_listing_page(resp.text, base_url)
    for p in products_p1:
        p["Category"] = category_name
        p["CATEGORY"] = category_name
        p["Country"]  = country_code
    all_products.extend(products_p1)

    if progress_callback:
        progress_callback(1, total_pages, len(all_products))

    for page_num in range(2, total_pages + 1):
        url_pn = _build_page_url(start_url, page_num)
        try:
            resp = _SESSION.get(url_pn, timeout=20)
            resp.raise_for_status()
            prods = _parse_listing_page(resp.text, base_url)
            for p in prods:
                p["Category"] = category_name
                p["CATEGORY"] = category_name
                p["Country"]  = country_code
            all_products.extend(prods)
            time.sleep(0.5)
        except Exception as exc:
            logger.warning(f"Scrape page {page_num} failed: {exc}")
        if progress_callback:
            progress_callback(page_num, total_pages, len(all_products))

    if not all_products:
        return pd.DataFrame(), 0

    df = pd.DataFrame(all_products)
    df['ACTIVE_STATUS_COUNTRY'] = country_code
    df['_IS_MULTI_COUNTRY']     = False
    return df, total_pages


def scrape_by_keyword(
    keyword: str,
    country_code: str,
    max_pages: int = 3,
    progress_callback=None,
) -> Tuple[pd.DataFrame, int]:
    """Scrape Jumia search results for a keyword."""
    base_url  = COUNTRY_BASE_URLS.get(country_code, "https://www.jumia.co.ke")
    search_url = f"{base_url}/catalog/?q={requests.utils.quote(keyword)}"
    return _scrape_pages(search_url, country_code, f'Search: "{keyword}"', max_pages, progress_callback)


def scrape_by_category_url(
    category_url: str,
    country_code: str,
    max_pages: int = 3,
    progress_callback=None,
) -> Tuple[pd.DataFrame, int]:
    """Scrape a Jumia category listing URL."""
    return _scrape_pages(category_url, country_code, "", max_pages, progress_callback)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISE UPLOADED FILES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_post_qc(df: pd.DataFrame, category_map: Dict[str, str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    col_map = {
        'SKU': 'PRODUCT_SET_SID', 'Name': 'NAME', 'Brand': 'BRAND',
        'Category': 'CATEGORY', 'Price': 'GLOBAL_PRICE', 'Old Price': 'OLD_PRICE',
        'Seller': 'SELLER_NAME', 'Image URL': 'MAIN_IMAGE', 'Product URL': 'PRODUCT_URL',
        'Rating': 'RATING', 'Total Ratings': 'TOTAL_RATINGS', 'Discount': 'DISCOUNT',
        'Stock': 'STOCK', 'Stock Status': 'STOCK_STATUS',
    }
    df = df.rename(columns=col_map)
    if 'CATEGORY' in df.columns:
        cmap = category_map or {}
        def resolve_code(raw: str) -> str:
            if not raw or raw == 'nan': return ''
            segs = [s.strip() for s in re.split(r'[>/]', raw) if s.strip()]
            for seg in reversed(segs):
                code = cmap.get(seg.lower())
                if code: return code
            last = segs[-1] if segs else raw
            return re.sub(r'[^a-z0-9]', '_', last.lower())
        df['CATEGORY_CODE'] = df['CATEGORY'].astype(str).apply(resolve_code)
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns: df['ACTIVE_STATUS_COUNTRY'] = 'UNKNOWN'
    df['_IS_MULTI_COUNTRY']  = False
    df['PARENTSKU']          = df.get('PRODUCT_SET_SID', pd.Series(dtype=str))
    df['COLOR']              = df['COLOR'] if 'COLOR' in df.columns else ''
    df['COLOR_FAMILY']       = ''
    df['GLOBAL_SALE_PRICE']  = df.get('GLOBAL_PRICE', '')
    if 'COUNT_VARIATIONS' not in df.columns: df['COUNT_VARIATIONS'] = '1'
    return df


def _resolve_cat_codes(df: pd.DataFrame, support_files: Dict) -> pd.DataFrame:
    """Fill CATEGORY_CODE from category_map / cat_path_to_code."""
    cat_map          = support_files.get('category_map', {})
    cat_path_to_code = support_files.get('cat_path_to_code', {})

    def _resolve(cat_name: str) -> str:
        if not cat_name: return ""
        segs = [s.strip() for s in re.split(r'[>/]', cat_name) if s.strip()]
        for seg in reversed(segs):
            code = cat_map.get(seg.lower()) or cat_path_to_code.get(seg.lower())
            if code: return str(code).split('.')[0]
        return re.sub(r'[^a-z0-9]', '_', segs[-1].lower()) if segs else ""

    df = df.copy()
    df['CATEGORY_CODE'] = df['CATEGORY'].astype(str).apply(_resolve)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_SYMBOLS = [
    'check_restricted_brands', 'check_suspected_fake_products',
    'check_refurb_seller_approval', 'check_product_warranty',
    'check_seller_approved_for_books', 'check_seller_approved_for_perfume',
    'check_perfume_tester', 'check_counterfeit_sneakers',
    'check_counterfeit_jerseys', 'check_prohibited_products',
    'check_unnecessary_words', 'check_single_word_name',
    'check_generic_brand_issues', 'check_fashion_brand_issues',
    'check_brand_in_name', 'check_wrong_variation',
    'check_generic_with_brand_in_name', 'check_missing_color',
    'check_weight_volume_in_name', 'check_incomplete_smartphone_name',
    'check_duplicate_products', 'check_miscellaneous_category',
    'compile_regex_patterns',
]
NG_SYMBOLS = [
    'check_nigeria_gift_card', 'check_nigeria_books', 'check_nigeria_tvs',
    'check_nigeria_hp_toners', 'check_nigeria_apple', 'check_nigeria_xmas_tree',
]

def _get_preqc_symbols() -> Tuple[dict, bool]:
    try:
        from _preqc_registry import REGISTRY
    except ImportError:
        return {}, False
    if not REGISTRY: return {}, False
    missing = [n for n in REQUIRED_SYMBOLS if n not in REGISTRY]
    syms    = {k: REGISTRY[k] for k in REQUIRED_SYMBOLS if k in REGISTRY}
    for s in NG_SYMBOLS:
        if s in REGISTRY: syms[s] = REGISTRY[s]
    return syms, len(missing) == 0


# ─────────────────────────────────────────────────────────────────────────────
# POST-QC SPECIFIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _empty(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=df.columns)

def check_fake_discount(df: pd.DataFrame, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    if not {'GLOBAL_PRICE', 'OLD_PRICE'}.issubset(df.columns): return _empty(df)
    d = df.copy()
    d['_p'] = pd.to_numeric(d['GLOBAL_PRICE'], errors='coerce')
    d['_o'] = pd.to_numeric(d['OLD_PRICE'],    errors='coerce')
    mask    = d['_p'].notna() & d['_o'].notna() & (d['_p'] > 0) & (d['_o'] > d['_p'] * multiplier_threshold)
    flagged = d[mask].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = flagged.apply(
            lambda r: f"Old price {float(r['_o']):,.0f} is {float(r['_o'])/float(r['_p']):,.0f}× current {float(r['_p']):,.0f}",
            axis=1,
        )
    return flagged.drop(columns=['_p', '_o'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_low_rating(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    if 'RATING' not in df.columns: return _empty(df)
    d = df.copy()
    d['_r'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_r'].notna() & (d['_r'] < threshold)].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Rating " + flagged['_r'].round(1).astype(str) + " < " + str(threshold)
    return flagged.drop(columns=['_r'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_no_ratings(df: pd.DataFrame) -> pd.DataFrame:
    if 'RATING' not in df.columns: return _empty(df)
    d = df.copy()
    d['_r'] = pd.to_numeric(d['RATING'], errors='coerce')
    flagged = d[d['_r'].isna()].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "No customer ratings"
    return flagged.drop(columns=['_r'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT BUILDER
# Runs all checks and produces the exact report format matching the sample.
# ─────────────────────────────────────────────────────────────────────────────

def _yn(flag: bool) -> str:
    return "YES" if flag else "NO"


def build_quality_report(
    df: pd.DataFrame,
    support_files: Dict,
    country_code: str,
) -> pd.DataFrame:
    """
    Run every applicable check and build the full quality report.
    Returns a DataFrame with all REPORT_COLS columns.
    """
    country_name = COUNTRY_CODE_TO_NAME.get(country_code, 'Kenya')
    symbols, have_preqc = _get_preqc_symbols()

    # ── Run all checks and collect per-SKU issue sets ────────────────────────
    # issue_map: {sku: [(severity, column_flag, message), ...]}
    issue_map: Dict[str, List[Tuple[str, str, str]]] = {
        str(r['PRODUCT_SET_SID']).strip(): [] for _, r in df.iterrows()
    }

    def _register(res: pd.DataFrame, severity: str, col_flag: str, msg_template: str = ""):
        if res.empty or 'PRODUCT_SET_SID' not in res.columns: return
        for _, r in res.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid not in issue_map: continue
            det = str(r.get('Comment_Detail', msg_template or col_flag)).strip()
            if not det or det == 'nan': det = msg_template or col_flag
            issue_map[sid].append((severity, col_flag, det))

    if have_preqc:
        crx = symbols['compile_regex_patterns']
        # Slice suspected_fake to the current country — each sheet uses local currency
        _sf_all = support_files.get('suspected_fake', {})
        suspected_fake_df = _sf_all.get(country_code, pd.DataFrame()) if isinstance(_sf_all, dict) else pd.DataFrame()

        # CRITICAL
        _register(symbols['check_prohibited_products'](df, prohibited_rules=support_files.get('prohibited_words_all', {}).get(country_code, [])),
                  'critical', 'Prohibited Item')
        _register(symbols['check_counterfeit_sneakers'](df, sneaker_category_codes=support_files.get('sneaker_category_codes', []), sneaker_sensitive_brands=support_files.get('sneaker_sensitive_brands', [])),
                  'critical', 'Counterfeit Flag', 'Counterfeit sneaker detected')
        _register(symbols['check_counterfeit_jerseys'](df, jerseys_data=support_files.get('jerseys_data', {}), country_code=country_code),
                  'critical', 'Counterfeit Flag', 'Counterfeit jersey detected')
        _register(symbols['check_suspected_fake_products'](df, suspected_fake_df=suspected_fake_df),
                  'critical', 'Counterfeit Flag', 'Suspected counterfeit (price)') # also flags Suspicious Price below

        # HIGH
        _register(symbols['check_miscellaneous_category'](df, categories_list=support_files.get('categories_names_list', []), cat_path_to_code=support_files.get('cat_path_to_code', {}), code_to_path=support_files.get('code_to_path', {})),
                  'high', 'Wrong Category')
        _register(symbols['check_restricted_brands'](df, country_rules=support_files.get('restricted_brands_all', {}).get(country_name, [])),
                  'high', 'Restricted Brand')
        _register(symbols['check_unnecessary_words'](df, pattern=crx(support_files.get('blacklisted_words', []))),
                  'high', 'Blacklisted Keyword')
        _register(symbols['check_suspected_fake_products'](df, suspected_fake_df=suspected_fake_df),
                  'high', 'Suspicious Price', 'Price below threshold for brand/category')

        # HIGH — Nigeria-specific seller restrictions
        if country_code == "NG":
            _ng = support_files.get("ng_qc_rules", {})
            for sym_name in ['check_nigeria_gift_card', 'check_nigeria_books', 'check_nigeria_tvs',
                             'check_nigeria_hp_toners', 'check_nigeria_apple', 'check_nigeria_xmas_tree']:
                fn = symbols.get(sym_name)
                if fn:
                    try:
                        _register(fn(df, ng_rules=_ng), 'high', 'NG Seller Restriction')
                    except Exception: pass

        # MEDIUM — naming issues
        for fn_name, msg in [
            ('check_unnecessary_words',            'Unnecessary words in product name'),
            ('check_single_word_name',             'Product name is a single word'),
            ('check_brand_in_name',                'Brand name repeated in product name'),
            ('check_generic_brand_issues',         'Generic brand used in wrong category'),
            ('check_fashion_brand_issues',         'Fashion brand used outside fashion category'),
            ('check_generic_with_brand_in_name',   'Product has Generic brand but name contains real brand'),
            ('check_incomplete_smartphone_name',   'Smartphone name missing memory/storage spec'),
            ('check_weight_volume_in_name',        'Product name missing weight/volume'),
        ]:
            fn = symbols.get(fn_name)
            if not fn: continue
            try:
                if fn_name == 'check_unnecessary_words':
                    res = fn(df, pattern=crx(support_files.get('unnecessary_words', [])))
                elif fn_name in ('check_generic_brand_issues', 'check_fashion_brand_issues'):
                    res = fn(df, valid_category_codes_fas=support_files.get('category_fas', []))
                elif fn_name == 'check_generic_with_brand_in_name':
                    res = fn(df, brands_list=support_files.get('known_brands', []))
                elif fn_name == 'check_incomplete_smartphone_name':
                    res = fn(df, smartphone_category_codes=support_files.get('smartphone_category_codes', []))
                elif fn_name == 'check_weight_volume_in_name':
                    res = fn(df, weight_category_codes=support_files.get('weight_category_codes', []))
                elif fn_name == 'check_single_word_name':
                    res = fn(df, book_category_codes=support_files.get('book_category_codes', []), books_data=support_files.get('books_data', {}))
                else:
                    res = fn(df)
                _register(res, 'medium', 'Naming Issue', msg)
            except Exception as exc:
                logger.debug(f"build_quality_report {fn_name}: {exc}")

        # MEDIUM — wrong variation
        try:
            _register(symbols['check_wrong_variation'](df, allowed_variation_codes=list(set(support_files.get('variation_allowed_codes', []) + support_files.get('category_fas', [])))),
                      'medium', 'Naming Issue', 'Wrong variation — should be separate SKUs')
        except Exception: pass

        # MEDIUM — seller/product-type restrictions (not NG-specific)
        try:
            _register(symbols['check_seller_approved_for_perfume'](df, perfume_category_codes=support_files.get('perfume_category_codes', []), perfume_data=support_files.get('perfume_data', {}), country_code=country_code),
                      'high', 'Restricted Brand', 'Seller not approved for perfume')
        except Exception: pass
        try:
            _register(symbols['check_perfume_tester'](df, perfume_category_codes=support_files.get('perfume_category_codes', []), perfume_data=support_files.get('perfume_data', {})),
                      'critical', 'Prohibited Item', 'Perfume tester — not permitted')
        except Exception: pass
        try:
            _register(symbols['check_seller_approved_for_books'](df, books_data=support_files.get('books_data', {}), country_code=country_code, book_category_codes=support_files.get('book_category_codes', [])),
                      'high', 'Restricted Brand', 'Seller not approved for books')
        except Exception: pass
        try:
            _register(symbols['check_refurb_seller_approval'](df, refurb_data=support_files.get('refurb_data', {}), country_code=country_code),
                      'high', 'Restricted Brand', 'Seller not approved for refurbished devices')
        except Exception: pass

    # ── Post-QC specific ─────────────────────────────────────────────────────
    try:
        _register(check_fake_discount(df), 'high', 'Suspicious Price')
    except Exception: pass

    # LOW
    # (Repeated description — only if we have description data)
    if 'DESCRIPTION' in df.columns:
        dup_desc = df[df.duplicated(subset=['DESCRIPTION'], keep=False) & df['DESCRIPTION'].notna() & (df['DESCRIPTION'].astype(str).str.strip() != '')].copy()
        if not dup_desc.empty:
            dup_desc['Comment_Detail'] = "Description appears in multiple products"
            _register(dup_desc, 'low', 'Repeated Description')

    # ── Image/description checks (only if data available from scraper) ────────
    has_img_count = 'No. of Images' in df.columns
    has_desc      = 'DESCRIPTION' in df.columns or 'DESCRIPTION_HTML' in df.columns

    if has_img_count:
        few_imgs = df[pd.to_numeric(df.get('No. of Images', pd.Series()), errors='coerce').fillna(1) < 5].copy()
        if not few_imgs.empty:
            few_imgs['Comment_Detail'] = few_imgs.apply(
                lambda r: f"Product has only {r.get('No. of Images', '?')} image(s), minimum recommended is 5",
                axis=1,
            )
            _register(few_imgs, 'medium', 'Few Images (<5)')

    if has_desc:
        desc_col = 'DESCRIPTION_HTML' if 'DESCRIPTION_HTML' in df.columns else 'DESCRIPTION'
        empty_desc = df[df[desc_col].isna() | (df[desc_col].astype(str).str.strip().isin(['', 'nan', 'None']))].copy()
        if not empty_desc.empty:
            empty_desc['Comment_Detail'] = "Product has no description"
            _register(empty_desc, 'medium', 'Empty Description')
        thin_desc = df[
            (~df[desc_col].isna()) &
            (df[desc_col].astype(str).str.len() < 100) &
            (~df[desc_col].astype(str).str.strip().isin(['', 'nan', 'None']))
        ].copy()
        if not thin_desc.empty:
            thin_desc['Comment_Detail'] = "Product description is too short"
            _register(thin_desc, 'medium', 'Thin Description')
        # Missing images in description (no <img> tag)
        no_img_desc = df[
            (~df[desc_col].isna()) &
            (~df[desc_col].astype(str).str.contains(r'<img', case=False, na=False))
        ].copy()
        if not no_img_desc.empty:
            no_img_desc['Comment_Detail'] = "Product description does not contain any images"
            _register(no_img_desc, 'medium', 'Missing Images in Desc')

    # ── Build report rows ─────────────────────────────────────────────────────
    rows: List[Dict] = []

    for _, r in df.iterrows():
        sid    = str(r.get('PRODUCT_SET_SID', r.get('SKU', ''))).strip()
        issues = issue_map.get(sid, [])

        # Deduplicate (same col_flag can appear from multiple checks — keep unique messages)
        seen_col_msg: set = set()
        deduped: List[Tuple[str, str, str]] = []
        for sev, col, msg in issues:
            key = (col, msg[:80])
            if key not in seen_col_msg:
                seen_col_msg.add(key)
                deduped.append((sev, col, msg))
        issues = deduped

        # Count by severity
        c_cnt = sum(1 for s, _, _ in issues if s == 'critical')
        h_cnt = sum(1 for s, _, _ in issues if s == 'high')
        m_cnt = sum(1 for s, _, _ in issues if s == 'medium')
        l_cnt = sum(1 for s, _, _ in issues if s == 'low')
        total = c_cnt + h_cnt + m_cnt + l_cnt

        quality = max(0, round(100 - c_cnt * _W['critical'] - h_cnt * _W['high']
                                   - m_cnt * _W['medium']  - l_cnt * _W['low']))

        # YES/NO flags
        cols_flagged = {col for _, col, _ in issues}

        # Top Issues text (severity-tagged, pipe-separated)
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_issues  = sorted(issues, key=lambda x: severity_order.get(x[0], 9))
        top_issues     = " | ".join(f"[{sev}] {msg}" for sev, _, msg in sorted_issues)

        # Image count
        img_count = r.get('No. of Images', 'N/A')

        row: Dict = {
            'SKU':           sid,
            'Name':          str(r.get('NAME', r.get('Name', ''))),
            'Brand':         str(r.get('BRAND', r.get('Brand', ''))),
            'Category':      str(r.get('CATEGORY', r.get('Category', ''))),
            'Price':         r.get('GLOBAL_PRICE', r.get('Price', '')),
            'Old Price':     r.get('OLD_PRICE', r.get('Old Price', '')),
            'Discount':      r.get('DISCOUNT', r.get('Discount', '')),
            'Rating':        r.get('RATING', r.get('Rating', '')),
            'Total Ratings': r.get('TOTAL_RATINGS', r.get('Total Ratings', '')),
            'Seller':        str(r.get('SELLER_NAME', r.get('Seller', ''))),
            'Jumia Express': r.get('Jumia Express', 'No'),
            'Shop Global':   r.get('Shop Global', 'No'),
            'Image URL':     str(r.get('MAIN_IMAGE', r.get('Image URL', ''))),
            'Product URL':   str(r.get('PRODUCT_URL', r.get('Product URL', ''))),
            'Stock':         r.get('STOCK_STATUS', r.get('Stock', 'In Stock')),
            'Tags':          r.get('Tags', ''),
            'Country':       r.get('Country', country_code),
            'Quality Score': quality,
            'Critical Issues': c_cnt,
            'High Issues':     h_cnt,
            'Medium Issues':   m_cnt,
            'Low Issues':      l_cnt,
            'Total Issues':    total,
            'Top Issues':      top_issues,
            'No. of Images':   img_count,
            'Few Images (<5)':           _yn('Few Images (<5)'           in cols_flagged),
            'Non-White Background':      _yn('Non-White Background'      in cols_flagged),
            'Low Resolution Images':     _yn('Low Resolution Images'     in cols_flagged),
            'Empty Description':         _yn('Empty Description'         in cols_flagged),
            'Thin Description':          _yn('Thin Description'          in cols_flagged),
            'Muddled Description':       _yn('Muddled Description'       in cols_flagged),
            'Repeated Description':      _yn('Repeated Description'      in cols_flagged),
            'Missing Images in Desc':    _yn('Missing Images in Desc'    in cols_flagged),
            'Naming Issue':              _yn('Naming Issue'              in cols_flagged),
            'Wrong Category':            _yn('Wrong Category'            in cols_flagged),
            'Prohibited Item':           _yn('Prohibited Item'           in cols_flagged),
            'Blacklisted Keyword':       _yn('Blacklisted Keyword'       in cols_flagged),
            'Restricted Brand':          _yn('Restricted Brand'          in cols_flagged),
            'NG Seller Restriction':     _yn('NG Seller Restriction'     in cols_flagged),
            'Counterfeit Flag':          _yn('Counterfeit Flag'          in cols_flagged),
            'Suspicious Price':          _yn('Suspicious Price'          in cols_flagged),
        }
        rows.append(row)

    report = pd.DataFrame(rows)
    # Ensure all report columns present
    for col in REPORT_COLS:
        if col not in report.columns:
            report[col] = 'NO' if col not in ('Quality Score', 'Critical Issues', 'High Issues',
                                               'Medium Issues', 'Low Issues', 'Total Issues',
                                               'No. of Images') else 0
    return report[REPORT_COLS]


# Legacy run_checks kept for backward compatibility with app.py's post-QC wiring
def run_checks(df, support_files, country_code=None):
    if not country_code:
        sel = st.session_state.get('pq_country', st.session_state.get('selected_country', 'Kenya'))
        country_code = COUNTRY_NAME_TO_CODE.get(sel, sel) if sel in COUNTRY_NAME_TO_CODE else sel
    report = build_quality_report(df, support_files, country_code)
    # Build minimal results dict for compatibility
    results: Dict[str, pd.DataFrame] = {}
    return report, results


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_export(report: pd.DataFrame, country_code: str = "KE") -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        report.to_excel(writer, sheet_name='QC Report', index=False)
        wb, ws = writer.book, writer.sheets['QC Report']

        # Header format
        hdr_fmt = wb.add_format({
            'bold': True, 'bg_color': '#F68B1E', 'font_color': '#FFFFFF',
            'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True,
        })
        for ci, col in enumerate(report.columns):
            ws.write(0, ci, col, hdr_fmt)

        # Conditional formats for YES/NO columns
        yes_fmt = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        ok_fmt  = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        yn_cols = [c for c in REPORT_COLS if c in ('Few Images (<5)', 'Non-White Background',
                   'Low Resolution Images', 'Empty Description', 'Thin Description',
                   'Muddled Description', 'Repeated Description', 'Missing Images in Desc',
                   'Naming Issue', 'Wrong Category', 'Prohibited Item', 'Blacklisted Keyword',
                   'Restricted Brand', 'NG Seller Restriction', 'Counterfeit Flag', 'Suspicious Price')]
        for col_name in yn_cols:
            if col_name not in report.columns: continue
            ci = report.columns.get_loc(col_name)
            ws.conditional_format(1, ci, len(report), ci,
                {'type': 'cell', 'criteria': 'equal', 'value': '"YES"', 'format': yes_fmt})
            ws.conditional_format(1, ci, len(report), ci,
                {'type': 'cell', 'criteria': 'equal', 'value': '"NO"',  'format': ok_fmt})

        # Quality Score colour scale (red→yellow→green)
        if 'Quality Score' in report.columns:
            qi = report.columns.get_loc('Quality Score')
            ws.conditional_format(1, qi, len(report), qi, {
                'type': '3_color_scale',
                'min_color': '#F8696B', 'mid_color': '#FFEB84', 'max_color': '#63BE7B',
                'min_type': 'num', 'min_value': 0,
                'mid_type': 'num', 'mid_value': 70,
                'max_type': 'num', 'max_value': 100,
            })

        # Auto-fit columns
        for ci, col in enumerate(report.columns):
            max_len = max(len(str(col)), report[col].astype(str).str.len().max() if len(report) > 0 else 0)
            ws.set_column(ci, ci, min(max_len + 2, 60))

    out.seek(0)
    return out.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(n: int, title: str):
    st.markdown(
        f"<div style='background:{JUMIA_COLORS['primary_orange']};color:#fff;"
        f"padding:10px 16px;border-radius:8px 8px 0 0;font-weight:700;font-size:15px;'>"
        f"{n} · {title}</div>",
        unsafe_allow_html=True,
    )


def render_post_qc_section(support_files: Dict) -> None:
    st.header(":material/fact_check: Post-QC Validator", anchor=False)
    st.caption("Analyse products already live on Jumia and score them against the full QC ruleset.")

    # ══════════════════════════════════════════════════════════════════════════
    # ① COUNTRY
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(1, "Select Country")
    with st.container(border=True):
        country_options = list(COUNTRY_CODE_TO_NAME.values())
        _default = st.session_state.get('pq_country',
                   st.session_state.get('selected_country', 'Kenya'))
        if _default not in country_options: _default = 'Kenya'

        pq_country = st.segmented_control(
            "Post-QC Country", country_options, default=_default,
            key="pq_country_ctrl", label_visibility="collapsed",
        )
        if pq_country and pq_country != st.session_state.get('pq_country'):
            st.session_state['pq_country']      = pq_country
            st.session_state['post_qc_data']    = pd.DataFrame()
            st.session_state['post_qc_report']  = pd.DataFrame()
            st.session_state.pop('pq_page', None)
            if 'exports_cache' in st.session_state:
                st.session_state['exports_cache'].pop('post_qc_export', None)
            st.rerun()

    country      = st.session_state.get('pq_country', 'Kenya')
    country_code = COUNTRY_NAME_TO_CODE.get(country, 'KE')
    base_url     = COUNTRY_BASE_URLS[country_code]

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ② LOAD PRODUCTS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(2, "Load Products")
    with st.container(border=True):
        tab_kw, tab_cat, tab_file = st.tabs([
            "🔍 Keyword Search", "🔗 Category URL", "📁 Upload File"
        ])

        # ── TAB: Keyword search ───────────────────────────────────────────────
        with tab_kw:
            st.info(
                f"Enter any search keyword — the same way a customer would search on **{country}** Jumia.  \n"
                f"Example: `laptop`, `iphone 14`, `perfume`, `hp printer`",
                icon=":material/search:",
            )
            kw_input = st.text_input("Search keyword", placeholder="e.g. laptop", key="pq_kw")
            kw_c1, kw_c2 = st.columns([3, 1])
            with kw_c1:
                kw_pages = st.slider("Pages (40 products/page)", 1, 50, 3, key="pq_kw_pages")
            with kw_c2:
                st.metric("≈ Products", f"{kw_pages * 40:,}")

            if st.button("🔍 Search", type="primary",
                         disabled=not bool(kw_input and kw_input.strip()),
                         key="pq_kw_btn"):
                pb  = st.progress(0, text="Starting…")
                stx = st.empty()
                def _cb(cur, tot, n):
                    pb.progress(int(cur/max(tot,1)*100), text=f"Page {cur}/{tot} — {n} products")
                    stx.caption(f"Scraping page {cur} of {tot}…")
                with st.spinner("Searching Jumia…"):
                    scraped, pages_done = scrape_by_keyword(
                        kw_input.strip(), country_code,
                        max_pages=kw_pages, progress_callback=_cb,
                    )
                pb.empty(); stx.empty()
                if scraped.empty:
                    st.error("No products found. Try a different keyword.", icon=":material/error:")
                else:
                    scraped = _resolve_cat_codes(scraped, support_files)
                    st.session_state['post_qc_data']   = scraped
                    st.session_state['post_qc_report'] = pd.DataFrame()
                    st.session_state.pop('pq_page', None)
                    if 'exports_cache' in st.session_state:
                        st.session_state['exports_cache'].pop('post_qc_export', None)
                    st.success(f"✅ **{len(scraped):,} products** from keyword `{kw_input}` — {pages_done} page(s).")
                    st.rerun()

        # ── TAB: Category URL ─────────────────────────────────────────────────
        with tab_cat:
            st.info(
                f"Paste a **{country}** Jumia category URL.  \n"
                f"Example: `{base_url}/phones-tablets/`",
                icon=":material/link:",
            )
            cat_url = st.text_input("Category URL", placeholder=f"{base_url}/phones-tablets/", key="pq_cat_url")

            if cat_url and cat_url.strip().startswith("http"):
                exp_d   = base_url.replace("https://","")
                giv_d   = cat_url.strip().replace("https://","").split("/")[0]
                if exp_d not in giv_d and giv_d not in exp_d:
                    st.warning(f"URL domain `{giv_d}` doesn't match **{country}** (`{exp_d}`).", icon=":material/warning:")

            cat_c1, cat_c2 = st.columns([3, 1])
            with cat_c1:
                cat_pages = st.slider("Pages (40 products/page)", 1, 50, 3, key="pq_cat_pages")
            with cat_c2:
                st.metric("≈ Products", f"{cat_pages * 40:,}")

            if st.button("🔗 Scrape Category", type="primary",
                         disabled=not bool(cat_url and cat_url.strip().startswith("http")),
                         key="pq_cat_btn"):
                pb  = st.progress(0, text="Connecting…")
                stx = st.empty()
                def _cb2(cur, tot, n):
                    pb.progress(int(cur/max(tot,1)*100), text=f"Page {cur}/{tot} — {n} products")
                    stx.caption(f"Scraping page {cur} of {tot}…")
                with st.spinner("Scraping Jumia…"):
                    scraped, pages_done = scrape_by_category_url(
                        cat_url.strip(), country_code,
                        max_pages=cat_pages, progress_callback=_cb2,
                    )
                pb.empty(); stx.empty()
                if scraped.empty:
                    st.error("No products found. Check the URL is a valid Jumia category page.", icon=":material/error:")
                else:
                    scraped = _resolve_cat_codes(scraped, support_files)
                    st.session_state['post_qc_data']   = scraped
                    st.session_state['post_qc_report'] = pd.DataFrame()
                    st.session_state.pop('pq_page', None)
                    if 'exports_cache' in st.session_state:
                        st.session_state['exports_cache'].pop('post_qc_export', None)
                    st.success(f"✅ **{len(scraped):,} products** across {pages_done} page(s).")
                    st.rerun()

        # ── TAB: Upload file ──────────────────────────────────────────────────
        with tab_file:
            uploaded = st.file_uploader("Upload CSV or XLSX", type=['csv', 'xlsx'], key="pq_file")
            if uploaded:
                try:
                    from io import BytesIO as _BIO
                    raw = (pd.read_excel(_BIO(uploaded.read()), engine='openpyxl', dtype=str)
                           if uploaded.name.endswith('.xlsx')
                           else pd.read_csv(_BIO(uploaded.read()), dtype=str))
                    norm = normalize_post_qc(raw, support_files.get('category_map', {}))
                    norm['ACTIVE_STATUS_COUNTRY'] = country_code
                    norm['Country']               = country_code
                    st.session_state['post_qc_data']   = norm
                    st.session_state['post_qc_report'] = pd.DataFrame()
                    st.session_state.pop('pq_page', None)
                    if 'exports_cache' in st.session_state:
                        st.session_state['exports_cache'].pop('post_qc_export', None)
                    st.success(f"Loaded **{len(norm):,}** rows from `{uploaded.name}`")
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")

    # Guard: no data
    data_pq = st.session_state.get('post_qc_data', pd.DataFrame())
    if data_pq.empty:
        st.info("No products loaded. Use one of the three tabs above.", icon=":material/info:")
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ③ RUN VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(3, "Run Validation")
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        cat_ok  = 'CATEGORY_CODE' in data_pq.columns
        res_n   = int(data_pq['CATEGORY_CODE'].str.match(r'^\d+$').sum()) if cat_ok else 0
        m1.metric("Products loaded",         f"{len(data_pq):,}")
        m2.metric("Category codes resolved", f"{res_n:,} / {len(data_pq):,}")
        m3.metric("Country",                 f"{country} ({country_code})")

        if res_n == 0 and len(data_pq) > 0:
            st.warning(
                "0 category codes resolved — category-dependent checks will be limited. "
                "Ensure `category_map.xlsx` is in the app root.",
                icon=":material/warning:",
            )

        if st.button(
            f"▶  Run {country} Quality Check  ({len(data_pq):,} products)",
            type="primary", use_container_width=True, key="pq_run_btn",
        ):
            with st.spinner(f"Running full QC for {country}…"):
                report = build_quality_report(data_pq, support_files, country_code)
            st.session_state['post_qc_report'] = report
            st.session_state.pop('pq_page', None)
            if 'exports_cache' in st.session_state:
                st.session_state['exports_cache'].pop('post_qc_export', None)
            st.rerun()

    # Guard: not validated yet
    report = st.session_state.get('post_qc_report', pd.DataFrame())
    if report.empty:
        return

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ④ RESULTS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(4, "Results")
    with st.container(border=True):

        # Top-line metrics
        avg_score     = report['Quality Score'].mean()
        n_any_issue   = (report['Total Issues'] > 0).sum()
        n_critical    = (report['Critical Issues'] > 0).sum()
        n_clean       = (report['Total Issues'] == 0).sum()
        issue_rate    = n_any_issue / max(len(report), 1) * 100

        n_cols = 5 if st.session_state.get('layout_mode') == 'wide' else 3
        p_cols = st.columns(n_cols)
        for i, (lbl, val, col) in enumerate([
            ("Products",      len(report),              JUMIA_COLORS['dark_gray']),
            ("Avg Quality",   f"{avg_score:.0f}/100",   JUMIA_COLORS['success_green'] if avg_score >= 80 else JUMIA_COLORS['warning_yellow']),
            ("Have Issues",   f"{n_any_issue:,}",       JUMIA_COLORS['jumia_red']),
            ("Critical",      f"{n_critical:,}",        '#C00000'),
            ("✅ Clean",       f"{n_clean:,}",           JUMIA_COLORS['success_green']),
        ]):
            with p_cols[i % n_cols]:
                st.markdown(f"<div style='height:4px;background:{col};border-radius:4px 4px 0 0;'></div>",
                            unsafe_allow_html=True)
                st.metric(lbl, val)

        # Flag breakdown chips
        yn_flag_cols = [
            'Prohibited Item', 'Counterfeit Flag', 'Wrong Category', 'Blacklisted Keyword',
            'Restricted Brand', 'NG Seller Restriction', 'Suspicious Price',
            'Naming Issue', 'Few Images (<5)', 'Missing Images in Desc',
            'Empty Description', 'Thin Description', 'Repeated Description',
        ]
        flag_counts = {c: int((report[c] == 'YES').sum()) for c in yn_flag_cols if c in report.columns and (report[c] == 'YES').sum() > 0}
        if flag_counts:
            st.markdown("**Issues breakdown:**")
            severity_col = {
                'Prohibited Item': '#C00000', 'Counterfeit Flag': '#C00000',
                'Wrong Category': JUMIA_COLORS['jumia_red'], 'Blacklisted Keyword': JUMIA_COLORS['jumia_red'],
                'Restricted Brand': JUMIA_COLORS['jumia_red'], 'NG Seller Restriction': JUMIA_COLORS['jumia_red'],
                'Suspicious Price': JUMIA_COLORS['warning_yellow'],
            }
            chips = []
            for fname, cnt in sorted(flag_counts.items(), key=lambda x: -x[1]):
                bg = severity_col.get(fname, JUMIA_COLORS['medium_gray'])
                chips.append(
                    f"<span style='background:{bg};color:#fff;border-radius:12px;"
                    f"padding:4px 12px;margin:3px;font-size:12px;font-weight:700;"
                    f"display:inline-block;'>{fname}&nbsp;<b>{cnt}</b></span>"
                )
            st.markdown("".join(chips), unsafe_allow_html=True)

        st.markdown("---")

        # ── Filters ──────────────────────────────────────────────────────────
        fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
        with fc1:
            search_q = st.text_input("🔍 Search", placeholder="SKU, Name, Brand, Seller…", key="pq_search")
        with fc2:
            show_opts = ["All", "Issues Only", "Critical Only", "Clean Only"]
            show_sel  = st.selectbox("Show", show_opts, key="pq_show")
        with fc3:
            flag_filter_opts = ["(all)"] + [c for c in yn_flag_cols if c in report.columns]
            flag_sel = st.selectbox("Flag", flag_filter_opts, key="pq_flag_sel")
        with fc4:
            min_score, max_score = int(report['Quality Score'].min()), int(report['Quality Score'].max())
            score_range = st.slider("Quality Score", 0, 100,
                                    (min_score, max_score), key="pq_score_range")

        disp = report.copy()
        if show_sel == "Issues Only":
            disp = disp[disp['Total Issues'] > 0]
        elif show_sel == "Critical Only":
            disp = disp[disp['Critical Issues'] > 0]
        elif show_sel == "Clean Only":
            disp = disp[disp['Total Issues'] == 0]
        if flag_sel != "(all)":
            disp = disp[disp[flag_sel] == 'YES']
        disp = disp[disp['Quality Score'].between(score_range[0], score_range[1])]
        if search_q:
            sq   = search_q.strip().lower()
            mask = (
                disp['SKU'].astype(str).str.lower().str.contains(sq, na=False)
                | disp['Name'].astype(str).str.lower().str.contains(sq, na=False)
                | disp['Brand'].astype(str).str.lower().str.contains(sq, na=False)
                | disp['Seller'].astype(str).str.lower().str.contains(sq, na=False)
            )
            disp = disp[mask]

        # ── Pagination (40 rows = one Jumia listing page) ─────────────────────
        ROWS_PER_PAGE = 40
        total_rows    = len(disp)
        total_pages   = max(1, (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        if st.session_state.get('pq_page', 0) >= total_pages:
            st.session_state['pq_page'] = 0

        pg_l, pg_m, pg_r = st.columns([1, 3, 1], vertical_alignment="center")
        with pg_l:
            if st.button("◀ Prev", key="pq_prev", disabled=st.session_state.get('pq_page', 0) == 0):
                st.session_state['pq_page'] -= 1
                st.rerun()
        with pg_m:
            cur  = st.session_state.get('pq_page', 0)
            ps   = cur * ROWS_PER_PAGE
            pe   = min(ps + ROWS_PER_PAGE, total_rows)
            st.caption(f"Rows {ps+1}–{pe} of {total_rows:,}  (Page {cur+1}/{total_pages})")
            jump = st.number_input("Page", 1, max(total_pages, 1), cur+1, step=1,
                                   key="pq_jump", label_visibility="collapsed")
            if jump - 1 != cur:
                st.session_state['pq_page'] = jump - 1
                st.rerun()
        with pg_r:
            if st.button("Next ▶", key="pq_next", disabled=st.session_state.get('pq_page', 0) >= total_pages - 1):
                st.session_state['pq_page'] += 1
                st.rerun()

        cur      = st.session_state.get('pq_page', 0)
        page_df  = disp.iloc[cur*ROWS_PER_PAGE : (cur+1)*ROWS_PER_PAGE].reset_index(drop=True)

        # ── Visible columns (core + link) ────────────────────────────────────
        # Add clickable link column
        page_display = page_df.copy()
        page_display.insert(0, 'Link', page_df['Product URL'])

        view_cols = ['Link', 'SKU', 'Name', 'Brand', 'Category', 'Price',
                     'Rating', 'Seller', 'Quality Score', 'Total Issues',
                     'Top Issues', 'Naming Issue', 'Wrong Category',
                     'Prohibited Item', 'Blacklisted Keyword', 'Restricted Brand',
                     'NG Seller Restriction', 'Counterfeit Flag', 'Suspicious Price',
                     'Few Images (<5)', 'Missing Images in Desc']
        # Only keep cols that exist
        view_cols = [c for c in view_cols if c in page_display.columns]

        st.dataframe(
            page_display[view_cols],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Link": st.column_config.LinkColumn(
                    "🔗", display_text="🔗", help="Open on Jumia", width="small",
                ),
                "SKU":           st.column_config.TextColumn("SKU",      pinned=True, width="medium"),
                "Name":          st.column_config.TextColumn("Name",     pinned=True, width="large"),
                "Quality Score": st.column_config.NumberColumn("Score",  format="%d"),
                "Total Issues":  st.column_config.NumberColumn("Issues", format="%d"),
                "Top Issues":    st.column_config.TextColumn("Top Issues", width="large"),
                "Price":         st.column_config.NumberColumn("Price",  format="%.0f"),
                "Rating":        st.column_config.NumberColumn("Rating", format="%.1f"),
            },
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ⑤ EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(5, "Export Full Report")
    with st.container(border=True):
        st.caption(
            "The export contains all columns matching the standard Jumia QC report format: "
            "Quality Score, severity counts, YES/NO flags per issue type, and Top Issues summary."
        )
        export_key = "post_qc_export"
        cache      = st.session_state.get('exports_cache', {})

        if export_key not in cache:
            if st.button("Generate Excel Report", type="primary",
                         icon=":material/download:", key="gen_pq_rpt"):
                with st.spinner("Building Excel report…"):
                    xlsx = build_export(report, country_code)
                    st.session_state.setdefault('exports_cache', {})[export_key] = xlsx
                st.rerun()
        else:
            date_str    = datetime.now().strftime('%Y-%m-%d')
            source_hint = (
                st.session_state.get('pq_kw', '') or
                st.session_state.get('pq_cat_url', '') or
                'Upload'
            )
            source_hint = source_hint[:20].replace(' ', '_').replace('/', '-')
            fname = f"PostQC_{country_code}_{source_hint}_{date_str}.xlsx"
            dl_c, cl_c = st.columns([4, 1])
            with dl_c:
                st.download_button(
                    f"⬇ Download — {fname}",
                    data=cache[export_key],
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary", icon=":material/file_download:", key="dl_pq_rpt",
                )
            with cl_c:
                if st.button("Clear", key="clr_pq_rpt"):
                    del st.session_state['exports_cache'][export_key]
                    st.rerun()
