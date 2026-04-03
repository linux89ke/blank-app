import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import st_yled
from io import BytesIO
from datetime import datetime
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import traceback
import json
import zipfile
import os
import shutil
import pickle
import concurrent.futures
from dataclasses import dataclass
import base64
import hashlib
import requests
from PIL import Image

from translations import LANGUAGES, get_translation

try:
    from postqc import detect_file_type, normalize_post_qc, run_checks as run_post_qc_checks, render_post_qc_section, load_category_map
except ImportError:
    pass

class DummyRegistry:
    pass

_reg = DummyRegistry()
_reg.REGISTRY = {}

try:
    from jumia_scraper import enrich_post_qc_df, COUNTRY_BASE_URLS as _SCRAPER_URLS
    _SCRAPER_AVAILABLE = True
except ImportError:
    _SCRAPER_AVAILABLE = False

# ── Category Matcher Engine ───────────────────────────────────────────────────
try:
    from category_matcher_engine import CategoryMatcherEngine, check_wrong_category, get_engine
    _CAT_MATCHER_AVAILABLE = True
except Exception as e:
    import logging
    import traceback
    logging.error(f"CRITICAL: Failed to import category_matcher_engine: {e}")
    logging.error(traceback.format_exc()) # This will print the EXACT line causing the crash
    _CAT_MATCHER_AVAILABLE = False
    
    def check_wrong_category(data, categories_list=None, cat_path_to_code=None, code_to_path=None, confidence_threshold=0.0):
        if 'CATEGORY' not in data.columns:
            return pd.DataFrame(columns=data.columns)
        flagged = data[data['CATEGORY'].astype(str).str.contains(
            "miscellaneous", case=False, na=False
        )].copy()
        if not flagged.empty:
            flagged['Comment_Detail'] = "Category contains 'Miscellaneous'"
        return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])


import threading as _threading

_engine_ready    = _threading.Event()
_engine_lock     = _threading.Lock()
_engine_instance = None   # set by background thread; never touched by Streamlit

def _load_engine_background():
    """Daemon thread: loads the engine so the first page render isn't blocked."""
    global _engine_instance
    if not _CAT_MATCHER_AVAILABLE:
        _engine_ready.set()
        return
    try:
        eng = get_engine()          # slow: TF-IDF fit + Firestore learning DB
        with _engine_lock:
            _engine_instance = eng
        logger.info("CategoryMatcherEngine loaded in background thread.")
    except Exception as e:
        logger.warning("Background engine load failed: %s", e)
    finally:
        _engine_ready.set()         # always unblock waiters

def _start_engine_preload():
    """Call once at module level. Safe to call multiple times."""
    if not _engine_ready.is_set():
        t = _threading.Thread(target=_load_engine_background, daemon=True)
        t.start()

_start_engine_preload()             # ← fires immediately when the module loads

@st.cache_resource(show_spinner=False)
def _get_cat_matcher_engine():
    """
    Returns the CategoryMatcherEngine instance.

    • If the background thread has finished   → returns immediately (zero wait).
    • If validation is triggered while loading → waits up to 20 s.
    • Never blocks the initial page render.
    """
    global _engine_instance
    if _engine_ready.is_set():
        with _engine_lock:
            return _engine_instance
    _engine_ready.wait(timeout=20)
    with _engine_lock:
        return _engine_instance


if 'load_category_map' not in dir():
    def load_category_map(filename: str = "category_map.xlsx") -> dict:
        return {}

logger = logging.getLogger(__name__)

# -------------------------------------------------
# CACHE DIRECTORIES & HELPERS
# -------------------------------------------------
PARQUET_CACHE_DIR = "app_cache_parquet"
FLAG_CACHE_DIR = "app_cache_flags"
os.makedirs(PARQUET_CACHE_DIR, exist_ok=True)
os.makedirs(FLAG_CACHE_DIR, exist_ok=True)

def prune_cache_dir(directory: str, max_files: int = 500):
    try:
        files = sorted(Path(directory).glob("*.pkl"), key=os.path.getmtime)
        stale = files[:-max_files]
        for f in stale:
            f.unlink(missing_ok=True)
        if stale:
            logger.info(f"Pruned {len(stale)} stale cache files from {directory}")
    except Exception as e:
        logger.warning(f"Cache pruning failed for {directory}: {e}")

prune_cache_dir(FLAG_CACHE_DIR)

try:
    for _cf in Path(FLAG_CACHE_DIR).glob("*.pkl"):
        if _cf.stat().st_size < 500:
            _cf.unlink(missing_ok=True)
except Exception:
    pass

def save_df_parquet(df, filename):
    try:
        df.to_parquet(os.path.join(PARQUET_CACHE_DIR, filename))
    except Exception as e:
        logger.warning(f"Failed to save parquet {filename}: {e}")

def load_df_parquet(filename):
    path = os.path.join(PARQUET_CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Failed to load parquet {filename}: {e}")
    return None

# -------------------------------------------------
# JUMIA THEME COLORS & GLOBAL CSS
# -------------------------------------------------
JUMIA_COLORS = {
    'primary_orange': '#F68B1E',
    'secondary_orange': '#FF9933',
    'jumia_red': '#E73C17',
    'dark_gray': '#313133',
    'medium_gray': '#5A5A5C',
    'light_gray': '#F5F5F5',
    'border_gray': '#E0E0E0',
    'success_green': '#4CAF50',
    'warning_yellow': '#FFC107',
    'white': '#FFFFFF',
    'black': '#000000'
}

# -------------------------------------------------
# CONSTANTS & MAPPING
# -------------------------------------------------
PRODUCTSETS_COLS = ["ProductSetSid", "ParentSKU", "Status", "Reason", "Comment", "FLAG", "SellerName"]
REJECTION_REASONS_COLS = ['CODE - REJECTION_REASON', 'COMMENT']

FULL_DATA_COLS = [
    "PRODUCT_SET_SID", "ACTIVE_STATUS_COUNTRY", "NAME", "BRAND", "CATEGORY", "CATEGORY_CODE",
    "FULL_CATEGORY_PATH",
    "COLOR", "COLOR_FAMILY", "MAIN_IMAGE", "VARIATION", "PARENTSKU", "SELLER_NAME", "SELLER_SKU",
    "GLOBAL_PRICE", "GLOBAL_SALE_PRICE", "TAX_CLASS", "FLAG", "LISTING_STATUS",
    "PRODUCT_WARRANTY", "WARRANTY_DURATION", "WARRANTY_ADDRESS", "WARRANTY_TYPE", "COUNT_VARIATIONS",
    "LIST_VARIATIONS"
]

GRID_COLS = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'SELLER_NAME', 'MAIN_IMAGE', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'COLOR']

COUNTRY_CURRENCY = {
    "Kenya":   {"code": "KES", "symbol": "KSh", "pair": "USD/KES"},
    "Uganda":  {"code": "UGX", "symbol": "USh", "pair": "USD/UGX"},
    "Nigeria": {"code": "NGN", "symbol": "₦",   "pair": "USD/NGN"},
    "Ghana":   {"code": "GHS", "symbol": "GH₵", "pair": "USD/GHS"},
    "Morocco": {"code": "MAD", "symbol": "MAD", "pair": "USD/MAD"},
}

@st.cache_data(ttl=3600)
def fetch_exchange_rate(country: str) -> float:
    cfg = COUNTRY_CURRENCY.get(country)
    if not cfg: return 1.0
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=3) as r:
            data = _json.loads(r.read())
        return float(data["rates"].get(cfg["code"], 1.0))
    except Exception as e:
        logger.warning(f"Exchange rate fetch failed for {country}: {e}")
        fallbacks = {"Kenya": 128.0, "Uganda": 3750.0, "Nigeria": 1550.0, "Ghana": 15.5, "Morocco": 10.1}
        return fallbacks.get(country, 1.0)

def format_local_price(usd_price, country: str) -> str:
    try:
        price = float(usd_price)
        if price <= 0: return ""
        cfg = COUNTRY_CURRENCY.get(country, {})
        rate = fetch_exchange_rate(country)
        local = price * rate
        symbol = cfg.get("symbol", "$")
        if cfg.get("code") in ("KES", "UGX", "NGN"): return f"{symbol} {local:,.0f}"
        else: return f"{symbol} {local:,.2f}"
    except (ValueError, TypeError): return ""

SPLIT_LIMIT = 9998

NEW_FILE_MAPPING = {
    'cod_productset_sid': 'PRODUCT_SET_SID',
    "2qz3wx4ec5rv6b7hnj8kl;'[]": 'PRODUCT_SET_SID',
    'dsc_name': 'NAME',
    'dsc_brand_name': 'BRAND',
    'cod_category_code': 'CATEGORY_CODE',
    'dsc_category_name': 'CATEGORY',
    'dsc_shop_seller_name': 'SELLER_NAME',
    'dsc_shop_active_country': 'ACTIVE_STATUS_COUNTRY',
    'cod_parent_sku': 'PARENTSKU',
    'color': 'COLOR',
    'colour': 'COLOR',
    'color_family': 'COLOR_FAMILY',
    'colour_family': 'COLOR_FAMILY',
    'colour family': 'COLOR_FAMILY',
    'color family': 'COLOR_FAMILY',
    'COLOUR FAMILY': 'COLOR_FAMILY',
    'list_seller_skus': 'SELLER_SKU',
    'image1': 'MAIN_IMAGE',
    'image_1': 'MAIN_IMAGE',
    'main_image': 'MAIN_IMAGE',
    'main image': 'MAIN_IMAGE',
    'image': 'MAIN_IMAGE',
    'img': 'MAIN_IMAGE',
    'img_url': 'MAIN_IMAGE',
    'image_url': 'MAIN_IMAGE',
    'photo': 'MAIN_IMAGE',
    'dsc_status': 'LISTING_STATUS',
    'dsc_shop_email': 'SELLER_EMAIL',
    'product_warranty': 'PRODUCT_WARRANTY',
    'warranty_duration': 'WARRANTY_DURATION',
    'warranty_address': 'WARRANTY_ADDRESS',
    'warranty_type': 'WARRANTY_TYPE',
    'count_variations': 'COUNT_VARIATIONS',
    'count variations': 'COUNT_VARIATIONS',
    'number of variations': 'COUNT_VARIATIONS',
    'list_variations': 'LIST_VARIATIONS',
    'list variations': 'LIST_VARIATIONS'
}

# -------------------------------------------------
# INITIALIZATION & CONTEXT
# -------------------------------------------------
if 'layout_mode' not in st.session_state: st.session_state.layout_mode = "wide"
if 'ui_lang' not in st.session_state: st.session_state.ui_lang = "en"
if 'final_report' not in st.session_state: st.session_state.final_report = pd.DataFrame()
if 'all_data_map' not in st.session_state: st.session_state.all_data_map = pd.DataFrame()
if 'post_qc_summary' not in st.session_state: st.session_state.post_qc_summary = pd.DataFrame()
if 'post_qc_results' not in st.session_state: st.session_state.post_qc_results = {}
if 'post_qc_data' not in st.session_state: st.session_state.post_qc_data = pd.DataFrame()
if 'file_mode' not in st.session_state: st.session_state.file_mode = None
if 'intersection_sids' not in st.session_state: st.session_state.intersection_sids = set()
if 'intersection_count' not in st.session_state: st.session_state.intersection_count = 0
if 'grid_page' not in st.session_state: st.session_state.grid_page = 0
if 'grid_items_per_page' not in st.session_state: st.session_state.grid_items_per_page = 50
if 'main_toasts' not in st.session_state: st.session_state.main_toasts = []
if 'exports_cache' not in st.session_state: st.session_state.exports_cache = {}
if 'do_scroll_top' not in st.session_state: st.session_state.do_scroll_top = False
if 'display_df_cache' not in st.session_state: st.session_state.display_df_cache = {}
if 'main_bridge_counter' not in st.session_state: st.session_state.main_bridge_counter = 0
if 'search_active' not in st.session_state: st.session_state.search_active = False
if 'pre_search_page' not in st.session_state: st.session_state.pre_search_page = 0
if 'desel_counter' not in st.session_state: st.session_state.desel_counter = 0
if 'quick_rejections' not in st.session_state: st.session_state.quick_rejections = {}
if 'batch_counter' not in st.session_state: st.session_state.batch_counter = 0
if 'clear_counter' not in st.session_state: st.session_state.clear_counter = 0
if 'ls_processed_flag' not in st.session_state: st.session_state.ls_processed_flag = False
if 'ls_read_trigger' not in st.session_state: st.session_state.ls_read_trigger = 0
if 'flags_expanded_initialized' not in st.session_state: st.session_state.flags_expanded_initialized = False

_pre_country = st.session_state.get("country_selector") or st.session_state.get("selected_country", "Kenya")
if _pre_country == "Morocco":
    st.session_state.ui_lang = "fr"
elif st.session_state.get("ui_lang") == "fr":
    st.session_state.ui_lang = "en"

def _t(key):
    return get_translation(st.session_state.ui_lang, key)

try: st.set_page_config(page_title="Product Tool", layout=st.session_state.layout_mode)
except: pass

st_yled.init()

rtl_css = """
        div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {
            direction: rtl !important;
            text-align: right !important;
        }
""" if st.session_state.ui_lang == "ar" else ""

st.markdown(f"""
    <style>
        {rtl_css}

        div[data-testid="stTextInput"]:has(input[placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE"]) {{
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
            opacity: 0 !important;
            z-index: -9999 !important;
        }}

        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

        :root {{
            --jumia-orange: {JUMIA_COLORS['primary_orange']};
            --jumia-red: {JUMIA_COLORS['jumia_red']};
            --jumia-dark: {JUMIA_COLORS['dark_gray']};
        }}
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        div[data-testid="stStatusWidget"] {{ z-index: 9999999 !important; }}
        .stButton > button {{ border-radius: 4px; font-weight: 600; transition: all 0.3s ease; }}
        .stButton > button[kind="primary"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; border: none !important; color: white !important; }}
        .stButton > button[kind="primary"]:hover {{ background-color: {JUMIA_COLORS['secondary_orange']} !important; box-shadow: 0 4px 8px rgba(246, 139, 30, 0.3); transform: translateY(-1px); }}
        .stButton > button[kind="secondary"] {{ background-color: white !important; border: 2px solid {JUMIA_COLORS['primary_orange']} !important; color: {JUMIA_COLORS['primary_orange']} !important; }}
        .stButton > button[kind="secondary"]:hover {{ background-color: {JUMIA_COLORS['light_gray']} !important; }}

        div[data-testid="stMetric"] {{
            background: {JUMIA_COLORS['light_gray']};
            border-radius: 0 0 8px 8px;
            padding: 12px 16px 16px 16px;
            text-align: center;
        }}
        div[data-testid="stMetricValue"] {{ color: {JUMIA_COLORS['dark_gray']}; font-weight: 700; font-size: 26px !important; }}
        div[data-testid="stMetricLabel"] {{ color: {JUMIA_COLORS['medium_gray']}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }}

        ::-webkit-scrollbar {{ width: 18px !important; height: 18px !important; }}
        ::-webkit-scrollbar-track {{ background: {JUMIA_COLORS['light_gray']}; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb {{ background: {JUMIA_COLORS['medium_gray']}; border-radius: 8px; border: 3px solid {JUMIA_COLORS['light_gray']}; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
        * {{ scrollbar-width: auto; scrollbar-color: {JUMIA_COLORS['medium_gray']} {JUMIA_COLORS['light_gray']}; }}

        div[data-baseweb="slider"] div[role="slider"] {{ height: 24px !important; width: 24px !important; border: 4px solid {JUMIA_COLORS['primary_orange']} !important; cursor: pointer !important; }}
        div[data-baseweb="slider"] > div > div {{ height: 12px !important; }}

        @media (prefers-color-scheme: dark) {{
            div[data-testid="stMetricValue"] {{ color: #F5F5F5 !important; }}
            div[data-testid="stMetricLabel"] {{ color: #B0B0B0 !important; }}
            div[data-testid="stMetric"] {{ background: #2a2a2e !important; }}
            h1, h2, h3 {{ color: #F5F5F5 !important; }}
            div[data-testid="stExpander"] summary {{ background-color: #2a2a2e !important; color: #F5F5F5 !important; }}
            div[data-testid="stExpander"] summary p, div[data-testid="stExpander"] summary span, div[data-testid="stExpander"] summary div {{ color: #F5F5F5 !important; }}
            div[data-testid="stDataFrame"] * {{ color: #F5F5F5 !important; }}
            .stDataFrame th {{ background-color: #2a2a2e !important; color: #F5F5F5 !important; }}
            .color-badge {{ background: #3a3a3e !important; border-color: #555 !important; color: #E0E0E0 !important; }}
            div[style*="position: sticky"], div[style*="position:sticky"] {{ background-color: #0e1117 !important; border-bottom-color: #2a2a2e !important; }}
            .stCaption, div[data-testid="stCaptionContainer"] p {{ color: #B0B0B0 !important; }}
            .prod-meta-text {{ color: #B0B0B0 !important; }}
            .prod-brand-text {{ color: {JUMIA_COLORS['secondary_orange']} !important; }}
            ::-webkit-scrollbar-track {{ background: #1e1e1e; border-color: #1e1e1e; }}
            ::-webkit-scrollbar-thumb {{ background: #555; border-color: #1e1e1e; }}
            ::-webkit-scrollbar-thumb:hover {{ background: {JUMIA_COLORS['primary_orange']}; }}
        }}

        div[data-testid="stExpander"] {{ border: 1px solid {JUMIA_COLORS['border_gray']}; border-radius: 8px; }}
        div[data-testid="stExpander"] summary {{ background-color: {JUMIA_COLORS['light_gray']}; padding: 12px; border-radius: 8px 8px 0 0; }}
        h1, h2, h3 {{ color: {JUMIA_COLORS['dark_gray']} !important; }}
        div[data-baseweb="segmented-control"] button {{ border-radius: 4px; }}
        div[data-baseweb="segmented-control"] button[aria-pressed="true"] {{ background-color: {JUMIA_COLORS['primary_orange']} !important; color: white !important; }}
        input[type="checkbox"]:checked {{ background-color: {JUMIA_COLORS['primary_orange']} !important; border-color: {JUMIA_COLORS['primary_orange']} !important; }}
        div[data-testid="stCheckbox"] {{ margin-top: 5px; margin-bottom: 5px; }}
    </style>
""", unsafe_allow_html=True)

def get_default_country():
    try:
        lang = st.context.headers.get("Accept-Language", "")
        if "KE" in lang: return "Kenya"
        if "UG" in lang: return "Uganda"
        if "NG" in lang: return "Nigeria"
        if "GH" in lang: return "Ghana"
        if "MA" in lang: return "Morocco"
    except: pass
    return "Kenya"

if 'selected_country' not in st.session_state: st.session_state.selected_country = get_default_country()

if st.session_state.main_toasts:
    for msg in st.session_state.main_toasts:
        if isinstance(msg, tuple): st.toast(msg[0], icon=msg[1])
        else: st.toast(msg)
    st.session_state.main_toasts.clear()

# -------------------------------------------------
# UTILITIES & EXTRACTION
# -------------------------------------------------
def clean_category_code(code) -> str:
    try:
        if pd.isna(code): return ""
        s = str(code).strip()
        if '.' in s: s = s.split('.')[0]
        return s
    except: return str(code).strip()

def normalize_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    noise = r'\b(new|sale|original|genuine|authentic|official|premium|quality|best|hot|2024|2025)\b'
    text = re.sub(noise, '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def create_match_key(row: pd.Series) -> str:
    name = normalize_text(row.get('NAME', ''))
    brand = normalize_text(row.get('BRAND', ''))
    color = normalize_text(row.get('COLOR', ''))
    return f"{brand}|{name}|{color}"

def df_hash(df: pd.DataFrame) -> str:
    try:
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    except Exception as e:
        logger.warning(f"df_hash primary failed, using fallback: {e}")
        fallback_str = str(df.shape) + str(df.columns.tolist())
        return hashlib.md5(fallback_str.encode()).hexdigest()

COLOR_PATTERNS = {
    'red': ['red', 'crimson', 'scarlet', 'maroon', 'burgundy', 'wine', 'ruby'],
    'blue': ['blue', 'navy', 'royal', 'sky', 'azure', 'cobalt', 'sapphire'],
    'green': ['green', 'lime', 'olive', 'emerald', 'mint', 'forest', 'jade'],
    'black': ['black', 'onyx', 'ebony', 'jet', 'charcoal', 'midnight'],
    'white': ['white', 'ivory', 'cream', 'pearl', 'snow', 'alabaster'],
    'gray': ['gray', 'grey', 'silver', 'slate', 'ash', 'graphite'],
    'yellow': ['yellow', 'gold', 'golden', 'amber', 'lemon', 'mustard'],
    'orange': ['orange', 'tangerine', 'peach', 'coral', 'apricot'],
    'pink': ['pink', 'rose', 'magenta', 'fuchsia', 'salmon', 'blush'],
    'purple': ['purple', 'violet', 'lavender', 'plum', 'mauve', 'lilac'],
    'brown': ['brown', 'tan', 'beige', 'khaki', 'chocolate', 'coffee', 'bronze'],
    'multicolor': ['multicolor', 'multicolour', 'multi-color', 'rainbow', 'mixed']
}

COLOR_VARIANT_TO_BASE = {}
for base_color, variants in COLOR_PATTERNS.items():
    for variant in variants: COLOR_VARIANT_TO_BASE[variant] = base_color

@dataclass
class ProductAttributes:
    base_name: str; colors: Set[str]; sizes: Set[str]; storage: Set[str]; memory: Set[str]; quantities: Set[str]; raw_name: str

def extract_colors(text: str, explicit_color: Optional[str] = None) -> Set[str]:
    colors = set()
    text_lower = str(text).lower() if text else ""
    if explicit_color and pd.notna(explicit_color):
        color_lower = str(explicit_color).lower().strip()
        for variant, base in COLOR_VARIANT_TO_BASE.items():
            if variant in color_lower: colors.add(base)
    for variant, base in COLOR_VARIANT_TO_BASE.items():
        if re.search(r'\b' + re.escape(variant) + r'\b', text_lower): colors.add(base)
    return colors

def remove_attributes(text: str) -> str:
    base = str(text).lower() if text else ""
    for variant in COLOR_VARIANT_TO_BASE.keys(): base = re.sub(r'\b' + re.escape(variant) + r'\b', '', base)
    base = re.sub(r'\b(?:xxs|xs|small|medium|large|xl|xxl|xxxl)\b', '', base)
    base = re.sub(r'\b\d+\s*(?:gb|tb|inch|inches|"|ram|memory|ddr|pack|piece|pcs)\b', '', base)
    for word in ['new', 'original', 'genuine', 'authentic', 'official', 'premium', 'quality', 'best', 'hot', 'sale', 'promo', 'deal']:
        base = re.sub(r'\b' + word + r'\b', '', base)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', base)).strip()

def extract_product_attributes(name: str, explicit_color: Optional[str] = None, brand: Optional[str] = None) -> ProductAttributes:
    name_str = str(name).strip() if pd.notna(name) else ""
    attrs = ProductAttributes(base_name="", colors=extract_colors(name_str, explicit_color), sizes=set(), storage=set(), memory=set(), quantities=set(), raw_name=name_str)
    base_name = remove_attributes(name_str)
    if brand and pd.notna(brand):
        brand_lower = str(brand).lower().strip()
        if brand_lower not in base_name and brand_lower not in ['generic', 'fashion']: base_name = f"{brand_lower} {base_name}"
    attrs.base_name = base_name.strip()
    return attrs

# -------------------------------------------------
# LOCAL EXCEL DATA LOADING HELPERS
# -------------------------------------------------
def load_txt_file(filename: str) -> List[str]:
    try:
        if not os.path.exists(os.path.abspath(filename)): return []
        with open(filename, 'r', encoding='utf-8') as f: return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.warning(f"load_txt_file({filename}): {e}")
        return []

@st.cache_data(ttl=3600)
def load_excel_file(filename: str, column: Optional[str] = None):
    try:
        if not os.path.exists(filename): return [] if column else pd.DataFrame()
        df = pd.read_excel(filename, engine='openpyxl', dtype=str)
        df.columns = df.columns.str.strip()
        if column and column in df.columns: return df[column].apply(clean_category_code).tolist()
        return df
    except Exception as e:
        logger.warning(f"load_excel_file({filename}, col={column}): {e}")
        return [] if column else pd.DataFrame()

def safe_excel_read(filename: str, sheet_name, usecols=None) -> pd.DataFrame:
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=usecols, engine='openpyxl', dtype=str)
        return df.dropna(how='all')
    except Exception as e:
        logger.error(f"safe_excel_read: tab='{sheet_name}' file={filename}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prohibited_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Prohibbited.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    prohibited_by_country = {}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if df.empty:
                prohibited_by_country[tab] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            keyword_col = next((c for c in df.columns if 'keyword' in c or 'prohibited' in c or 'name' in c), df.columns[0])
            category_col = next((c for c in df.columns if 'cat' in c), None)
            country_rules = []
            for _, row in df.iterrows():
                keyword = str(row.get(keyword_col, '')).strip().lower()
                if not keyword or keyword == 'nan' or keyword == 'keywords': continue
                categories = set()
                if category_col:
                    cats_raw = str(row.get(category_col, '')).strip()
                    if cats_raw and cats_raw.lower() != 'nan':
                        split_cats = re.split(r'[,\n]+', cats_raw)
                        categories.update([clean_category_code(c.strip()) for c in split_cats if c.strip()])
                country_rules.append({'keyword': keyword, 'categories': categories})
            prohibited_by_country[tab] = country_rules
        except Exception as e:
            logger.warning(f"load_prohibited_from_local tab={tab}: {e}")
            prohibited_by_country[tab] = []
    return prohibited_by_country

@st.cache_data(ttl=3600)
def load_restricted_brands_from_local() -> Dict[str, List[Dict]]:
    FILE_NAME = "Restricted_Brands.xlsx"
    COUNTRY_TABS = {"Kenya": "KE", "Uganda": "UG", "Nigeria": "NG", "Ghana": "GH", "Morocco": "MA"}
    config_by_country = {}
    for country_name, tab_name in COUNTRY_TABS.items():
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab_name)
            if df.empty:
                config_by_country[country_name] = []
                continue
            df.columns = [str(c).strip().lower() for c in df.columns]
            brand_dict = {}
            for _, row in df.iterrows():
                brand = str(row.get('brand', '')).strip()
                if not brand or brand.lower() == 'nan': continue
                b_lower = brand.lower()
                if b_lower not in brand_dict:
                    brand_dict[b_lower] = {'brand_raw': brand, 'sellers': set(), 'categories': set(), 'variations': set(), 'has_blank_category': False}
                sellers_raw = str(row.get('approved sellers', '')).strip().lower()
                if sellers_raw != 'nan' and sellers_raw:
                    brand_dict[b_lower]['sellers'].update([s.strip() for s in sellers_raw.split(',') if s.strip()])
                cats_raw = str(row.get('categories', '')).strip()
                if cats_raw == 'nan' or not cats_raw:
                    brand_dict[b_lower]['has_blank_category'] = True
                else:
                    brand_dict[b_lower]['categories'].update([clean_category_code(c.strip()) for c in cats_raw.split(',') if c.strip()])
                vars_raw = str(row.get('variations', '')).strip().lower()
                if vars_raw != 'nan' and vars_raw:
                    brand_dict[b_lower]['variations'].update([v.strip() for v in vars_raw.split(',') if v.strip()])
            country_rules = []
            for b_lower, data in brand_dict.items():
                if data['has_blank_category']: data['categories'] = set()
                country_rules.append({'brand': b_lower, 'brand_raw': data['brand_raw'], 'sellers': data['sellers'], 'categories': data['categories'], 'variations': list(data['variations'])})
            config_by_country[country_name] = country_rules
        except Exception as e:
            logger.warning(f"load_restricted_brands tab={tab_name}: {e}")
            config_by_country[country_name] = []
    return config_by_country

@st.cache_data(ttl=3600)
def load_refurb_data_from_local() -> dict:
    FILE_NAME = "Refurb.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "categories": {"Phones": set(), "Laptops": set()}, "keywords": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab, usecols=[0, 1])
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                phones_set  = set(df.iloc[:, 0].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "phones", "phone"}
                laptops_set = set(df.iloc[:, 1].dropna().astype(str).str.strip().str.lower()) - {"", "nan", "laptops", "laptop"}
                result["sellers"][tab] = {"Phones": phones_set, "Laptops": laptops_set}
        except Exception as e:
            logger.warning(f"load_refurb_data tab={tab}: {e}")
            result["sellers"][tab] = {"Phones": set(), "Laptops": set()}
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories", usecols=[0, 1])
        if df_cats.empty: df_cats = safe_excel_read(FILE_NAME, sheet_name="Categries", usecols=[0, 1])
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            result["categories"]["Phones"] = {clean_category_code(c) for c in df_cats.iloc[:, 0].dropna().astype(str) if c.strip() and c.strip().lower() not in ("phones", "phone", "nan")}
            result["categories"]["Laptops"] = {clean_category_code(c) for c in df_cats.iloc[:, 1].dropna().astype(str) if c.strip() and c.strip().lower() not in ("laptops", "laptop", "nan")}
    except Exception as e:
        logger.warning(f"load_refurb_data categories: {e}")
    try:
        df_names = safe_excel_read(FILE_NAME, sheet_name="Name", usecols=[0])
        if not df_names.empty:
            first_col = df_names.columns[0]
            result["keywords"] = {k for k in df_names[first_col].dropna().astype(str).str.strip().str.lower() if k and k not in ("name", "keyword", "keywords", "words", "nan")}
    except Exception as e:
        logger.warning(f"load_refurb_data keywords: {e}")
        result["keywords"] = {"refurb", "refurbished", "renewed"}
    return result

@st.cache_data(ttl=3600)
def load_perfume_data_from_local() -> Dict:
    FILE_NAME = "Perfume.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "keywords": set(), "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                sellers = set(df[seller_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])]))
                result["sellers"][tab] = sellers
        except Exception as e:
            logger.warning(f"load_perfume_data tab={tab}: {e}")
            result["sellers"][tab] = set()
    try:
        df_kw = safe_excel_read(FILE_NAME, sheet_name="Keywords")
        if not df_kw.empty:
            df_kw.columns = [str(c).strip() for c in df_kw.columns]
            kw_col = next((c for c in df_kw.columns if 'brand' in c.lower() or 'keyword' in c.lower()), df_kw.columns[0])
            result["keywords"] = set(df_kw[kw_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "brand", "keyword", "keywords"])]))
    except Exception as e:
        logger.warning(f"load_perfume_data keywords: {e}")
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            result["category_codes"] = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
    except Exception as e:
        logger.warning(f"load_perfume_data categories: {e}")
    return result

@st.cache_data(ttl=3600)
def load_books_data_from_local() -> Dict:
    FILE_NAME = "Books_sellers.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result = {"sellers": {}, "category_codes": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                seller_col = next((c for c in df.columns if 'seller' in c.lower()), df.columns[0])
                result["sellers"][tab] = set(df[seller_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "sellername", "seller name", "seller"])]))
        except Exception as e:
            logger.warning(f"load_books_data tab={tab}: {e}")
            result["sellers"][tab] = set()
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="Categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if 'cat' in c.lower()), df_cats.columns[0])
            result["category_codes"] = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
    except Exception as e:
        logger.warning(f"load_books_data categories: {e}")
    return result

@st.cache_data(ttl=3600)
def load_jerseys_from_local() -> Dict:
    FILE_NAME = "Jersey_validation.xlsx"
    COUNTRY_TABS = ["KE", "UG", "NG", "GH", "MA"]
    result: Dict = {"keywords": {tab: set() for tab in COUNTRY_TABS}, "exempted": {tab: set() for tab in COUNTRY_TABS}, "categories": set()}
    for tab in COUNTRY_TABS:
        try:
            df = safe_excel_read(FILE_NAME, sheet_name=tab)
            if not df.empty:
                df.columns = [str(c).strip() for c in df.columns]
                kw_col = next((c for c in df.columns if "keyword" in c.lower()), df.columns[0])
                result["keywords"][tab] = set(df[kw_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "keywords", "keyword"])]))
                ex_col = next((c for c in df.columns if "exempt" in c.lower() or "seller" in c.lower()), None)
                if ex_col:
                    result["exempted"][tab] = set(df[ex_col].dropna().astype(str).str.strip().str.lower().pipe(lambda s: s[~s.isin(["", "nan", "exempted sellers", "seller"])]))
        except Exception as e:
            logger.warning(f"load_jerseys tab={tab}: {e}")
    try:
        df_cats = safe_excel_read(FILE_NAME, sheet_name="categories")
        if not df_cats.empty:
            df_cats.columns = [str(c).strip().lower() for c in df_cats.columns]
            cat_col = next((c for c in df_cats.columns if "cat" in c), df_cats.columns[0])
            result["categories"] = set(df_cats[cat_col].dropna().astype(str).apply(clean_category_code).pipe(lambda s: s[~s.isin(["", "nan", "categories", "category"])]))
    except Exception as e:
        logger.warning(f"load_jerseys categories: {e}")
    return result

@st.cache_data(ttl=3600)
def load_suspected_fake_from_local() -> Dict:
    """Returns {country_code: DataFrame} for all countries.
    Each sheet uses that country's local currency — no FX conversion needed."""
    country_codes = ["KE", "UG", "NG", "MA", "GH"]
    if not os.path.exists('suspected_fake.xlsx'):
        logger.warning("suspected_fake.xlsx not found")
        return {code: pd.DataFrame() for code in country_codes}
    result = {}
    for code in country_codes:
        try:
            result[code] = pd.read_excel('suspected_fake.xlsx', sheet_name=code, engine='openpyxl', dtype=str)
        except Exception as e:
            logger.warning(f"load_suspected_fake country={code}: {e}")
            result[code] = pd.DataFrame()
    return result

# -------------------------------------------------
# NIGERIA QC RULES LOADER
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_nigeria_qc_rules() -> Dict:
    FILE_NAME = "Nigeria_QC_Rules.xlsx"
    result: Dict = {
        "gift_card":  {"sellers": set(), "category_codes": set()},
        "books":      {},   # book_name_lower -> approved_seller_lower | None (nobody)
        "tvs":        {"category_codes": set(), "brand_sellers": {}},
        "hp_toners":  {"sellers": set(), "category_codes": set()},
        "apple":      {"sellers": set()},
        "xmas_tree":  {"sellers": set(), "keywords": set()},
        "rice":       {},   # brand_lower -> {"sellers": set, "category_codes": set}
        "powerbanks": {"brands": set(), "category_codes": set()},
    }

    if not os.path.exists(FILE_NAME):
        logger.warning("Nigeria_QC_Rules.xlsx not found — NG-specific checks will be skipped.")
        return result

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="Gift card")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col, cat_col = df.columns[0], df.columns[1]
            result["gift_card"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower())
                - {"", "nan", "seller"}
            )
            result["gift_card"]["category_codes"] = (
                set(df[cat_col].dropna().astype(str).apply(clean_category_code))
                - {"", "nan"}
            )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules gift_card: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="Books")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            name_col, seller_col = df.columns[0], df.columns[1]
            for _, row in df.iterrows():
                book = str(row[name_col]).strip().lower()
                if not book or book == "nan":
                    continue
                raw_seller = str(row.get(seller_col, "")).strip().lower()
                result["books"][book] = None if (not raw_seller or raw_seller == "nan") else raw_seller
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules books: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="TVs")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            cat_col    = df.columns[0]
            brand_cols = df.columns[1:]
            result["tvs"]["category_codes"] = (
                set(df[cat_col].dropna().astype(str).apply(clean_category_code))
                - {"", "nan"}
            )
            for bc in brand_cols:
                brand_lower = bc.strip().lower()
                result["tvs"]["brand_sellers"][brand_lower] = (
                    set(df[bc].dropna().astype(str).str.strip().str.lower())
                    - {"", "nan", brand_lower}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules tvs: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="HP ink toners")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            cat_col    = df.columns[1] if len(df.columns) > 1 else None
            result["hp_toners"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower())
                - {"", "nan", "seller"}
            )
            if cat_col:
                result["hp_toners"]["category_codes"] = (
                    set(df[cat_col].dropna().astype(str).apply(clean_category_code))
                    - {"", "nan"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules hp_toners: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="Apple")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            result["apple"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower())
                - {"", "nan", "seller"}
            )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules apple: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="Xmas Tree")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            seller_col = df.columns[0]
            kw_col     = df.columns[1] if len(df.columns) > 1 else None
            result["xmas_tree"]["sellers"] = (
                set(df[seller_col].dropna().astype(str).str.strip().str.lower())
                - {"", "nan", "seller"}
            )
            if kw_col:
                result["xmas_tree"]["keywords"] = (
                    set(df[kw_col].dropna().astype(str).str.strip().str.lower())
                    - {"", "nan", "keyword", "keywords"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules xmas_tree: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="Rice")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            brand_col   = df.columns[0]
            sellers_col = df.columns[1]
            cat_col     = df.columns[2] if len(df.columns) > 2 else None
            for _, row in df.iterrows():
                brand = str(row[brand_col]).strip().lower()
                if not brand or brand == "nan":
                    continue
                raw_sellers = str(row.get(sellers_col, "")).strip()
                sellers = set()
                if raw_sellers and raw_sellers.lower() != "nan":
                    sellers = {s.strip().lower() for s in raw_sellers.split(",") if s.strip()}
                cat_code = ""
                if cat_col:
                    raw_cat = str(row.get(cat_col, "")).strip()
                    if raw_cat and raw_cat.lower() != "nan":
                        cat_code = clean_category_code(raw_cat)
                if brand not in result["rice"]:
                    result["rice"][brand] = {"sellers": set(), "category_codes": set()}
                result["rice"][brand]["sellers"].update(sellers)
                if cat_code:
                    result["rice"][brand]["category_codes"].add(cat_code)
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules rice: {e}")

    try:
        df = safe_excel_read(FILE_NAME, sheet_name="20,000mah Powerbanks")
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            brand_col = df.columns[0]
            cat_col   = df.columns[1] if len(df.columns) > 1 else None
            result["powerbanks"]["brands"] = (
                set(df[brand_col].dropna().astype(str).str.strip().str.lower())
                - {"", "nan", "brand"}
            )
            if cat_col:
                result["powerbanks"]["category_codes"] = (
                    set(df[cat_col].dropna().astype(str).apply(clean_category_code))
                    - {"", "nan"}
                )
    except Exception as e:
        logger.warning(f"load_nigeria_qc_rules powerbanks: {e}")

    return result

# -------------------------------------------------
# LOAD FLAGS MAPPING (WITH MULTI-LINGUAL SUPPORT)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_flags_mapping(filename="reason.xlsx") -> Dict[str, dict]:
    raw_default = {
        'Restricted brands': ('1000024 - Product does not have a license to be sold via Jumia (Not Authorized)', "Missing license for this item. Raise a claim via Vendor Center."),
        'Suspected Fake product': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Product confirmed counterfeit."),
        'Seller Not approved to sell Refurb': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Refurbished approval."),
        'Product Warranty': ('1000013 - Kindly Provide Product Warranty Details', "Valid warranty required in Description/Warranty tabs."),
        'Seller Approve to sell books': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Book category approval."),
        'Seller Approved to Sell Perfume': ('1000028 - Kindly Contact Jumia Seller Support To Confirm Possibility Of Sale Of This Product By Raising A Claim', "Contact Seller Support for Perfume approval."),
        'Counterfeit Sneakers': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Sneaker confirmed counterfeit."),
        'Suspected counterfeit Jerseys': ('1000023 - Confirmation of counterfeit product by Jumia technical team (Not Authorized)', "Jersey confirmed counterfeit."),
        'Prohibited products': ('1000007 - Other Reason', "Listing of this product is prohibited."),
        'Unnecessary words in NAME': ('1000008 - Kindly Improve Product Name Description', "Avoid unnecessary words in title."),
        'Single-word NAME': ('1000008 - Kindly Improve Product Name Description', "Update product title format: Name – Type – Color."),
        'Generic BRAND Issues': ('1000007 - Other Reason', "Use correct brand instead of Generic/Fashion. Apply for brand approval if needed."),
        'Fashion brand issues': ('1000007 - Other Reason', "Use correct brand instead of Fashion. Apply for brand approval if needed."),
        'BRAND name repeated in NAME': ('1000007 - Other Reason', "Brand name should not be repeated in product name."),
        'Generic branded products with genuine brands': ('1000007 - Other Reason', "Use the displayed brand on the product instead of Generic."),
        'Missing COLOR': ('1000005 - Kindly confirm the actual product colour', "Product color must be mentioned in title/color tab."),
        'Duplicate product': ('1000007 - Other Reason', "This product is a duplicate."),
        'Wrong Variation': ('1000039 - Product Poorly Created. Each Variation Of This Product Should Be Created Uniquely (Not Authorized)', "Create different SKUs instead of variations (variations only for sizes)."),
        'Missing Weight/Volume': ('1000008 - Kindly Improve Product Name Description', "Include weight or volume (e.g., '1kg', '500ml')."),
        'Incomplete Smartphone Name': ('1000008 - Kindly Improve Product Name Description', "Include memory/storage details (e.g., '128GB')."),
        'Wrong Category': ('1000004 - Wrong Category', "Assigned to Wrong Category. Please use correct category."),
        'Poor images': ('1000042 - Kindly follow our product image upload guideline.', "Poor Image Quality"),
        'Perfume Tester': ('1000007 - Other Reason', "Sale of perfume testers is not permitted on Jumia."),
        'NG - Gift Card Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell Gift Cards in these categories."),
        'NG - Books Seller':      ('1000003 - Restricted Brand', "Seller not authorised to sell this book title, or no seller is approved for it."),
        'NG - TV Brand Seller':   ('1000003 - Restricted Brand', "Seller not authorised to sell this TV brand."),
        'NG - HP Toners Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell HP Ink/Toners in these categories."),
        'NG - Apple Seller':      ('1000003 - Restricted Brand', "Seller not authorised to sell Apple products."),
        'NG - Xmas Tree Seller':  ('1000003 - Restricted Brand', "Seller not authorised to sell Christmas Tree products."),
        'NG - Rice Brand Seller': ('1000003 - Restricted Brand', "Seller not authorised to sell this rice brand."),
        'NG - Powerbank Capacity':('1000007 - Other Reason',     "Only approved brands may list powerbanks with 20,000mAh or above capacity."),
    }

    default_mapping = {}
    for k, v in raw_default.items():
        default_mapping[k] = {'reason': v[0], 'en': v[1], 'fr': v[1], 'ar': v[1]}

    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename, engine='openpyxl', dtype=str)
            df.columns = df.columns.str.strip().str.lower()
            if 'flag' in df.columns and 'reason' in df.columns and 'comment' in df.columns:
                custom_mapping = {}
                for _, row in df.iterrows():
                    flag = str(row['flag']).strip()
                    reason = str(row['reason']).strip()
                    comment_en = str(row['comment']).strip()
                    comment_fr = str(row['french']).strip() if 'french' in df.columns else comment_en
                    comment_ar = str(row['arabic']).strip() if 'arabic' in df.columns else comment_en
                    if comment_fr.lower() == 'nan' or not comment_fr: comment_fr = comment_en
                    if comment_ar.lower() == 'nan' or not comment_ar: comment_ar = comment_en
                    if flag and flag.lower() != 'nan':
                        custom_mapping[flag] = {'reason': reason, 'en': comment_en, 'fr': comment_fr, 'ar': comment_ar}
                if custom_mapping:
                    ng_keys = {k: v for k, v in default_mapping.items() if k.startswith('NG - ')}
                    merged = {**custom_mapping, **ng_keys}
                    return merged
    except Exception as e:
        logger.warning(f"load_flags_mapping({filename}): {e}")

    return default_mapping

@st.cache_data(ttl=3600)
def load_all_support_files() -> Dict:
    def safe_load_txt(f): return load_txt_file(f) if os.path.exists(f) else []
    support = {
        'blacklisted_words': safe_load_txt('blacklisted.txt'),
        'book_category_codes': safe_load_txt('Books_cat.txt'),
        'books_data': load_books_data_from_local(),
        'perfume_category_codes': safe_load_txt('Perfume_cat.txt'),
        'perfume_data': load_perfume_data_from_local(),
        'sneaker_category_codes': safe_load_txt('Sneakers_Cat.txt'),
        'sneaker_sensitive_brands': [b.lower() for b in safe_load_txt('Sneakers_Sensitive.txt')],
        'sensitive_words': [w.lower() for w in safe_load_txt('sensitive_words.txt')],
        'unnecessary_words': [w.lower() for w in safe_load_txt('unnecessary.txt')],
        'colors': [c.lower() for c in safe_load_txt('colors.txt')],
        'color_categories': safe_load_txt('color_cats.txt'),
        'category_fas': safe_load_txt('Fashion_cat.txt'),
        'reasons': load_excel_file('reasons.xlsx'),
        'flags_mapping': load_flags_mapping(),
        'jerseys_data': load_jerseys_from_local(),
        'warranty_category_codes': safe_load_txt('warranty.txt'),
        'suspected_fake': load_suspected_fake_from_local(),
        'duplicate_exempt_codes': safe_load_txt('duplicate_exempt.txt'),
        'restricted_brands_all': load_restricted_brands_from_local(),
        'prohibited_words_all': load_prohibited_from_local(),
        'known_brands': safe_load_txt('brands.txt'),
        'variation_allowed_codes': safe_load_txt('variation.txt'),
        'weight_category_codes': safe_load_txt('weight.txt'),
        'smartphone_category_codes': safe_load_txt('smartphones.txt'),
        'refurb_data': load_refurb_data_from_local(),
        'category_map': load_category_map(),
        'ng_qc_rules': load_nigeria_qc_rules(),   
    }
    _cat_names: list[str] = []
    _cat_path_to_code: dict[str, str] = {}
    _code_to_path: dict[str, str] = {}

    # Load category_map.xlsx ONCE — builds categories_names_list, cat_path_to_code,
    # and code_to_path all in a single pass.
    _cm_path = "category_map.xlsx"
    try:
        if os.path.exists(_cm_path):
            _cm_df = pd.read_excel(_cm_path, engine="openpyxl", dtype=str)
            _cm_df.columns = [c.strip() for c in _cm_df.columns]

            _path_col = next(
                (c for c in _cm_df.columns if c.lower() == "category path"),
                next((c for c in _cm_df.columns if "path" in c.lower()), None)
            )
            _code_col = next(
                (c for c in _cm_df.columns if "code" in c.lower()), None
            )

            logger.info(
                f"[CategoryMap] Loaded {_cm_path}: {len(_cm_df)} rows, "
                f"path_col={_path_col!r}, code_col={_code_col!r}"
            )

            if _path_col:
                _valid = _cm_df[_path_col].dropna().astype(str)
                _valid = _valid[_valid.str.strip().ne("")]
                _cat_names = _valid.tolist()

                if _code_col:
                    for _, _row in _cm_df[[_path_col, _code_col]].dropna().iterrows():
                        _p = str(_row[_path_col]).strip()
                        _c = str(_row[_code_col]).strip().split(".")[0]
                        if _p and _c:
                            _cat_path_to_code[_p.lower()] = _c  # lowercase for lookups
                            _code_to_path[_c] = _p               # original case

                logger.info(
                    f"[CategoryMap] Built: {len(_cat_names)} category paths, "
                    f"{len(_code_to_path)} code->path entries"
                )
            else:
                logger.warning(
                    f"[CategoryMap] No 'Category Path' column found in {_cm_path}. "
                    f"Columns: {list(_cm_df.columns)}"
                )
        else:
            logger.warning(
                f"[CategoryMap] {_cm_path} not found — wrong-category detection "
                f"will be skipped. Place category_map.xlsx in the app root directory."
            )
    except Exception as _ce:
        logger.error(f"[CategoryMap] Failed to load {_cm_path}: {_ce}")

    support['categories_names_list'] = _cat_names
    support['cat_path_to_code'] = _cat_path_to_code
    support['code_to_path'] = _code_to_path

    logger.info(
        f"[CategoryMap] Final: categories_names_list={len(_cat_names)}, "
        f"cat_path_to_code={len(_cat_path_to_code)}, code_to_path={len(_code_to_path)}"
    )

    # Compile JSON boost rules now that code_to_path is available
    support['compiled_json_rules'] = load_and_compile_json_rules("category_qc_weighted.json")
    return support

@st.cache_data(ttl=3600)
def load_support_files_lazy(): return load_all_support_files()

@st.cache_resource(ttl=3600)
def load_and_compile_json_rules(json_path="category_qc_weighted.json") -> dict:
    if not os.path.exists(json_path):
        logger.warning(f"{json_path} not found. Running without JSON boosts.")
        return {}
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_rules = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load JSON rules: {e}")
        return {}

    if isinstance(raw_rules, list):
        fixed_rules = {}
        for item in raw_rules:
            if isinstance(item, dict):
                cat = (item.get("category") or item.get("Category Path")
                       or item.get("name") or item.get("category_name"))
                kws = (item.get("keywords") or item.get("weights")
                       or item.get("positive"))
                if cat and isinstance(kws, dict):
                    fixed_rules[cat] = kws
        raw_rules = fixed_rules

    if not isinstance(raw_rules, dict):
        logger.warning("JSON rules file has an unrecognizable format. Must be a dict.")
        return {}

    compiled_rules = {}
    for cat_path, keywords_dict in raw_rules.items():
        if not isinstance(keywords_dict, dict) or not keywords_dict:
            continue
            
        try:
            safe_kws = {str(k): float(w) for k, w in keywords_dict.items()}
            sorted_kws = sorted(safe_kws.keys(), key=len, reverse=True)
            if not sorted_kws:
                continue
                
            pattern_str = r'\b(' + '|'.join(re.escape(k) for k in sorted_kws) + r')\b'
            
            compiled_rules[str(cat_path)] = {
                'pattern': re.compile(pattern_str, re.IGNORECASE),
                'weights': {k.lower(): w for k, w in safe_kws.items()}
            }
        except Exception as e:
            logger.warning(f"Skipping bad JSON rule block for {cat_path}: {e}")
            continue
            
    return compiled_rules

@st.cache_data(ttl=3600)
def compile_regex_patterns(words: List[str]) -> re.Pattern:
    if not words: return None
    pattern = '|'.join(r'\b' + re.escape(w) + r'\b' for w in sorted(words, key=len, reverse=True))
    return re.compile(pattern, re.IGNORECASE)

class CountryValidator:
    COUNTRY_CONFIG = {
        "Kenya": {"code": "KE", "skip_validations": []},
        "Uganda": {"code": "UG", "skip_validations": ["Counterfeit Sneakers", "Product Warranty", "Generic BRAND Issues"]},
        "Nigeria": {"code": "NG", "skip_validations": []},
        "Ghana": {"code": "GH", "skip_validations": []},
        "Morocco": {"code": "MA", "skip_validations": []}
    }

    def __init__(self, country: str):
        self.country = country
        self.config = self.COUNTRY_CONFIG.get(country, self.COUNTRY_CONFIG["Kenya"])
        self.code = self.config["code"]
        self.skip_validations = self.config["skip_validations"]

    def should_skip_validation(self, validation_name: str) -> bool: return validation_name in self.skip_validations
    def ensure_status_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and 'Status' not in df.columns: df['Status'] = 'Approved'
        return df

# -------------------------------------------------
# ENCODING HELPERS
# -------------------------------------------------
def _detect_and_read_csv(buf) -> pd.DataFrame:
    _ENCODINGS = ['utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']
    raw_bytes = buf.read()

    for enc in _ENCODINGS:
        for sep in [',', ';', '\t']:
            try:
                from io import BytesIO as _BIO
                df = pd.read_csv(_BIO(raw_bytes), sep=sep, encoding=enc, dtype=str)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue

    from io import BytesIO as _BIO
    return pd.read_csv(_BIO(raw_bytes), sep=None, engine='python', encoding='utf-8', dtype=str)


def _repair_mojibake(df: pd.DataFrame) -> pd.DataFrame:
    import re as _re
    _ILLEGAL_XML = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')

    def _fix(val):
        if not isinstance(val, str):
            return val
        for enc in ('cp1252', 'latin-1'):
            try:
                fixed = val.encode(enc).decode('utf-8')
                if fixed != val and '\ufffd' not in fixed:
                    val = fixed
                    break
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        return _ILLEGAL_XML.sub('', val)

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(_fix)
    return df

def standardize_input_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    map_lower = {k.lower(): v for k, v in NEW_FILE_MAPPING.items()}
    renamed = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in map_lower: renamed[col] = map_lower[col_lower]
        else: renamed[col] = col.upper()
    df = df.rename(columns=renamed)
    for col in ['ACTIVE_STATUS_COUNTRY', 'CATEGORY_CODE', 'BRAND', 'TAX_CLASS', 'NAME', 'SELLER_NAME']:
        if col in df.columns: df[col] = df[col].astype(str)
    if 'MAIN_IMAGE' not in df.columns: df['MAIN_IMAGE'] = ''
    return df

def validate_input_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = [f"Missing: {f}" for f in ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY_CODE', 'ACTIVE_STATUS_COUNTRY'] if f not in df.columns]
    return len(errors) == 0, errors

MULTI_COUNTRY_VALUES = {'MULTIPLE', 'MULTI'}

def filter_by_country(df: pd.DataFrame, country_validator: CountryValidator) -> Tuple[pd.DataFrame, List[str]]:
    if 'ACTIVE_STATUS_COUNTRY' not in df.columns: return df, []
    s = df['ACTIVE_STATUS_COUNTRY'].astype(str).str.strip().str.upper().str.replace(r'^JUMIA-', '', regex=True)
    df['ACTIVE_STATUS_COUNTRY'] = s
    if country_validator.code == 'NG':
        is_ng = df['ACTIVE_STATUS_COUNTRY'] == 'NG'
        is_multi = df['ACTIVE_STATUS_COUNTRY'].isin(MULTI_COUNTRY_VALUES)
        filtered = df[is_ng | is_multi].copy()
        filtered['_IS_MULTI_COUNTRY'] = is_multi[filtered.index]
    else:
        filtered = df[df['ACTIVE_STATUS_COUNTRY'] == country_validator.code].copy()
        filtered['_IS_MULTI_COUNTRY'] = False
    detected_names = []
    if filtered.empty:
        detected_codes = [c for c in df['ACTIVE_STATUS_COUNTRY'].unique() if str(c).strip() and str(c).strip().lower() != 'nan']
        emoji_map = {"KE": "Kenya", "UG": "Uganda", "NG": "Nigeria", "GH": "Ghana", "MA": "Morocco"}
        detected_names = [emoji_map.get(c, f"'{c}'") for c in detected_codes]
    return filtered, detected_names

def propagate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ['COLOR_FAMILY', 'PRODUCT_WARRANTY', 'WARRANTY_DURATION', 'WARRANTY_ADDRESS', 'WARRANTY_TYPE', 'COUNT_VARIATIONS', 'LIST_VARIATIONS']:
        if col not in df.columns: df[col] = pd.NA
        df[col] = df.groupby('PRODUCT_SET_SID')[col].transform(lambda x: x.ffill().bfill())
    return df

# -------------------------------------------------
# CACHE-AWARE VALIDATION CHECKS
# -------------------------------------------------
FLAG_RELEVANT_COLS = {
    "Wrong Category": ["NAME", "CATEGORY", "CATEGORY_CODE"],
    "Restricted brands": ["NAME", "BRAND", "SELLER_NAME", "CATEGORY_CODE"],
    "Suspected Fake product": ["CATEGORY_CODE", "BRAND", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE"],
    "Seller Not approved to sell Refurb": ["PRODUCT_SET_SID", "CATEGORY_CODE", "SELLER_NAME", "NAME"],
    "Product Warranty": ["PRODUCT_WARRANTY", "WARRANTY_DURATION", "CATEGORY_CODE"],
    "Seller Approve to sell books": ["CATEGORY_CODE", "SELLER_NAME"],
    "Seller Approved to Sell Perfume": ["CATEGORY_CODE", "SELLER_NAME", "BRAND", "NAME"],
    "Counterfeit Sneakers": ["CATEGORY_CODE", "NAME", "BRAND"],
    "Suspected counterfeit Jerseys": ["CATEGORY_CODE", "NAME", "SELLER_NAME"],
    "Unnecessary words in NAME": ["NAME"],
    "Single-word NAME": ["CATEGORY_CODE", "NAME"],
    "Generic BRAND Issues": ["CATEGORY_CODE", "BRAND"],
    "Fashion brand issues": ["CATEGORY_CODE", "BRAND"],
    "BRAND name repeated in NAME": ["BRAND", "NAME"],
    "Wrong Variation": ["COUNT_VARIATIONS", "CATEGORY_CODE"],
    "Generic branded products with genuine brands": ["NAME", "BRAND", "CATEGORY"],
    "Missing COLOR": ["CATEGORY_CODE", "NAME", "COLOR"],
    "Missing Weight/Volume": ["CATEGORY_CODE", "NAME"],
    "Incomplete Smartphone Name": ["CATEGORY_CODE", "NAME"],
    "Duplicate product": ["NAME", "SELLER_NAME", "BRAND", "CATEGORY_CODE"],
    "Perfume Tester": ["CATEGORY_CODE", "NAME"],
    "NG - Gift Card Seller":  ["CATEGORY_CODE", "SELLER_NAME"],
    "NG - Books Seller":      ["NAME", "SELLER_NAME"],
    "NG - TV Brand Seller":   ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "NG - HP Toners Seller":  ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "NG - Apple Seller":      ["BRAND", "SELLER_NAME"],
    "NG - Xmas Tree Seller":  ["NAME", "SELLER_NAME"],
    "NG - Rice Brand Seller": ["CATEGORY_CODE", "BRAND", "SELLER_NAME"],
    "NG - Powerbank Capacity":["CATEGORY_CODE", "NAME", "BRAND"],
}

def compute_flag_input_hash(data: pd.DataFrame, flag_name: str, kwargs: dict) -> str:
    cols = FLAG_RELEVANT_COLS.get(flag_name, data.columns.tolist())
    available_cols = [c for c in cols if c in data.columns]
    if not available_cols: return "empty"
    df_hash_str = df_hash(data[available_cols])
    kwargs_repr = ""
    _skip_keys = {'categories_list', 'cat_path_to_code', 'code_to_path'}
    for k, v in kwargs.items():
        if k == 'data' or k in _skip_keys: continue
        if isinstance(v, pd.DataFrame): kwargs_repr += df_hash(v)
        else: kwargs_repr += repr(v)
    return hashlib.md5((df_hash_str + kwargs_repr).encode()).hexdigest()

def run_cached_check(func, cache_path, ckwargs):
    if func is check_miscellaneous_category:
        return func(**ckwargs)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f: return pickle.load(f)
        except Exception as e:
            logger.warning(f"run_cached_check load failed {cache_path}: {e}")
    res = func(**ckwargs)
    try:
        with open(cache_path, 'wb') as f: pickle.dump(res, f)
    except Exception as e:
        logger.warning(f"run_cached_check save failed {cache_path}: {e}")
    return res

def check_miscellaneous_category(
    data: pd.DataFrame,
    categories_list: list = None,
    compiled_rules: dict = None,
    cat_path_to_code: dict = None,
    code_to_path: dict = None,
) -> pd.DataFrame:
    """
    Batch-vectorised wrong-category detection.

    When CategoryMatcherEngine exposes predict_batch(), all product names are
    scored in a single TF-IDF transform instead of one transform per row.
    Falls back to the original per-row path for engines that don't yet have
    predict_batch().
    """
    if categories_list is None:
        categories_list = []
    if compiled_rules is None:
        compiled_rules = st.session_state.get('compiled_json_rules', {})
    if cat_path_to_code is None:
        cat_path_to_code = {}
    if code_to_path is None:
        code_to_path = {}

    # Pull from session state if callers omitted them
    if not categories_list or not code_to_path:
        try:
            _sf = st.session_state.get("support_files", {})
            categories_list  = categories_list  or _sf.get("categories_names_list", [])
            cat_path_to_code = cat_path_to_code or _sf.get("cat_path_to_code", {})
            code_to_path     = code_to_path     or _sf.get("code_to_path", {})
        except Exception:
            pass

    if not _CAT_MATCHER_AVAILABLE:
        # Minimal fallback — flag only "Miscellaneous" names
        if 'CATEGORY' not in data.columns:
            return pd.DataFrame(columns=data.columns)
        flagged = data[
            data['CATEGORY'].astype(str).str.contains("miscellaneous", case=False, na=False)
        ].copy()
        if not flagged.empty:
            flagged['Comment_Detail'] = "Category contains 'Miscellaneous'"
        return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

    try:
        _engine = _get_cat_matcher_engine()
        if _engine is None:
            return pd.DataFrame(columns=data.columns)

        # Build TF-IDF index once
        if categories_list and not _engine._tfidf_built:
            _engine.build_tfidf_index(categories_list)

        _engine.set_compiled_rules(compiled_rules)

        # ── Batch path (fast) ─────────────────────────────────────────────
        if hasattr(_engine, 'predict_batch') and callable(_engine.predict_batch):
            names_series  = data['NAME'].astype(str).str.strip().fillna('')
            unique_names  = names_series.unique().tolist()

            # Single TF-IDF transform across all unique names
            batch_results = _engine.predict_batch(unique_names)

            # Normalise to {name: (predicted_path, confidence)}
            if isinstance(batch_results, dict):
                name_to_result = batch_results
            else:
                name_to_result = dict(zip(unique_names, batch_results))

            _data = data.copy()
            _data['_pred_path'] = names_series.map(name_to_result).apply(
                lambda r: r[0] if isinstance(r, (list, tuple)) and r else (r or '')
            )
            _data['_pred_conf'] = names_series.map(name_to_result).apply(
                lambda r: float(r[1]) if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0
            )
            _data['_actual_path'] = _data['CATEGORY_CODE'].apply(
                lambda c: code_to_path.get(str(c).strip(), '') if pd.notna(c) else ''
            )

            def _top_domain(path: str) -> str:
                if not path:
                    return ''
                return str(path).split('>')[0].split('/')[0].strip().lower()

            pred_domain   = _data['_pred_path'].apply(_top_domain)
            actual_domain = _data['_actual_path'].apply(_top_domain)
            conf_thr      = st.session_state.get('cat_conf_threshold', 0.0)

            mismatch = (
                pred_domain.ne('')
                & actual_domain.ne('')
                & pred_domain.ne(actual_domain)
                & (_data['_pred_conf'] >= conf_thr)
            )

            flagged = _data[mismatch].copy()
            if not flagged.empty:
                flagged['Comment_Detail'] = (
                    "Predicted: " + flagged['_pred_path'].str[:60]
                    + " (conf: " + flagged['_pred_conf'].round(2).astype(str) + ")"
                )
                # Extra columns visible in the flag expander table
                flagged['Suggested_Category'] = flagged['_pred_path']
                flagged['Confidence']         = flagged['_pred_conf'].round(3)

            flagged.drop(
                columns=[c for c in ['_pred_path', '_pred_conf', '_actual_path']
                          if c in flagged.columns],
                inplace=True, errors='ignore'
            )
            return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

        # ── Per-row fallback (original behaviour) ────────────────────────
        return check_wrong_category(
            data, categories_list,
            compiled_rules=compiled_rules,
            cat_path_to_code=cat_path_to_code,
            code_to_path=code_to_path,
        )

    except Exception as _e:
        logger.warning("check_miscellaneous_category engine error: %s", _e)
        return pd.DataFrame(columns=data.columns)


def check_restricted_brands(data: pd.DataFrame, country_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME', 'BRAND', 'SELLER_NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not country_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_name_lower'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_brand_lower'] = d['BRAND'].astype(str).str.lower().str.strip().fillna('')
    d['_seller_lower'] = d['SELLER_NAME'].astype(str).str.lower().str.strip().fillna('')
    d['_cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    flagged_indices = set()
    comment_map = {}
    match_details = {}
    for rule in country_rules:
        brand_name = rule['brand']
        brand_raw = rule['brand_raw']
        brand_pattern = r'(?<!\w)' + re.escape(brand_name) + r'(?!\w)'
        main_brand_matches = (d['_brand_lower'] == brand_name)
        main_name_matches = d['_name_lower'].str.contains(brand_pattern, regex=True, na=False)
        current_match_mask = main_brand_matches | main_name_matches
        for idx in d[main_brand_matches].index: match_details[idx] = ('main_brand', brand_raw)
        for idx in d[main_name_matches & ~main_brand_matches].index: match_details[idx] = ('main_name', brand_raw)
        if rule['variations']:
            sorted_vars = sorted(rule['variations'], key=len, reverse=True)
            var_pattern = r'(?<!\w)(' + '|'.join([re.escape(v) for v in sorted_vars]) + r')(?!\w)'
            var_brand_matches = d['_brand_lower'].str.contains(var_pattern, regex=True, na=False)
            var_name_matches = d['_name_lower'].str.contains(var_pattern, regex=True, na=False)
            for idx in d[var_brand_matches | var_name_matches].index:
                if idx not in match_details:
                    text_to_check = d.loc[idx, '_brand_lower'] if var_brand_matches[idx] else d.loc[idx, '_name_lower']
                    for var in sorted_vars:
                        if var in text_to_check:
                            match_details[idx] = ('variation', f"{brand_raw} (as '{var}')")
                            break
            current_match_mask = current_match_mask | var_brand_matches | var_name_matches
        if not current_match_mask.any(): continue
        current_match = d[current_match_mask]
        if rule['categories']: current_match = current_match[current_match['_cat_clean'].isin(rule['categories'])]
        if current_match.empty: continue
        rejected = current_match[~current_match['_seller_lower'].isin(rule['sellers'])]
        if not rejected.empty:
            for idx in rejected.index:
                flagged_indices.add(idx)
                match_type, match_info = match_details.get(idx, ('unknown', brand_raw))
                seller_status = "Seller not in approved list" if rule['sellers'] else "No sellers approved"
                comment_map[idx] = f"Restricted Brand: {match_info} - {seller_status}"
    if not flagged_indices: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_indices)].copy()
    result['Comment_Detail'] = result.index.map(comment_map)
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_prohibited_products(data: pd.DataFrame, prohibited_rules: List[Dict]) -> pd.DataFrame:
    if not {'NAME', 'CATEGORY_CODE'}.issubset(data.columns) or not prohibited_rules: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d['_name_lower'] = d['NAME'].astype(str).str.lower().fillna('')
    d['_cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    flagged_indices = set()
    comment_map = {}
    name_replacements = {}
    for rule in prohibited_rules:
        keyword = rule['keyword']
        target_cats = rule['categories']
        pattern = re.compile(r'(?<!\w)' + re.escape(keyword) + r'(?!\w)', re.IGNORECASE)
        match_mask = d['_name_lower'].str.contains(pattern, regex=True, na=False)
        if not match_mask.any(): continue
        current_match = d[match_mask]
        if target_cats: current_match = current_match[current_match['_cat_clean'].isin(target_cats)]
        if current_match.empty: continue
        for idx in current_match.index:
            flagged_indices.add(idx)
            existing_comment = comment_map.get(idx, "Prohibited:")
            if keyword not in existing_comment: comment_map[idx] = f"{existing_comment} {keyword},"
            raw_name = str(d.loc[idx, 'NAME'])
            highlighted = pattern.sub(lambda m: f"[!]{m.group(0)}[!]", raw_name)
            name_replacements[idx] = highlighted
    if not flagged_indices: return pd.DataFrame(columns=data.columns)
    result = data.loc[list(flagged_indices)].copy()
    result['Comment_Detail'] = result.index.map(lambda i: comment_map[i].rstrip(','))
    for idx, new_name in name_replacements.items(): result.loc[idx, 'NAME'] = new_name
    return result.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_suspected_fake_products(data: pd.DataFrame, suspected_fake_df: pd.DataFrame) -> pd.DataFrame:
    if not all(c in data.columns for c in ['CATEGORY_CODE', 'BRAND', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE']) or suspected_fake_df.empty:
        return pd.DataFrame(columns=data.columns)
    try:
        ref_data = suspected_fake_df.copy()
        brand_cat_price = {}
        for brand in [c for c in ref_data.columns if c not in ['Unnamed: 0', 'Brand', 'Price'] and pd.notna(c)]:
            try:
                pt = pd.to_numeric(ref_data[brand].iloc[0], errors='coerce')
                if pd.isna(pt) or pt <= 0: continue
            except: continue
            for cat in ref_data[brand].iloc[1:].dropna():
                cat_base = str(cat).strip().split('.')[0]
                if cat_base and cat_base.lower() != 'nan': brand_cat_price[(brand.strip().lower(), cat_base)] = pt
        if not brand_cat_price: return pd.DataFrame(columns=data.columns)
        d = data.copy()
        d['price_to_use'] = pd.to_numeric(d['GLOBAL_SALE_PRICE'].where(d['GLOBAL_SALE_PRICE'].notna() & (pd.to_numeric(d['GLOBAL_SALE_PRICE'], errors='coerce') > 0), d['GLOBAL_PRICE']), errors='coerce').fillna(0)
        d['BRAND_LOWER'] = d['BRAND'].astype(str).str.strip().str.lower()
        d['CAT_BASE'] = d['CATEGORY_CODE'].apply(clean_category_code)
        prices = d['price_to_use'].values
        brands = d['BRAND_LOWER'].values
        cats = d['CAT_BASE'].values
        d['is_fake'] = [p < brand_cat_price.get((b, c), -1) for p, b, c in zip(prices, brands, cats)]
        return d[d['is_fake'] == True][data.columns].drop_duplicates(subset=['PRODUCT_SET_SID'])
    except Exception as e:
        logger.warning(f"check_suspected_fake_products: {e}")
        return pd.DataFrame(columns=data.columns)

def check_refurb_seller_approval(data: pd.DataFrame, refurb_data: dict, country_code: str) -> pd.DataFrame:
    required = {'PRODUCT_SET_SID', 'CATEGORY_CODE', 'SELLER_NAME', 'NAME'}
    if not required.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    phone_cats = refurb_data.get("categories", {}).get("Phones", set())
    laptop_cats = refurb_data.get("categories", {}).get("Laptops", set())
    keywords = refurb_data.get("keywords", set())
    sellers = refurb_data.get("sellers", {}).get(country_code, {})
    if not phone_cats and not laptop_cats: return pd.DataFrame(columns=data.columns)
    if not keywords: return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
    d = data.copy()
    d['_cat'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['_seller'] = d['SELLER_NAME'].astype(str).str.strip().str.lower()
    d['_name'] = d['NAME'].astype(str).str.strip()
    is_phone = d['_cat'].isin(phone_cats)
    is_laptop = d['_cat'].isin(laptop_cats)
    in_scope = is_phone | is_laptop
    has_keyword = d['_name'].str.contains(kw_pattern, na=False)
    approved_phones = sellers.get("Phones", set())
    approved_laptops = sellers.get("Laptops", set())
    not_approved = ((is_phone & ~d['_seller'].isin(approved_phones)) | (is_laptop & ~d['_seller'].isin(approved_laptops)))
    flagged = d[in_scope & has_keyword & not_approved].copy()
    if not flagged.empty:
        def build_comment(row):
            ptype = "Phone" if row['_cat'] in phone_cats else "Laptop"
            match = kw_pattern.search(row['_name'])
            kw_found = match.group(0) if match else "?"
            return f"Unapproved {ptype} refurb seller — keyword '{kw_found}' in name (cat: {row['_cat']})"
        flagged['Comment_Detail'] = flagged.apply(build_comment, axis=1)
    flagged = flagged.drop(columns=['_cat', '_seller', '_name'], errors='ignore')
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_product_warranty(data: pd.DataFrame, warranty_category_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']:
        if c not in d.columns: d[c] = ""
        d[c] = d[c].astype(str).fillna('').str.strip()
    if not warranty_category_codes: return pd.DataFrame(columns=d.columns)
    d['CAT_CLEAN'] = d['CATEGORY_CODE'].apply(clean_category_code)
    target = d[d['CAT_CLEAN'].isin([clean_category_code(c) for c in warranty_category_codes])]
    if target.empty: return pd.DataFrame(columns=d.columns)
    def is_present(s): return (s != 'nan') & (s != '') & (s != 'none') & (s != 'nat') & (s != 'n/a')
    mask = ~(is_present(target['PRODUCT_WARRANTY']) | is_present(target['WARRANTY_DURATION']))
    return target[mask].drop(columns=['CAT_CLEAN'], errors='ignore').drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_books(data: pd.DataFrame, books_data: Dict, country_code: str, book_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    category_codes = books_data.get('category_codes') or set(clean_category_code(c) for c in book_category_codes)
    if not category_codes: return pd.DataFrame(columns=data.columns)
    approved_sellers = books_data.get('sellers', {}).get(country_code, set())
    if not approved_sellers: return pd.DataFrame(columns=data.columns)
    books = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(category_codes)].copy()
    if books.empty: return pd.DataFrame(columns=data.columns)
    not_approved = ~books['SELLER_NAME'].astype(str).str.strip().str.lower().isin(approved_sellers)
    flagged = books[not_approved].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Seller not approved to sell books: " + flagged['SELLER_NAME'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_seller_approved_for_perfume(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict, country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'SELLER_NAME', 'BRAND', 'NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sheet_cat_codes = perfume_data.get('category_codes')
    cat_codes = sheet_cat_codes if sheet_cat_codes else set(clean_category_code(c) for c in perfume_category_codes)
    perfume = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)].copy()
    if perfume.empty: return pd.DataFrame(columns=data.columns)
    keywords = perfume_data.get('keywords', set())
    approved_sellers = perfume_data.get('sellers', {}).get(country_code, set())
    has_seller_list = bool(approved_sellers)
    b_lower = perfume['BRAND'].astype(str).str.strip().str.lower()
    n_lower = perfume['NAME'].astype(str).str.strip().str.lower()
    GENERIC_PLACEHOLDERS = {'designers collection', 'smart collection', 'generic', 'original', 'fashion'}
    if keywords:
        kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r')\b', re.IGNORECASE)
        sneaky_mask = b_lower.isin(GENERIC_PLACEHOLDERS) & n_lower.apply(lambda x: bool(kw_pattern.search(x)))
    else: sneaky_mask = pd.Series([False] * len(perfume), index=perfume.index)
    if has_seller_list:
        brand_sens_mask = b_lower.apply(lambda x: bool(kw_pattern.search(x))) if keywords else pd.Series([False]*len(perfume), index=perfume.index)
        needs_approval = sneaky_mask | brand_sens_mask
        not_approved = ~perfume['SELLER_NAME'].astype(str).str.strip().str.lower().isin(approved_sellers)
        flagged_mask = needs_approval & not_approved
    else: flagged_mask = sneaky_mask
    flagged = perfume[flagged_mask].copy()
    if not flagged.empty:
        def describe(row):
            b, n = str(row['BRAND']).strip(), str(row['NAME']).strip()[:40]
            if b.lower() in GENERIC_PLACEHOLDERS: return f"Sneaky brand in name: '{n}'"
            return f"Sensitive brand '{b}' — seller not approved"
        flagged['Comment_Detail'] = flagged.apply(describe, axis=1)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_perfume_tester(data: pd.DataFrame, perfume_category_codes: List[str], perfume_data: Dict) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)
    sheet_cat_codes = perfume_data.get('category_codes')
    cat_codes = sheet_cat_codes if sheet_cat_codes else set(clean_category_code(c) for c in perfume_category_codes)
    if not cat_codes:
        return pd.DataFrame(columns=data.columns)
    perfume = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)].copy()
    if perfume.empty:
        return pd.DataFrame(columns=data.columns)
    tester_pattern = re.compile(r'\btester\b', re.IGNORECASE)
    flagged = perfume[perfume['NAME'].astype(str).str.contains(tester_pattern, na=False)].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Perfume tester listed for sale: " + flagged['NAME'].astype(str).str[:60]
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_sneakers(data: pd.DataFrame, sneaker_category_codes: List[str], sneaker_sensitive_brands: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    sneakers = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in sneaker_category_codes))].copy()
    if sneakers.empty: return pd.DataFrame(columns=data.columns)
    b_lower, n_lower = sneakers['BRAND'].astype(str).str.strip().str.lower(), sneakers['NAME'].astype(str).str.strip().str.lower()
    return sneakers[b_lower.isin(['generic', 'fashion']) & n_lower.apply(lambda x: any(b in x for b in sneaker_sensitive_brands))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_counterfeit_jerseys(data: pd.DataFrame, jerseys_data: Dict, country_code: str) -> pd.DataFrame:
    if not {"CATEGORY_CODE", "NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    categories = jerseys_data.get("categories", set())
    keywords   = jerseys_data.get("keywords",   {}).get(country_code, set())
    exempted   = jerseys_data.get("exempted",   {}).get(country_code, set())
    if not categories or not keywords: return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    d = data.copy()
    d["_cat"]    = d["CATEGORY_CODE"].apply(clean_category_code)
    d["_seller"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    d["_name"]   = d["NAME"].astype(str).str.strip()
    in_scope     = d["_cat"].isin(categories)
    has_keyword  = d["_name"].str.contains(kw_pattern, na=False)
    not_exempted = ~d["_seller"].isin(exempted)
    flagged = d[in_scope & has_keyword & not_exempted].copy()
    if not flagged.empty:
        def build_comment(row):
            match = kw_pattern.search(row["_name"])
            kw_found = match.group(0) if match else "?"
            return f"Suspected counterfeit jersey — keyword '{kw_found}' (cat: {row['_cat']})"
        flagged["Comment_Detail"] = flagged.apply(build_comment, axis=1)
    return flagged.drop(columns=["_cat", "_seller", "_name"], errors="ignore").drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_unnecessary_words(data: pd.DataFrame, pattern: re.Pattern) -> pd.DataFrame:
    if not {'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    d = data.copy()
    mask = d['NAME'].astype(str).str.strip().str.lower().str.contains(pattern, na=False)
    flagged = d[mask].copy()
    if not flagged.empty:
        def get_matches(text):
            if pd.isna(text): return ""
            matches = pattern.findall(str(text))
            return ", ".join(set(m.lower() for m in matches if isinstance(m, str)))
        def highlight_matches(text):
            if pd.isna(text): return text
            return pattern.sub(lambda m: f"[*]{m.group(0)}[*]", str(text))
        flagged['Comment_Detail'] = "Unnecessary: " + flagged['NAME'].apply(get_matches)
        flagged['NAME'] = flagged['NAME'].apply(highlight_matches)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_single_word_name(
    data: pd.DataFrame,
    book_category_codes: List[str],
    books_data: Dict = None,
) -> pd.DataFrame:
    """Flag products whose NAME has 1 or 2 words (excluding book categories)."""
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)
    cat_codes = (books_data or {}).get('category_codes') or set(
        clean_category_code(c) for c in book_category_codes
    )
    non_books   = data[~data['CATEGORY_CODE'].apply(clean_category_code).isin(cat_codes)]
    word_counts = non_books['NAME'].astype(str).str.split().str.len()
    flagged     = non_books[word_counts <= 2].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = (
            "Name too short ("
            + word_counts[flagged.index].astype(str)
            + " word(s)) — use format: Name – Type – Colour"
        )
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE','BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    return data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in valid_category_codes_fas)) & (data['BRAND'].astype(str).str.lower() == 'generic')].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_fashion_brand_issues(data: pd.DataFrame, valid_category_codes_fas: List[str], code_to_path: Dict = None) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    if code_to_path is None:
        code_to_path = {}
    fashion_brand = data[data['BRAND'].astype(str).str.strip().str.lower() == 'fashion'].copy()
    if fashion_brand.empty:
        return pd.DataFrame(columns=data.columns)
    def _in_fashion_domain(cat_code: str) -> bool:
        full_path = code_to_path.get(str(cat_code).strip(), '')
        if full_path:
            return full_path.strip().lower().startswith('fashion')
        return clean_category_code(cat_code) in fas_codes
    fas_codes = set(clean_category_code(c) for c in valid_category_codes_fas)
    flagged = fashion_brand[
        ~fashion_brand['CATEGORY_CODE'].apply(lambda c: _in_fashion_domain(clean_category_code(c)))
    ].copy()
    if not flagged.empty:
        flagged['Comment_Detail'] = "Brand 'Fashion' used outside Fashion category: " + flagged['CATEGORY_CODE'].astype(str)
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_brand_in_name(data: pd.DataFrame) -> pd.DataFrame:
    if not {'BRAND','NAME'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    brands = data['BRAND'].astype(str).str.strip().str.lower().values
    names = data['NAME'].astype(str).str.strip().str.lower().values
    mask = [b in n if b and b != 'nan' else False for b, n in zip(brands, names)]
    return data[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_wrong_variation(data: pd.DataFrame, allowed_variation_codes: List[str]) -> pd.DataFrame:
    d = data.copy()
    if 'COUNT_VARIATIONS' not in d.columns: d['COUNT_VARIATIONS'] = 1
    if 'CATEGORY_CODE' not in d.columns: return pd.DataFrame(columns=data.columns)
    d['cat_clean'] = d['CATEGORY_CODE'].apply(clean_category_code)
    d['qty_var'] = pd.to_numeric(d['COUNT_VARIATIONS'], errors='coerce').fillna(1).astype(int)
    flagged = d[(d['qty_var'] >= 3) & (~d['cat_clean'].isin(set(clean_category_code(c) for c in allowed_variation_codes)))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Variations: " + flagged['qty_var'].astype(str) + ", Category: " + flagged['cat_clean']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_generic_with_brand_in_name(data: pd.DataFrame, brands_list: List[str]) -> pd.DataFrame:
    if not {'NAME', 'BRAND'}.issubset(data.columns) or not brands_list: return pd.DataFrame(columns=data.columns)
    mask = (data['BRAND'].astype(str).str.strip().str.lower() == 'generic')
    if 'CATEGORY' in data.columns: mask = mask & ~data['CATEGORY'].astype(str).str.lower().str.contains(r'\b(case|cases|cover|covers)\b', regex=True, na=False)
    gen = data[mask].copy()
    if gen.empty: return pd.DataFrame(columns=data.columns)
    sorted_b = sorted([str(b).strip().lower() for b in brands_list if b], key=len, reverse=True)
    def detect(n):
        nc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', str(n).lower())).strip()
        for b in sorted_b:
            bc = re.sub(r'\s+', ' ', re.sub(r"['\.\-]", ' ', b)).strip()
            if nc.startswith(bc) and (len(nc) == len(bc) or not nc[len(bc)].isalnum()): return b.title()
        return None
    gen['Detected_Brand'] = [detect(n) for n in gen['NAME'].values]
    flagged = gen[gen['Detected_Brand'].notna()].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Detected Brand: " + flagged['Detected_Brand']
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_missing_color(data: pd.DataFrame, pattern: re.Pattern, color_categories: List[str], country_code: str) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or pattern is None: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in color_categories))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    has_color = 'COLOR' in data.columns
    names = target['NAME'].astype(str).values
    colors = target['COLOR'].astype(str).str.strip().str.lower().values if has_color else [''] * len(target)
    mask = []
    for n, c in zip(names, colors):
        if pattern.search(n): mask.append(False)
        elif has_color and c not in ['nan', '', 'none', 'null']: mask.append(False)
        else: mask.append(True)
    return target[mask].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_weight_volume_in_name(data: pd.DataFrame, weight_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not weight_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in weight_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:kg|kgs|g|gm|gms|grams|mg|mcg|ml|l|ltr|liter|litres|litre|cl|oz|ounces|lb|lbs"
        r"|tablets?|tabs?|capsules?|caps?|sachets?|count|ct|sticks?|iu"
        r"|tea\s*bags?|teabags?|bags?"
        r"|pieces?|pcs|pack|packs"
        r"|dozens?|pairs?|rolls?|sheets?|wipes?|pods?|softgels?|lozenges?|gummies|gummy|units?|serves?|servings?|vegan\s+pieces?)"
        r"|\b\d+[\u0027\u2019]?s\b"
        r"|\b(?:a\s+)?dozen\b"
        r"|\b(?:pack|box|set|bundle|lot)\s+of\s+\d+\b"
        r"|\bper\s+(?:kg|kgs?|g|gm|grams?|mg|mcg|ml|l|ltr|oz|lb)\b"
        r"|\d+\s*(?:\xc2\xb5g|\xce\xbcg|\xb5g|\u00b5g|\u03bcg|mcg|µg|μg)",
        re.IGNORECASE
    )
    return target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_incomplete_smartphone_name(data: pd.DataFrame, smartphone_category_codes: List[str]) -> pd.DataFrame:
    if not {'CATEGORY_CODE', 'NAME'}.issubset(data.columns) or not smartphone_category_codes: return pd.DataFrame(columns=data.columns)
    target = data[data['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in smartphone_category_codes))].copy()
    if target.empty: return pd.DataFrame(columns=data.columns)
    pat = re.compile(r'\b\d+\s*(gb|tb)\b', re.IGNORECASE)
    flagged = target[~target['NAME'].apply(lambda n: bool(pat.search(str(n))))].copy()
    if not flagged.empty: flagged['Comment_Detail'] = "Name missing Storage/Memory spec (e.g., 64GB)"
    return flagged.drop_duplicates(subset=['PRODUCT_SET_SID'])

def check_duplicate_products(data: pd.DataFrame, exempt_categories: List[str] = None, similarity_threshold: float = 0.70, known_colors: List[str] = None, **kwargs) -> pd.DataFrame:
    if not {'NAME', 'SELLER_NAME', 'BRAND'}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    if exempt_categories and 'CATEGORY_CODE' in d.columns:
        d = d[~d['CATEGORY_CODE'].apply(clean_category_code).isin(set(clean_category_code(c) for c in exempt_categories))]
    if d.empty: return pd.DataFrame(columns=data.columns)

    d['_norm_name']   = d['NAME'].astype(str).apply(lambda x: re.sub(r'\s+', '', normalize_text(x)))
    d['_norm_brand']  = d['BRAND'].astype(str).str.lower().str.strip()
    d['_norm_seller'] = d['SELLER_NAME'].astype(str).str.lower().str.strip()
    d['_dedup_key']   = d['_norm_seller'] + '|' + d['_norm_brand'] + '|' + d['_norm_name']

    first_seen_mask = ~d.duplicated(subset=['_dedup_key'], keep='first')
    dup_mask        = d.duplicated(subset=['_dedup_key'], keep='first')

    if not dup_mask.any(): return pd.DataFrame(columns=data.columns)

    first_occurrence = d[first_seen_mask].set_index('_dedup_key')['NAME']
    rdf = d[dup_mask].copy()
    rdf['Comment_Detail'] = rdf['_dedup_key'].map(
        lambda k: f"Duplicate: '{str(first_occurrence.get(k, ''))[:40]}'"
    )
    base_cols  = data.columns.tolist()
    extra_cols = [c for c in ['Comment_Detail'] if c not in base_cols]
    return rdf[base_cols + extra_cols].drop_duplicates(subset=['PRODUCT_SET_SID'])


# -------------------------------------------------
# NIGERIA-SPECIFIC QC CHECKS
# -------------------------------------------------
def check_nigeria_gift_card(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("gift_card", {})
    cat_codes        = rules.get("category_codes", set())
    approved_sellers = rules.get("sellers", set())
    if not cat_codes or not approved_sellers: return pd.DataFrame(columns=data.columns)
    if not {"CATEGORY_CODE", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    in_scope = data[data["CATEGORY_CODE"].apply(clean_category_code).isin(cat_codes)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised for Gift Card categories: " + flagged["SELLER_NAME"].astype(str)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_books(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    book_rules = ng_rules.get("books", {})
    if not book_rules or not {"NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_name_l"]   = d["NAME"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    flagged_idx, comment_map = [], {}
    for idx, row in d.iterrows():
        name_l, seller_l = row["_name_l"], row["_seller_l"]
        for book_name, allowed_seller in book_rules.items():
            if book_name in name_l:
                if allowed_seller is None:
                    flagged_idx.append(idx)
                    comment_map[idx] = f"No seller authorised for book: '{book_name[:60]}'"
                elif seller_l != allowed_seller:
                    flagged_idx.append(idx)
                    comment_map[idx] = f"Only '{allowed_seller}' may sell '{book_name[:40]}'"
                break
    if not flagged_idx: return pd.DataFrame(columns=data.columns)
    result = data.loc[flagged_idx].copy()
    result["Comment_Detail"] = result.index.map(comment_map)
    return result.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_tvs(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    tv_rules      = ng_rules.get("tvs", {})
    cat_codes     = tv_rules.get("category_codes", set())
    brand_sellers = tv_rules.get("brand_sellers", {})
    if not cat_codes or not brand_sellers or not {"CATEGORY_CODE", "BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    in_scope = data[data["CATEGORY_CODE"].apply(clean_category_code).isin(cat_codes)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    in_scope["_brand_l"]  = in_scope["BRAND"].astype(str).str.strip().str.lower()
    in_scope["_seller_l"] = in_scope["SELLER_NAME"].astype(str).str.strip().str.lower()
    chunks = []
    for brand_lower, approved in brand_sellers.items():
        if not approved: continue
        brand_rows = in_scope[in_scope["_brand_l"] == brand_lower]
        if brand_rows.empty: continue
        bad = brand_rows[~brand_rows["_seller_l"].isin(approved)].copy()
        if not bad.empty:
            bad["Comment_Detail"] = f"Seller not authorised for {brand_lower.upper()} TVs: " + bad["SELLER_NAME"].astype(str)
            chunks.append(bad)
    if not chunks: return pd.DataFrame(columns=data.columns)
    return pd.concat(chunks).drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_hp_toners(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("hp_toners", {})
    cat_codes        = rules.get("category_codes", set())
    approved_sellers = rules.get("sellers", set())
    if not cat_codes or not approved_sellers or not {"CATEGORY_CODE", "BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_cat"]      = d["CATEGORY_CODE"].apply(clean_category_code)
    d["_brand_l"]  = d["BRAND"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    in_scope = d[d["_cat"].isin(cat_codes) & (d["_brand_l"] == "hp")].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["_seller_l"].isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised for HP Ink/Toners: " + flagged["SELLER_NAME"].astype(str)
    return flagged[[c for c in data.columns if c in flagged.columns] + ["Comment_Detail"]].drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_apple(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    approved_sellers = ng_rules.get("apple", {}).get("sellers", set())
    if not approved_sellers or not {"BRAND", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    apple = data[data["BRAND"].astype(str).str.strip().str.lower() == "apple"].copy()
    if apple.empty: return pd.DataFrame(columns=data.columns)
    flagged = apple[~apple["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty: flagged["Comment_Detail"] = "Seller not authorised to sell Apple products: " + flagged["SELLER_NAME"].astype(str)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_xmas_tree(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rules            = ng_rules.get("xmas_tree", {})
    approved_sellers = rules.get("sellers", set())
    keywords         = rules.get("keywords", set())
    if not approved_sellers or not keywords or not {"NAME", "SELLER_NAME"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    kw_pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
    in_scope = data[data["NAME"].astype(str).str.contains(kw_pattern, na=False)].copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    flagged = in_scope[~in_scope["SELLER_NAME"].astype(str).str.strip().str.lower().isin(approved_sellers)].copy()
    if not flagged.empty:
        def _comment(row):
            m = kw_pattern.search(str(row["NAME"]))
            kw = m.group(0) if m else "?"
            return f"Seller not authorised for Xmas Tree products (keyword '{kw}'): {row['SELLER_NAME']}"
        flagged["Comment_Detail"] = flagged.apply(_comment, axis=1)
    return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_rice(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    rice_rules = ng_rules.get("rice", {})
    if not rice_rules or not {"BRAND", "SELLER_NAME", "CATEGORY_CODE"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    d = data.copy()
    d["_brand_l"]  = d["BRAND"].astype(str).str.strip().str.lower()
    d["_seller_l"] = d["SELLER_NAME"].astype(str).str.strip().str.lower()
    d["_cat"]      = d["CATEGORY_CODE"].apply(clean_category_code)
    chunks = []
    for brand_lower, rules in rice_rules.items():
        approved  = rules.get("sellers", set())
        cat_codes = rules.get("category_codes", set())
        brand_rows = d[d["_brand_l"] == brand_lower].copy()
        if brand_rows.empty: continue
        if cat_codes: brand_rows = brand_rows[brand_rows["_cat"].isin(cat_codes)]
        if brand_rows.empty: continue
        bad = brand_rows[~brand_rows["_seller_l"].isin(approved)].copy()
        if not bad.empty:
            bad["Comment_Detail"] = f"Seller not authorised to sell {brand_lower.title()} rice: " + bad["SELLER_NAME"].astype(str)
            chunks.append(bad)
    if not chunks: return pd.DataFrame(columns=data.columns)
    return pd.concat(chunks).drop_duplicates(subset=["PRODUCT_SET_SID"])

def check_nigeria_powerbanks(data: pd.DataFrame, ng_rules: Dict) -> pd.DataFrame:
    pb_rules       = ng_rules.get("powerbanks", {})
    allowed_brands = pb_rules.get("brands", set())
    cat_codes      = pb_rules.get("category_codes", set())
    MIN_MAH        = 20_000
    if not allowed_brands or not {"CATEGORY_CODE", "NAME", "BRAND"}.issubset(data.columns): return pd.DataFrame(columns=data.columns)
    _mah_pat = re.compile(r'\b(\d[\d,]*)\s*mah\b', re.IGNORECASE)
    def _exceeds_threshold(name: str) -> bool:
        for m in _mah_pat.finditer(str(name)):
            try:
                val = int(m.group(1).replace(',', ''))
                if val >= MIN_MAH: return True
            except ValueError: pass
        return False
    d = data.copy()
    d["_cat"]      = d["CATEGORY_CODE"].apply(clean_category_code)
    d["_name"]     = d["NAME"].astype(str)
    d["_brand_l"]  = d["BRAND"].astype(str).str.strip().str.lower()
    in_scope = d[d["_cat"].isin(cat_codes)].copy() if cat_codes else d.copy()
    if in_scope.empty: return pd.DataFrame(columns=data.columns)
    high_cap = in_scope[in_scope["_name"].apply(_exceeds_threshold)].copy()
    if high_cap.empty: return pd.DataFrame(columns=data.columns)
    flagged = high_cap[~high_cap["_brand_l"].isin(allowed_brands)].copy()
    if not flagged.empty:
        def _comment(row):
            m = _mah_pat.search(row["_name"])
            mah_str = m.group(0) if m else ">=20,000mAh"
            return f"Brand '{row['BRAND']}' not approved for {mah_str} powerbanks. Approved: {', '.join(b.title() for b in sorted(allowed_brands))}"
        flagged["Comment_Detail"] = flagged.apply(_comment, axis=1)
    return flagged[[c for c in data.columns if c in flagged.columns] + ["Comment_Detail"]].drop_duplicates(subset=["PRODUCT_SET_SID"])


if _reg is not None:
    _reg.REGISTRY.update({
        'check_restricted_brands':           check_restricted_brands,
        'check_suspected_fake_products':     check_suspected_fake_products,
        'check_refurb_seller_approval':      check_refurb_seller_approval,
        'check_product_warranty':            check_product_warranty,
        'check_seller_approved_for_books':   check_seller_approved_for_books,
        'check_seller_approved_for_perfume': check_seller_approved_for_perfume,
        'check_perfume_tester':              check_perfume_tester,
        'check_counterfeit_sneakers':        check_counterfeit_sneakers,
        'check_counterfeit_jerseys':         check_counterfeit_jerseys,
        'check_prohibited_products':         check_prohibited_products,
        'check_unnecessary_words':           check_unnecessary_words,
        'check_single_word_name':            check_single_word_name,
        'check_generic_brand_issues':        check_generic_brand_issues,
        'check_fashion_brand_issues':        check_fashion_brand_issues,
        'check_brand_in_name':               check_brand_in_name,
        'check_wrong_variation':             check_wrong_variation,
        'check_generic_with_brand_in_name':  check_generic_with_brand_in_name,
        'check_missing_color':               check_missing_color,
        'check_weight_volume_in_name':       check_weight_volume_in_name,
        'check_incomplete_smartphone_name':  check_incomplete_smartphone_name,
        'check_duplicate_products':          check_duplicate_products,
        'check_miscellaneous_category':      check_miscellaneous_category,
        'compile_regex_patterns':            compile_regex_patterns,
        'check_nigeria_gift_card':           check_nigeria_gift_card,
        'check_nigeria_books':               check_nigeria_books,
        'check_nigeria_tvs':                 check_nigeria_tvs,
        'check_nigeria_hp_toners':           check_nigeria_hp_toners,
        'check_nigeria_apple':               check_nigeria_apple,
        'check_nigeria_xmas_tree':           check_nigeria_xmas_tree,
        'check_nigeria_rice':                check_nigeria_rice,
        'check_nigeria_powerbanks':          check_nigeria_powerbanks,
        'load_nigeria_qc_rules':             load_nigeria_qc_rules,
    })

# -------------------------------------------------
# MASTER VALIDATION RUNNER
# -------------------------------------------------
def validate_products(data: pd.DataFrame, support_files: Dict, country_validator: CountryValidator, data_has_warranty_cols: bool, common_sids: Optional[set] = None, skip_validators: Optional[List[str]] = None):
    data['PRODUCT_SET_SID'] = data['PRODUCT_SET_SID'].astype(str).str.strip()
    flags_mapping = support_files['flags_mapping']
    country_restricted_rules = support_files.get('restricted_brands_all', {}).get(country_validator.country, [])
    suspected_fake_df = support_files.get('suspected_fake', {}).get(country_validator.code, pd.DataFrame()) if isinstance(support_files.get('suspected_fake'), dict) else pd.DataFrame()
    country_prohibited_words = support_files.get('prohibited_words_all', {}).get(country_validator.code, [])
    validations = [
        ("Wrong Category", check_miscellaneous_category, {
            'categories_list': support_files.get('categories_names_list', []),
            'compiled_rules': st.session_state.get('compiled_json_rules', {}),
            'cat_path_to_code': support_files.get('cat_path_to_code', {}),
            'code_to_path': support_files.get('code_to_path', {}),
        }),
        ("Restricted brands", check_restricted_brands, {'country_rules': country_restricted_rules}),
        ("Suspected Fake product", check_suspected_fake_products, {'suspected_fake_df': suspected_fake_df}),
        ("Seller Not approved to sell Refurb", check_refurb_seller_approval, {'refurb_data': support_files.get('refurb_data', {}), 'country_code': country_validator.code}),
        ("Product Warranty", check_product_warranty, {'warranty_category_codes': support_files['warranty_category_codes']}),
        ("Seller Approve to sell books", check_seller_approved_for_books, {'books_data': support_files.get('books_data', {}), 'country_code': country_validator.code, 'book_category_codes': support_files['book_category_codes']}),
        ("Seller Approved to Sell Perfume", check_seller_approved_for_perfume, {'perfume_category_codes': support_files['perfume_category_codes'], 'perfume_data': support_files.get('perfume_data', {}), 'country_code': country_validator.code}),
        ("Perfume Tester", check_perfume_tester, {'perfume_category_codes': support_files['perfume_category_codes'], 'perfume_data': support_files.get('perfume_data', {})}),
        ("Counterfeit Sneakers", check_counterfeit_sneakers, {'sneaker_category_codes': support_files['sneaker_category_codes'], 'sneaker_sensitive_brands': support_files['sneaker_sensitive_brands']}),
        ("Suspected counterfeit Jerseys", check_counterfeit_jerseys, {'jerseys_data': support_files.get('jerseys_data', {}), 'country_code': country_validator.code}),
        ("Prohibited products", check_prohibited_products, {'prohibited_rules': country_prohibited_words}),
        ("Unnecessary words in NAME", check_unnecessary_words, {'pattern': compile_regex_patterns(support_files['unnecessary_words'])}),
        ("Single-word NAME", check_single_word_name, {'book_category_codes': support_files['book_category_codes'], 'books_data': support_files.get('books_data', {})}),
        ("Generic BRAND Issues", check_generic_brand_issues, {}),
        ("Fashion brand issues", check_fashion_brand_issues, {}),
        ("BRAND name repeated in NAME", check_brand_in_name, {}),
        ("Wrong Variation", check_wrong_variation, {'allowed_variation_codes': list(set(support_files.get('variation_allowed_codes', []) + support_files.get('category_fas', [])))}),
        ("Generic branded products with genuine brands", check_generic_with_brand_in_name, {'brands_list': support_files.get('known_brands', [])}),
        ("Missing COLOR", check_missing_color, {'pattern': compile_regex_patterns(support_files['colors']), 'color_categories': support_files['color_categories'], 'country_code': country_validator.code}),
        ("Missing Weight/Volume", check_weight_volume_in_name, {'weight_category_codes': support_files.get('weight_category_codes', [])}),
        ("Incomplete Smartphone Name", check_incomplete_smartphone_name, {'smartphone_category_codes': support_files.get('smartphone_category_codes', [])}),
        ("Duplicate product", check_duplicate_products, {'exempt_categories': support_files.get('duplicate_exempt_codes', []), 'known_colors': support_files['colors']}),
    ]

    if country_validator.code == "NG":
        _ng = support_files.get("ng_qc_rules", {})
        validations += [
            ("NG - Gift Card Seller",  check_nigeria_gift_card,  {"ng_rules": _ng}),
            ("NG - Books Seller",      check_nigeria_books,      {"ng_rules": _ng}),
            ("NG - TV Brand Seller",   check_nigeria_tvs,        {"ng_rules": _ng}),
            ("NG - HP Toners Seller",  check_nigeria_hp_toners,  {"ng_rules": _ng}),
            ("NG - Apple Seller",      check_nigeria_apple,      {"ng_rules": _ng}),
            ("NG - Xmas Tree Seller",  check_nigeria_xmas_tree,  {"ng_rules": _ng}),
            ("NG - Rice Brand Seller", check_nigeria_rice,       {"ng_rules": _ng}),
            ("NG - Powerbank Capacity",check_nigeria_powerbanks, {"ng_rules": _ng}),
        ]

    results = {}
    dup_groups = {}
    if {'NAME','BRAND','SELLER_NAME','COLOR'}.issubset(data.columns):
        dt = data.copy()
        dt['dup_key'] = dt[['NAME','BRAND','SELLER_NAME','COLOR']].apply(lambda r: tuple(str(v).strip().lower() for v in r), axis=1)
        for k, v in dt.groupby('dup_key')['PRODUCT_SET_SID'].apply(list).items():
            if len(v) > 1:
                for sid in v: dup_groups[sid] = v
    restricted_keys = {}
    validation_errors = []

    with st.spinner("Validating products... This may take a moment."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_name = {}
            for i, (name, func, kwargs) in enumerate(validations):
                if skip_validators and name in skip_validators: continue
                if country_validator.should_skip_validation(name): continue
                ckwargs = {'data': data, **kwargs}
                if name in ["Generic BRAND Issues", "Fashion brand issues"]: ckwargs['valid_category_codes_fas'] = support_files.get('category_fas', [])
                if name == "Fashion brand issues": ckwargs['code_to_path'] = support_files.get('code_to_path', {})
                flag_hash = compute_flag_input_hash(data, name, ckwargs)
                cache_path = os.path.join(FLAG_CACHE_DIR, f"{flag_hash}.pkl")
                future_to_name[executor.submit(run_cached_check, func, cache_path, ckwargs)] = name

            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    res = future.result()
                    if not res.empty and 'PRODUCT_SET_SID' in res.columns:
                        res = res.loc[:, ~res.columns.duplicated()].copy()
                        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
                        if name in ["Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers", "Seller Not approved to sell Refurb", "Restricted brands"]:
                            res['match_key'] = res.apply(create_match_key, axis=1)
                            restricted_keys.setdefault(name, set()).update(res['match_key'].unique())
                        expanded_sids = set()
                        for sid in set(res['PRODUCT_SET_SID'].unique()): expanded_sids.update(dup_groups.get(sid, [sid]))
                        final_res = data[data['PRODUCT_SET_SID'].isin(expanded_sids)].copy()
                        if 'Comment_Detail' in res.columns: final_res['Comment_Detail'] = res['Comment_Detail']
                        if name in results and not results[name].empty: results[name] = pd.concat([results[name], final_res]).drop_duplicates(subset=['PRODUCT_SET_SID'])
                        else: results[name] = final_res
                    else:
                        if name not in results: results[name] = pd.DataFrame(columns=data.columns)
                except Exception as e:
                    logger.error(f"Validation error in '{name}': {e}")
                    validation_errors.append((name, str(e)))
                    if name not in results: results[name] = pd.DataFrame(columns=data.columns)

    if validation_errors:
        st.warning(f"{len(validation_errors)} validation checks encountered errors.")
        with st.expander("View Error Details"):
            for e_name, e_msg in validation_errors: st.error(f"**{e_name}**: {e_msg}")
    if restricted_keys:
        data['match_key'] = data.apply(create_match_key, axis=1)
        for fname, keys in restricted_keys.items():
            extra = data[data['match_key'].isin(keys)].copy()
            results[fname] = pd.concat([results.get(fname, pd.DataFrame()), extra]).drop_duplicates(subset=['PRODUCT_SET_SID'])

    target_lang = 'fr' if country_validator.country == "Morocco" else 'en'

    rows = []
    processed = set()
    for name, _, _ in validations:
        if name not in results or results[name].empty or 'PRODUCT_SET_SID' not in results[name].columns: continue
        res = results[name]
        rinfo = flags_mapping.get(name, {'reason': "1000007 - Other Reason", 'en': f"Flagged by {name}", 'fr': f"Flagged by {name}", 'ar': f"Flagged by {name}"})
        base_comment = rinfo.get(target_lang, rinfo.get('en'))
        res['PRODUCT_SET_SID'] = res['PRODUCT_SET_SID'].astype(str).str.strip()
        flagged = pd.merge(res[['PRODUCT_SET_SID', 'Comment_Detail']] if 'Comment_Detail' in res.columns else res[['PRODUCT_SET_SID']], data, on='PRODUCT_SET_SID', how='left')
        if 'Comment_Detail' not in flagged.columns and 'Comment_Detail' in res.columns:
            if isinstance(res['Comment_Detail'], pd.DataFrame): flagged['Comment_Detail'] = res['Comment_Detail'].iloc[:, 0]
            else: flagged['Comment_Detail'] = res['Comment_Detail']
        for _, r in flagged.iterrows():
            sid = str(r['PRODUCT_SET_SID']).strip()
            if sid in processed: continue
            processed.add(sid)
            det = r.get('Comment_Detail', '')
            comment_str = f"{base_comment} ({det})" if pd.notna(det) and det else base_comment
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Rejected', 'Reason': rinfo['reason'], 'Comment': comment_str, 'FLAG': name, 'SellerName': r.get('SELLER_NAME', '')})

    for _, r in data[~data['PRODUCT_SET_SID'].astype(str).str.strip().isin(processed)].iterrows():
        sid = str(r['PRODUCT_SET_SID']).strip()
        if sid not in processed:
            rows.append({'ProductSetSid': sid, 'ParentSKU': r.get('PARENTSKU', ''), 'Status': 'Approved', 'Reason': "", 'Comment': "", 'FLAG': "", 'SellerName': r.get('SELLER_NAME', '')})
            processed.add(sid)
    final_df = pd.DataFrame(rows)
    for c in ["ProductSetSid", "ParentSKU", "Status", "Reason", "Comment", "FLAG", "SellerName"]:
        if c not in final_df.columns: final_df[c] = ""
    return country_validator.ensure_status_column(final_df), results

@st.cache_data(show_spinner=False, ttl=3600)
def cached_validate_products(data_hash: str, _data: pd.DataFrame, _support_files: Dict, country_code: str, data_has_warranty_cols: bool, skip_validators: Optional[List[str]] = None):
    country_name = next((k for k, v in CountryValidator.COUNTRY_CONFIG.items() if v['code'] == country_code), "Kenya")
    cv = CountryValidator(country_name)
    return validate_products(_data, _support_files, cv, data_has_warranty_cols, skip_validators=skip_validators)

# -------------------------------------------------
# EXPORTS UTILITIES
# -------------------------------------------------
def to_excel_base(df, sheet, cols, writer, format_rules=False):
    df_p = df.copy()
    for c in cols:
        if c not in df_p.columns: df_p[c] = pd.NA
    df_to_write = df_p[[c for c in cols if c in df_p.columns]]
    df_to_write = _repair_mojibake(df_to_write.copy())
    df_to_write.to_excel(writer, index=False, sheet_name=sheet)
    if format_rules and 'Status' in df_to_write.columns:
        wb = writer.book
        ws = writer.sheets[sheet]
        rf = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        gf = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        idx = df_to_write.columns.get_loc('Status')
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Rejected"', 'format': rf})
        ws.conditional_format(1, idx, len(df_to_write), idx, {'type': 'cell', 'criteria': 'equal', 'value': '"Approved"', 'format': gf})

def write_excel_single(df, sheet_name, cols, auxiliary_df=None, aux_sheet_name=None, aux_cols=None, format_status=False, full_data_stats=False):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        to_excel_base(df, sheet_name, cols, writer, format_rules=format_status)
        if auxiliary_df is not None and not auxiliary_df.empty: to_excel_base(auxiliary_df, aux_sheet_name, aux_cols, writer)
        if full_data_stats and 'SELLER_NAME' in df.columns and 'Status' in df.columns:
            ws = writer.book.add_worksheet('Sellers Data')
            fmt = writer.book.add_format({'bold': True, 'bg_color': '#E6F0FA', 'border': 1, 'align': 'center'})
            df['Rejected_Count'] = (df['Status'] == 'Rejected').astype(int)
            df['Approved_Count'] = (df['Status'] == 'Approved').astype(int)
            summ = df.groupby('SELLER_NAME').agg(Rejected=('Rejected_Count', 'sum'), Approved=('Approved_Count', 'sum')).reset_index().sort_values('Rejected', ascending=False)
            summ.insert(0, 'Rank', range(1, len(summ) + 1))
            ws.write(0, 0, "Sellers Summary (This File)", fmt)
            summ.to_excel(writer, sheet_name='Sellers Data', startrow=1, index=False)
    output.seek(0)
    return output

def generate_smart_export(df, filename_prefix, export_type='simple', auxiliary_df=None):
    cols = FULL_DATA_COLS + [c for c in ["Status", "Reason", "Comment", "FLAG", "SellerName"] if c not in FULL_DATA_COLS] if export_type == 'full' else PRODUCTSETS_COLS
    if len(df) <= SPLIT_LIMIT:
        data = write_excel_single(df, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
        return data, f"{filename_prefix}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        zb = BytesIO()
        with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(df), SPLIT_LIMIT):
                chunk = df.iloc[i : i + SPLIT_LIMIT]
                excel_data = write_excel_single(chunk, "ProductSets", cols, auxiliary_df, "RejectionReasons", REJECTION_REASONS_COLS, True, export_type == 'full')
                zf.writestr(f"{filename_prefix}_Part_{(i//SPLIT_LIMIT)+1}.xlsx", excel_data.getvalue())
        zb.seek(0)
        return zb, f"{filename_prefix}.zip", "application/zip"

def prepare_full_data_merged(data_df, final_report_df):
    try:
        d_cp, r_cp = data_df.copy(), final_report_df.copy()
        d_cp['PRODUCT_SET_SID'] = d_cp['PRODUCT_SET_SID'].astype(str).str.strip()
        r_cp['ProductSetSid'] = r_cp['ProductSetSid'].astype(str).str.strip()
        
        # Build full category path column
        _code_to_path = st.session_state.get('support_files', {}).get('code_to_path', {})
        if _code_to_path and 'CATEGORY_CODE' in d_cp.columns:
            d_cp['FULL_CATEGORY_PATH'] = d_cp['CATEGORY_CODE'].apply(
                lambda c: _code_to_path.get(str(c).strip(), '') if pd.notna(c) else ''
            )
        else:
            d_cp['FULL_CATEGORY_PATH'] = ''
            
        merged = pd.merge(
            d_cp, 
            r_cp[["ProductSetSid", "Status", "Reason", "Comment", "FLAG", "SellerName"]], 
            left_on="PRODUCT_SET_SID", right_on="ProductSetSid", how='left'
        )
        
        if 'ProductSetSid' in merged.columns: 
            merged.drop(columns=['ProductSetSid'], inplace=True)
            
        return merged
        
    except Exception as e:
        logger.error(f"prepare_full_data_merged: {e}")
        return pd.DataFrame()

# -------------------------------------------------
# UTILITIES FOR BRIDGE & DATA MUTATION
# -------------------------------------------------
def apply_rejection(sids: list, reason_code: str, comment: str, flag_name: str):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(sids), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', reason_code, comment, flag_name]
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()

def restore_single_item(sid):
    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
    st.session_state.quick_rejections.pop(sid, None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Restored item to previous state!")

REASON_MAP = {
    "REJECT_POOR_IMAGE": "Poor images",
    "REJECT_WRONG_CAT": "Wrong Category",
    "REJECT_FAKE": "Suspected Fake product",
    "REJECT_BRAND": "Restricted brands",
    "REJECT_PROHIBITED": "Prohibited products",
    "REJECT_COLOR": "Missing COLOR",
    "REJECT_WRONG_BRAND": "Generic branded products with genuine brands",
    "OTHER_CUSTOM": "Other Reason (Custom)"
}

# -------------------------------------------------
# HTML GRID BUILDER
# -------------------------------------------------
def build_fast_grid_html(page_data, flags_mapping, country, page_warnings, rejected_state, cols_per_row):
    O = JUMIA_COLORS["primary_orange"]
    G = JUMIA_COLORS["success_green"]
    R = JUMIA_COLORS["jumia_red"]

    committed_json = json.dumps(rejected_state)
    html_dir = "rtl" if st.session_state.ui_lang == "ar" else "ltr"

    cards_data = []
    for _, row in page_data.iterrows():
        sid = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        if img_url.startswith("http://"): img_url = img_url.replace("http://", "https://")
        if not img_url.startswith("http"): img_url = "https://via.placeholder.com/150?text=No+Image"
        sale_p = row.get("GLOBAL_SALE_PRICE")
        reg_p = row.get("GLOBAL_PRICE")
        usd_val = sale_p if pd.notna(sale_p) and str(sale_p).strip() != "" else reg_p
        price_str = format_local_price(usd_val, st.session_state.selected_country) if pd.notna(usd_val) else ""
        cards_data.append({
            "sid": sid, "img": img_url,
            "name": str(row.get("NAME", "")),
            "brand": str(row.get("BRAND", "Unknown Brand")),
            "cat": str(row.get("CATEGORY", "Unknown Category")),
            "seller": str(row.get("SELLER_NAME", "Unknown Seller")),
            "warnings": page_warnings.get(sid, []),
            "price": price_str
        })
    cards_json = json.dumps(cards_data)

    rejected_label = str(_t('rejected') or 'REJECTED').upper()

    return f"""<!DOCTYPE html>
<html dir="{html_dir}">
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
  body{{background:#f5f5f5;padding:8px;}}
  .ctrl-bar{{
    position:-webkit-sticky;position:sticky;top:0;z-index:99999;
    display:flex;align-items:center;gap:8px;flex-wrap:wrap;
    padding:8px 12px;
    background:rgba(255,255,255,0.95);
    backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
    border-bottom:2px solid {O};border-radius:4px;margin-bottom:12px;
    box-shadow:0 4px 16px rgba(0,0,0,0.15);
  }}
  .sel-count{{font-weight:700;color:{O};font-size:13px;min-width:80px;}}
  .reason-sel{{flex:1;min-width:160px;padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff;cursor:pointer;outline:none;}}
  .batch-btn{{padding:7px 14px;background:{O};color:#fff;border:none;border-radius:4px;font-weight:700;font-size:12px;cursor:pointer;white-space:nowrap;}}
  .batch-btn:hover{{opacity:.88;}}
  .desel-btn{{padding:7px 12px;background:#fff;color:#555;border:1px solid #ccc;border-radius:4px;font-size:12px;cursor:pointer;white-space:nowrap;}}
  .desel-btn:hover{{background:#f5f5f5;}}
  .grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
  .card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s,box-shadow .15s;z-index:1;}}
  .card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
  .card.staged-rej{{border-color:{R};box-shadow:0 0 0 3px rgba(231,60,23,.2);background:rgba(231,60,23,.04);}}
  .card.committed-rej{{border-color:#bbb;opacity:.6;}}
  .card-img-wrap{{position:relative;cursor:pointer;border-radius:6px;background:#fff;display:flex;align-items:center;justify-content:center;height:180px;}}
  .card-img{{width:100%;height:180px;object-fit:contain;border-radius:6px;display:block;transition:transform 0.2s ease-out,box-shadow 0.2s ease-out;}}
  .card.committed-rej .card-img{{filter:grayscale(80%);}}
  .card-img.locally-zoomed{{transform:scale(2.3);box-shadow:0 15px 50px rgba(0,0,0,0.6);border:2px solid {O};background:#fff;position:relative;z-index:9999;border-radius:8px;}}
  .zoom-btn{{position:absolute;bottom:6px;left:6px;width:28px;height:28px;background:rgba(255,255,255,0.95);border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,0.3);z-index:10000;font-size:14px;transition:background 0.1s,transform 0.1s;}}
  .zoom-btn:hover{{background:#fff;transform:scale(1.1);}}
  .tick{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;color:transparent;font-size:13px;font-weight:900;pointer-events:none;z-index:10;}}
  .card.selected .tick{{background:{G};color:#fff;}}
  .card.staged-rej .tick{{background:{R};color:#fff;}}
  .warn-wrap{{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;z-index:5;pointer-events:none;}}
  .warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;padding:3px 7px;border-radius:10px;}}
  .price-badge{{position:absolute;top:6px;left:6px;background:rgba(76,175,80,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 7px;border-radius:10px;z-index:5;pointer-events:none;box-shadow:0 2px 4px rgba(0,0,0,0.2);}}
  .rej-overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.90);border-radius:6px;flex-direction:column;align-items:center;justify-content:center;z-index:20;gap:5px;padding:8px;text-align:center;}}
  .card.committed-rej .rej-overlay{{display:flex;}}
  .card.staged-rej .rej-overlay.staged{{display:flex;}}
  .rej-badge{{background:{R};color:#fff;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;}}
  .rej-badge.pending{{background:{O};}}
  .rej-label{{font-size:10px;color:{R};font-weight:600;max-width:120px;}}
  .undo-btn{{margin-top:8px;padding:6px 12px;background:#313133;color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:bold;cursor:pointer;box-shadow:0 2px 4px rgba(0,0,0,0.2);}}
  .undo-btn:hover{{background:#000;}}
  .meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
  .meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
  .meta .br{{color:{O};font-weight:700;margin:2px 0;}}
  .meta .ct{{color:#666;font-size:10px;word-break:break-word;white-space:normal;line-height:1.4;}}
  .meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}
  .acts{{display:flex;gap:4px;margin-top:8px;}}
  .act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;background:{O};}}
  .act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
</style>
</head>
<body>
<div class="ctrl-bar">
  <span class="sel-count" id="sel-count-bar">0 {_t("items_pending")}</span>
  <select class="reason-sel" id="batch-reason">
    <option value="REJECT_POOR_IMAGE">{_t("poor_img")}</option>
    <option value="REJECT_WRONG_CAT">{_t("wrong_cat")}</option>
    <option value="REJECT_FAKE">{_t("fake_prod")}</option>
    <option value="REJECT_BRAND">{_t("restr_brand")}</option>
    <option value="REJECT_WRONG_BRAND">{_t("wrong_brand")}</option>
    <option value="REJECT_PROHIBITED">{_t("prohibited")}</option>
    <option value="REJECT_COLOR">{_t("missing_color")}</option>
  </select>
  <button class="batch-btn" onclick="doBatchReject()">{_t("batch_reject")}</button>
  <button class="desel-btn" onclick="window.doSelectAll()">{_t("select_all")}</button>
  <button class="desel-btn" onclick="doDeselAll()">{_t("deselect_all")}</button>
</div>
<div class="grid" id="card-grid"></div>
<script>
function escapeHtml(unsafe) {{
    return (unsafe || "").toString()
         .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}}
var CARDS = {cards_json};
var COMMITTED = {committed_json};
window._gridSelected = window._gridSelected || {{}};
window._stagedRejections = window._stagedRejections || {{}};
var selected = window._gridSelected;
var staged = window._stagedRejections;
try {{
    window.addEventListener("beforeunload", function() {{
        sessionStorage.setItem("jt_iframe_scroll", window.scrollY);
        if (window.parent && window.parent.document) {{
            var main = window.parent.document.querySelector('.main');
            if (main) window.parent.sessionStorage.setItem("jt_parent_scroll", main.scrollTop);
        }}
    }});
    window.addEventListener("load", function() {{
        var iScroll = sessionStorage.getItem("jt_iframe_scroll");
        if (iScroll) {{ setTimeout(function() {{ window.scrollTo(0, parseInt(iScroll)); }}, 20); }}
        if (window.parent && window.parent.document) {{
            var pScroll = window.parent.sessionStorage.getItem("jt_parent_scroll");
            if (pScroll) {{
                var main = window.parent.document.querySelector('.main');
                if (main) {{ setTimeout(function() {{ main.scrollTo({{top: parseInt(pScroll), behavior: 'instant'}}); }}, 30); }}
            }}
        }}
    }});
}} catch(e) {{}}
try {{
    if (window.parent && window.parent.document) {{
        window.parent._jtClickListener = function(e) {{
            let btn = e.target.closest('button');
            if (!btn) return;
            let txt = btn.innerText;
            if (txt.includes('Next') || txt.includes('Prev') || txt.includes('Generate') || txt.includes('Download') || txt.includes('Jump')) {{
                let total = Object.keys(window._gridSelected).length + Object.keys(window._stagedRejections).length;
                if (total > 0) {{
                    if (!confirm("Wait! You have " + total + " products selected.\\nClick 'Cancel' to stay and Batch Reject.\\nClick 'OK' to ignore them.")) {{
                        e.preventDefault(); e.stopPropagation();
                    }} else {{
                        for(let k in window._gridSelected) delete window._gridSelected[k];
                        for(let k in window._stagedRejections) delete window._stagedRejections[k];
                    }}
                }}
            }}
        }};
        window.parent.document.removeEventListener('click', window.parent._jtClickListener, true);
        window.parent.document.addEventListener('click', window.parent._jtClickListener, true);
    }}
}} catch(e) {{ console.warn("Interceptor blocked"); }}
function sendMsg(type, payload) {{
  try {{
    var par = window.parent;
    var inputs = par.document.querySelectorAll('input[type="text"]');
    var bridge = null;
    for (var i = 0; i < inputs.length; i++) {{
      if (inputs[i].getAttribute('aria-label') === 'jtbridge' || inputs[i].placeholder === 'JTBRIDGE_UNIQUE_DO_NOT_USE') {{
        bridge = inputs[i]; break;
      }}
    }}
    if (!bridge) return;
    var currIframeScroll = window.scrollY;
    var main = par.document.querySelector('.main');
    var currParentScroll = main ? main.scrollTop : 0;
    var msg = JSON.stringify({{action: type, payload: payload}});
    bridge.focus({{ preventScroll: true }});
    window.scrollTo(0, currIframeScroll);
    Object.getOwnPropertyDescriptor(par.HTMLInputElement.prototype, 'value').set.call(bridge, msg);
    bridge.dispatchEvent(new par.Event('input', {{bubbles: true}}));
    setTimeout(function() {{
        bridge.blur();
        if (main) main.scrollTop = currParentScroll;
        bridge.dispatchEvent(new par.KeyboardEvent('keydown', {{bubbles: true, cancelable: true, key: 'Enter', keyCode: 13}}));
    }}, 150);
  }} catch(ex) {{ console.error('jtbridge error:', ex); }}
}}
function updateSelCount() {{
  const n = Object.keys(selected).length + Object.keys(staged).length;
  document.getElementById('sel-count-bar').textContent = n + ' {_t("items_pending")}';
}}
window.toggleZoom = function(sid) {{
    const img = document.querySelector('#card-' + sid + ' .card-img');
    if (!img) return;
    if (img.classList.contains('locally-zoomed')) {{
        img.classList.remove('locally-zoomed');
        if(img.closest('.card')) img.closest('.card').style.zIndex = '1';
    }} else {{
        document.querySelectorAll('.locally-zoomed').forEach(el => {{
            el.classList.remove('locally-zoomed');
            if (el.closest('.card')) el.closest('.card').style.zIndex = '1';
        }});
        img.classList.add('locally-zoomed');
        img.closest('.card').style.zIndex = '999';
    }}
}}
function renderCard(card) {{
  const sid = card.sid;
  const img = escapeHtml(card.img);
  const isCommitted = sid in COMMITTED;
  const isStaged = sid in staged;
  const isSelected = !isCommitted && !isStaged && (sid in selected);
  let cls = 'card';
  if (isCommitted) cls += ' committed-rej';
  else if (isStaged) cls += ' staged-rej';
  else if (isSelected) cls += ' selected';
  const shortName = card.name.length > 38 ? escapeHtml(card.name.slice(0,38))+'…' : escapeHtml(card.name);
  const warnHtml = (card.warnings || []).map(w => `<span class="warn-badge">${{escapeHtml(w)}}</span>`).join('');
  const priceHtml = card.price ? `<div class="price-badge">${{escapeHtml(card.price)}}</div>` : '';
  const zoomHtml = `<div class="zoom-btn" onclick="event.stopPropagation();window.toggleZoom('${{sid}}')"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>`;
  let overlayHtml = '';
  let actHtml = '';
  if (isCommitted) {{
      const rejLabel = escapeHtml((COMMITTED[sid]||'').replace(/_/g,' '));
      overlayHtml = `<div class="rej-overlay"><div class="rej-badge">{rejected_label}</div><div class="rej-label">${{rejLabel}}</div><button class="undo-btn" onclick="event.stopPropagation();window.undoReject('${{sid}}')">{_t('undo')}</button></div>`;
  }} else if (isStaged) {{
      const stagedLabel = escapeHtml((staged[sid]||'').replace(/_/g,' '));
      overlayHtml = `<div class="rej-overlay staged"><div class="rej-badge pending">PENDING</div><div class="rej-label">${{stagedLabel}}</div><button class="undo-btn" onclick="event.stopPropagation();window.clearStaged('${{sid}}')">{_t('clear_sel')}</button></div>`;
  }} else {{
      actHtml = `<div class="acts"><button class="act-btn" onclick="event.stopPropagation();window.stageReject('${{sid}}','REJECT_POOR_IMAGE')">{_t('poor_img')}</button><select class="act-more" onchange="if(this.value){{event.stopPropagation();window.stageReject('${{sid}}',this.value);this.value=''}}"><option value="">{_t('more_options')}</option><option value="REJECT_WRONG_CAT">{_t('wrong_cat')}</option><option value="REJECT_FAKE">{_t('fake_prod')}</option><option value="REJECT_BRAND">{_t('restr_brand')}</option><option value="REJECT_PROHIBITED">{_t('prohibited')}</option><option value="REJECT_COLOR">{_t('missing_color')}</option><option value="REJECT_WRONG_BRAND">{_t('wrong_brand')}</option></select></div>`;
  }}
  return `<div class="${{cls}}" id="card-${{sid}}"><div class="card-img-wrap" onclick="window.toggleSelect('${{sid}}', event)">${{priceHtml}}<div class="warn-wrap">${{warnHtml}}</div><img class="card-img" src="${{img}}" onerror="this.src='https://via.placeholder.com/150?text=No+Image'">${{zoomHtml}}${{overlayHtml}}<div class="tick">&#10003;</div></div><div class="meta"><div class="nm" title="${{escapeHtml(card.name)}}">${{shortName}}</div><div class="br">${{escapeHtml(card.brand)}}</div><div class="ct">${{escapeHtml(card.cat)}}</div><div class="sl">${{escapeHtml(card.seller)}}</div></div>${{actHtml}}</div>`;
}}
function renderAll() {{
  document.getElementById('card-grid').innerHTML = CARDS.map(renderCard).join('');
  updateSelCount();
}}
function replaceCard(sid) {{
  const el = document.getElementById('card-'+sid);
  if (!el) return;
  const card = CARDS.find(c => c.sid === sid);
  if (card) {{ const t=document.createElement('div'); t.innerHTML=renderCard(card); el.replaceWith(t.firstElementChild); }}
}}
window.doSelectAll = function() {{
  CARDS.forEach(c => {{ if (!(c.sid in COMMITTED) && !(c.sid in staged)) selected[c.sid] = true; }});
  renderAll(); updateSelCount();
}}
window.toggleSelect = function(sid, e) {{
  const img = document.querySelector('#card-' + sid + ' .card-img');
  if (img && img.classList.contains('locally-zoomed')) {{
      img.classList.remove('locally-zoomed');
      img.closest('.card').style.zIndex = '1';
      return;
  }}
  if (sid in COMMITTED) return;
  if (sid in staged) {{ delete staged[sid]; }}
  else if (sid in selected) {{ delete selected[sid]; }}
  else {{ selected[sid] = true; }}
  replaceCard(sid); updateSelCount();
}}
window.stageReject = function(sid, reasonKey) {{
  if (sid in selected) delete selected[sid];
  staged[sid] = reasonKey;
  replaceCard(sid); updateSelCount();
}}
window.clearStaged = function(sid) {{
  delete staged[sid];
  replaceCard(sid); updateSelCount();
}}
window.undoReject = function(sid) {{
  sendMsg('undo', {{[sid]: true}});
  delete COMMITTED[sid];
  replaceCard(sid); updateSelCount();
}}
window.doBatchReject = function() {{
  const batchReason = document.getElementById('batch-reason').value;
  const payload = {{}};
  let count = 0;
  for (let sid in staged) {{ payload[sid] = staged[sid]; count++; }}
  for (let sid in selected) {{ payload[sid] = batchReason; count++; }}
  if (count === 0) return;
  for (let sid in payload) {{
      COMMITTED[sid] = payload[sid];
      delete selected[sid];
      delete staged[sid];
  }}
  sendMsg('reject', payload);
  renderAll(); updateSelCount();
}}
window.doDeselAll = function() {{
  for(let k in selected) delete selected[k];
  for(let k in staged) delete staged[k];
  renderAll(); updateSelCount();
}}
renderAll();
</script>
</body>
</html>"""

# -------------------------------------------------
# UI COMPONENTS
# -------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def analyze_image_quality_cached(url: str) -> List[str]:
    if not url or not str(url).startswith("http"): return []
    warnings = []
    try:
        resp = requests.get(url, timeout=1, stream=True)
        if resp.status_code == 200:
            img = Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300: warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5: warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6: warnings.append("Wide Aspect")
    except Exception:
        pass
    return warnings

def _clear_flag_df_selection(title: str):
    if f"df_{title}" in st.session_state:
        del st.session_state[f"df_{title}"]

@st.dialog("Confirm Bulk Approval")
def bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button(_t("approve_btn"), type="primary", use_container_width=True):
        with st.spinner("Processing..."):
            data_hash = df_hash(subset_data) + country_validator.code + "_skip_" + title
            new_report, _ = cached_validate_products(data_hash, subset_data, support_files, country_validator.code, data_has_warranty_cols_check, skip_validators=[title])
            msg_moved, msg_approved = {}, 0
            
            for sid in sids_to_process:
                new_row = new_report[new_report['ProductSetSid'] == sid]
                if new_row.empty or not str(new_row.iloc[0]['FLAG']):
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Approved', '', '', 'Approved by User']
                    msg_approved += 1
                else:
                    new_flag = str(new_row.iloc[0]['FLAG'])
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'] == sid, ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', new_row.iloc[0]['Reason'], new_row.iloc[0]['Comment'], new_flag]
                    msg_moved[new_flag] = msg_moved.get(new_flag, 0) + 1

            if title == "Wrong Category" and _CAT_MATCHER_AVAILABLE:
                try:
                    _engine = _get_cat_matcher_engine()
                    if _engine is not None:
                        learned_count = 0
                        for sid in sids_to_process:
                            row = subset_data[subset_data['PRODUCT_SET_SID'].astype(str).str.strip() == str(sid)]
                            if row.empty:
                                continue
                            name = str(row.iloc[0].get('NAME', '')).strip()
                            if not name:
                                continue

                            _engine.set_compiled_rules(st.session_state.get('compiled_json_rules', {}))
                            predicted = _engine.get_category_with_boost(name)

                            if predicted and predicted.lower() not in ('nan', 'none', 'uncategorized', ''):
                                _engine.apply_learned_correction(name, predicted, auto_save=False)
                                learned_count += 1
                        
                        if learned_count:
                            _engine.save_learning_db()
                            if _CAT_MATCHER_AVAILABLE and hasattr(_engine, '_retrain_correction_classifier'):
                                try:
                                    _engine._retrain_correction_classifier()
                                except:
                                    pass
                            st.session_state.main_toasts.append(
                                f"🧠 Engine learned {learned_count} correction(s) from your approvals."
                            )
                except Exception as _le:
                    logger.warning("Wrong Category approval learning failed: %s", _le)

            if msg_approved > 0: 
                st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items(): 
                st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")
            
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            st.session_state[f"exp_{title}"] = True
            _clear_flag_df_selection(title)
        st.rerun()

def render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check, support_files, country_validator):
    cache_key = f"display_df_{title}"
    base_display_cols = ['PRODUCT_SET_SID', 'NAME', 'BRAND', 'CATEGORY', 'COLOR', 'GLOBAL_SALE_PRICE', 'GLOBAL_PRICE', 'PARENTSKU', 'SELLER_NAME']
    current_display_cols = base_display_cols.copy()
    
    if title == "Wrong Variation":
        if 'COUNT_VARIATIONS' in data.columns: current_display_cols.append('COUNT_VARIATIONS')
        if 'LIST_VARIATIONS' in data.columns: current_display_cols.append('LIST_VARIATIONS')

    if cache_key not in st.session_state.display_df_cache:
        _extra_cols = [c for c in current_display_cols if c in data.columns]
        if 'CATEGORY_CODE' in data.columns and 'CATEGORY_CODE' not in _extra_cols:
            _extra_cols = _extra_cols + ['CATEGORY_CODE']
        
        df_display = pd.merge(
            df_flagged_sids[['ProductSetSid']],
            data,
            left_on='ProductSetSid', right_on='PRODUCT_SET_SID', how='left'
        )[[c for c in _extra_cols if c in data.columns]]

        _code_to_path = support_files.get('code_to_path', {})
        if _code_to_path and 'CATEGORY_CODE' in df_display.columns:
            df_display['CATEGORY'] = df_display['CATEGORY_CODE'].apply(
                lambda c: _code_to_path.get(str(c).strip()) or df_display.get('CATEGORY', pd.Series(dtype=str)).iloc[0] if pd.notna(c) else ''
            )
            df_display = df_display.drop(columns=['CATEGORY_CODE'])

        df_display = df_display[[c for c in current_display_cols if c in df_display.columns]]
        st.session_state.display_df_cache[cache_key] = df_display
    else:
        df_display = st.session_state.display_df_cache[cache_key]

    c1, c2 = st.columns([1, 1])
    with c1: search_term = st.text_input(_t("search_grid"), placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")

    df_view = df_display.copy()
    if search_term: 
        df_view = df_view[df_view.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: 
        df_view = df_view[df_view['SELLER_NAME'].isin(seller_filter)]
    df_view = df_view.reset_index(drop=True)
    
    if 'NAME' in df_view.columns:
        def strip_html(text): return re.sub('<[^<]+?>', '', text) if isinstance(text, str) else text
        df_view['NAME'] = df_view['NAME'].apply(strip_html)

    if 'GLOBAL_PRICE' in df_view.columns and 'GLOBAL_SALE_PRICE' in df_view.columns:
        def _get_local_p(row):
            sp = row.get('GLOBAL_SALE_PRICE')
            rp = row.get('GLOBAL_PRICE')
            val = sp if pd.notna(sp) and str(sp).strip() != "" else rp
            return format_local_price(val, country_validator.country)
        try:
            loc_idx = df_view.columns.get_loc('GLOBAL_PRICE') + 1
            df_view.insert(loc_idx, 'Local Price', df_view.apply(_get_local_p, axis=1))
        except Exception:
            df_view['Local Price'] = df_view.apply(_get_local_p, axis=1)

    event = st.dataframe(
        df_view, hide_index=True, use_container_width=True, selection_mode="multi-row", on_select="rerun",
        column_config={
            "PRODUCT_SET_SID": st.column_config.TextColumn(pinned=True),
            "NAME": st.column_config.TextColumn(pinned=True),
            "CATEGORY": st.column_config.TextColumn("Full Category", width="large"),
            "GLOBAL_SALE_PRICE": st.column_config.NumberColumn("Sale Price (USD)", format="$%.2f"),
            "GLOBAL_PRICE": st.column_config.NumberColumn("Price (USD)", format="$%.2f"),
            "Local Price": st.column_config.TextColumn(f"Local Price ({country_validator.country})"),
        }, key=f"df_{title}"
    )
    raw_selected_indices = list(event.selection.rows)
    selected_indices = [i for i in raw_selected_indices if i < len(df_view)]
    st.caption(f"{len(selected_indices)} / {len(df_view)} selected")
    has_selection = len(selected_indices) > 0

    _fm = support_files['flags_mapping']
    _reason_options = [
        "Wrong Category", "Restricted brands", "Suspected Fake product", "Seller Not approved to sell Refurb",
        "Product Warranty", "Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys", "Prohibited products", "Unnecessary words in NAME", "Single-word NAME",
        "Generic BRAND Issues", "Fashion brand issues", "BRAND name repeated in NAME", "Wrong Variation",
        "Generic branded products with genuine brands", "Missing COLOR", "Missing Weight/Volume",
        "Incomplete Smartphone Name", "Duplicate product", "Poor images", "Perfume Tester",
        "NG - Gift Card Seller", "NG - Books Seller", "NG - TV Brand Seller",
        "NG - HP Toners Seller", "NG - Apple Seller", "NG - Xmas Tree Seller",
        "NG - Rice Brand Seller", "NG - Powerbank Capacity",
        "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button(_t("approve_btn"), key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids_to_process)]
            _clear_flag_df_selection(title)
            bulk_approve_dialog(sids_to_process, title, subset, data_has_warranty_cols_check, support_files, country_validator)

    with btn_col2:
        with st.popover(_t("reject_as"), use_container_width=True, disabled=not has_selection):
            chosen_reason = st.selectbox("Reason", _reason_options, key=f"rej_reason_dd_{title}", label_visibility="collapsed")
            if chosen_reason == "Other Reason (Custom)":
                custom_comment = st.text_area("Custom comment", placeholder="Type your rejection reason here...", key=f"custom_comment_{title}", height=80)
                if st.button("Apply", key=f"apply_custom_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    final_comment = custom_comment.strip() if custom_comment.strip() else "Other Reason"
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', '1000007 - Other Reason', final_comment, 'Other Reason (Custom)']
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected with custom reason.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.session_state[f"exp_{title}"] = True
                    _clear_flag_df_selection(title)
                    st.rerun()
            else:
                _rinfo = _fm.get(chosen_reason, {'reason': '1000007 - Other Reason', 'en': chosen_reason})
                _rcode = _rinfo['reason']
                _cmt_lang = 'fr' if st.session_state.selected_country == "Morocco" else 'en'
                _rcmt = _rinfo.get(_cmt_lang, _rinfo.get('en'))
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', _rcode, _rcmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")

                    if chosen_reason == "Wrong Category" and title != "Wrong Category" and _CAT_MATCHER_AVAILABLE:
                        try:
                            _engine = _get_cat_matcher_engine()
                            _cats   = support_files.get('categories_names_list', [])
                            if _engine is not None and _cats:
                                if not _engine._tfidf_built:
                                    _engine.build_tfidf_index(_cats)
                                
                                learned_count = 0
                                for sid in to_reject:
                                    prod_row = data[data['PRODUCT_SET_SID'].astype(str).str.strip() == str(sid)]
                                    if prod_row.empty:
                                        continue
                                    name = str(prod_row.iloc[0].get('NAME', '')).strip()
                                    if not name:
                                        continue

                                    _engine.set_compiled_rules(st.session_state.get('compiled_json_rules', {}))
                                    predicted = _engine.get_category_with_boost(name)

                                    if predicted and predicted.lower() not in ('nan', 'none', 'uncategorized', ''):
                                        _engine.apply_learned_correction(name, predicted, auto_save=False)
                                        learned_count += 1
                                
                                if learned_count:
                                    _engine.save_learning_db()
                                    if _CAT_MATCHER_AVAILABLE and hasattr(_engine, '_retrain_correction_classifier'):
                                        try:
                                            _engine._retrain_correction_classifier()
                                        except:
                                            pass
                                    st.session_state.main_toasts.append(
                                        f"🧠 Engine noted {learned_count} missed Wrong Category item(s) for future runs."
                                    )
                        except Exception as _le:
                            logger.warning("Wrong Category manual rejection learning failed: %s", _le)

                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    st.session_state[f"exp_{title}"] = True
                    _clear_flag_df_selection(title)
                    st.rerun()

# ==========================================
# APP INITIALIZATION
# ==========================================
try: support_files = load_support_files_lazy(); st.session_state.support_files = support_files; st.session_state['compiled_json_rules'] = support_files.get('compiled_json_rules', {})
except Exception as e: st.error(f"Failed to load configs: {e}"); st.stop()

def get_image_base64(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as img_file: return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.warning(f"get_image_base64({path}): {e}")
    return ""

logo_base64 = get_image_base64("jumia logo.png") or get_image_base64("jumia_logo.png")
logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='height: 42px; margin-right: 15px;'>" if logo_base64 else "<span class='material-symbols-outlined' style='font-size: 42px; margin-right: 15px;'>verified_user</span>"

st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.3);'><h1 style='color: white; margin: 0; font-size: 36px; display: flex; align-items: center;'>{logo_html}Product Validation Tool</h1></div>""", unsafe_allow_html=True)

with st.sidebar:
    lang_names = list(LANGUAGES.keys())
    current_lang_code = st.session_state.get('ui_lang', 'en')
    current_lang_name = next((k for k, v in LANGUAGES.items() if v == current_lang_code), "English")
    selected_lang_name = st.selectbox("Language / Langue / اللغة", lang_names, index=lang_names.index(current_lang_name))
    new_lang_code = LANGUAGES[selected_lang_name]
    if new_lang_code != current_lang_code:
        st.session_state.ui_lang = new_lang_code
        st.rerun()
    st.markdown("---")
    st.header(_t("system_status"))
    if st.button(_t("clear_cache"), use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.session_state.display_df_cache = {}
        if os.path.exists(PARQUET_CACHE_DIR): shutil.rmtree(PARQUET_CACHE_DIR)
        if os.path.exists(FLAG_CACHE_DIR): shutil.rmtree(FLAG_CACHE_DIR)
        st.rerun()
    st.markdown("---")
    st.header(_t("display_settings"))
    new_mode = "wide" if "Wide" in st.radio("Layout Mode", ["Centered", "Wide"], index=1 if st.session_state.layout_mode == "wide" else 0) else "centered"
    if new_mode != st.session_state.layout_mode: st.session_state.layout_mode = new_mode; st.rerun()

# ==========================================
# SECTION 1: UPLOAD & VALIDATION
# ==========================================
st.header(f":material/upload_file: {_t('upload_files')}", anchor=False)

current_country = st.session_state.get('selected_country', get_default_country())

country_choice = st.segmented_control(
    "Country",
    ["Kenya", "Uganda", "Nigeria", "Ghana", "Morocco"],
    default=current_country,
    key="country_selector",
)

if country_choice and country_choice != current_country:
    st.session_state.selected_country = country_choice
    st.session_state.last_processed_files = None
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.flags_expanded_initialized = False
    if country_choice == "Morocco":
        st.session_state.ui_lang = "fr"
    else:
        st.session_state.ui_lang = "en"
    st.toast(f"Switching to {country_choice}…", icon=":material/public:")
country_validator = CountryValidator(st.session_state.selected_country)

uploaded_files = st.file_uploader("", type=['csv', 'xlsx'], accept_multiple_files=True, key="daily_files")

if uploaded_files:
    st.session_state.cached_uploaded_files = [
        {"name": uf.name, "bytes": uf.read()} for uf in uploaded_files
    ]
elif uploaded_files is not None and len(uploaded_files) == 0:
    st.session_state.cached_uploaded_files = []
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.post_qc_summary = pd.DataFrame()
    st.session_state.post_qc_results = {}
    st.session_state.post_qc_data = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.last_processed_files = "empty"

_files_for_processing = st.session_state.get("cached_uploaded_files", [])

if _files_for_processing:
    current_file_signature = sorted([f["name"] + hashlib.md5(f["bytes"]).hexdigest() for f in _files_for_processing])
    process_signature = str(current_file_signature) + f"_{country_validator.code}"
else:
    process_signature = "empty"

if st.session_state.get('last_processed_files') != process_signature:
    st.session_state.final_report = pd.DataFrame()
    st.session_state.all_data_map = pd.DataFrame()
    st.session_state.post_qc_summary = pd.DataFrame()
    st.session_state.post_qc_results = {}
    st.session_state.post_qc_data = pd.DataFrame()
    st.session_state.file_mode = None
    st.session_state.intersection_sids = set()
    st.session_state.intersection_count = 0
    st.session_state.grid_page = 0
    st.session_state.exports_cache = {}
    st.session_state.display_df_cache = {}
    st.session_state.flags_expanded_initialized = False
    if 'main_bridge_counter' not in st.session_state: st.session_state.main_bridge_counter = 0
    st.session_state.desel_counter = 0
    st.session_state.quick_rejections = {}
    st.session_state.batch_counter = 0
    st.session_state.clear_counter = 0
    st.session_state.ls_processed_flag = False
    st.session_state.ls_read_trigger = 0
    st.session_state.search_active = False
    st.session_state.pre_search_page = 0
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("grid_chk_", "toast_"))]
    for k in keys_to_delete: del st.session_state[k]

    if process_signature == "empty":
        st.session_state.last_processed_files = "empty"
    else:
        _engine_for_cache = _get_cat_matcher_engine() if _CAT_MATCHER_AVAILABLE else None
        _learning_stamp   = str(len(_engine_for_cache.learning_db)) if _engine_for_cache else "0"
        sig_hash = hashlib.md5((process_signature + _learning_stamp).encode()).hexdigest()
        cached_data = load_df_parquet(f"{sig_hash}_data.parquet")
        cached_report = load_df_parquet(f"{sig_hash}_report.parquet")

        if cached_data is not None and cached_report is not None:
            st.session_state.final_report = cached_report
            st.session_state.all_data_map = cached_data
            st.session_state.last_processed_files = process_signature
            st.toast("Loaded from cache", icon=":material/bolt:")
        else:
            try:
                all_dfs = []
                file_sids_sets = []
                detected_modes = []
                for uf in _files_for_processing:
                    from io import BytesIO as _BytesIO
                    _buf = _BytesIO(uf["bytes"])
                    if uf["name"].endswith('.xlsx'):
                        raw_data = pd.read_excel(_buf, engine='openpyxl', dtype=str)
                    else:
                        raw_data = _detect_and_read_csv(_buf)
                    raw_data = _repair_mojibake(raw_data)
                    detected_modes.append(detect_file_type(raw_data))
                    all_dfs.append(raw_data)

                file_mode = detected_modes[0] if detected_modes else 'pre_qc'
                st.session_state.file_mode = file_mode

                if file_mode == 'post_qc':
                    st.info(
                        "Post-QC file detected. "
                        "Please use the **Post-QC** page in the sidebar to process this file.",
                        icon=":material/fact_check:",
                    )
                    st.session_state.last_processed_files = process_signature
                else:
                    std_dfs = []
                    for raw_data in all_dfs:
                        std_data = standardize_input_data(raw_data)
                        if 'PRODUCT_SET_SID' in std_data.columns:
                            std_data['PRODUCT_SET_SID'] = std_data['PRODUCT_SET_SID'].astype(str).str.strip()
                            file_sids_sets.append(set(std_data['PRODUCT_SET_SID'].unique()))
                        std_dfs.append(std_data)
                    merged_data = pd.concat(std_dfs, ignore_index=True)
                    if len(file_sids_sets) > 1: st.session_state.intersection_sids = set.intersection(*file_sids_sets)
                    else: st.session_state.intersection_sids = set()
                    st.session_state.intersection_count = len(st.session_state.intersection_sids)
                    data_prop = propagate_metadata(merged_data)
                    is_valid, errors = validate_input_schema(data_prop)
                    if is_valid:
                        data_filtered, det_names = filter_by_country(data_prop, country_validator)
                        if data_filtered.empty:
                            st.error(f"No {country_validator.country} products found. Detected countries: {', '.join(det_names) if det_names else 'None'}", icon=":material/error:")
                            st.stop()
                        actual_counts = data_filtered.groupby('PRODUCT_SET_SID')['PRODUCT_SET_SID'].transform('count')
                        if 'COUNT_VARIATIONS' in data_filtered.columns:
                            file_counts = pd.to_numeric(data_filtered['COUNT_VARIATIONS'], errors='coerce').fillna(1)
                            data_filtered['COUNT_VARIATIONS'] = actual_counts.combine(file_counts, max)
                        else:
                            data_filtered['COUNT_VARIATIONS'] = actual_counts
                        data = data_filtered.drop_duplicates(subset=['PRODUCT_SET_SID'], keep='first')
                        if '_IS_MULTI_COUNTRY' not in data.columns: data['_IS_MULTI_COUNTRY'] = False
                        data_has_warranty = all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION'])
                        for c in ['NAME', 'BRAND', 'COLOR', 'SELLER_NAME', 'CATEGORY_CODE', 'LIST_VARIATIONS']:
                            if c in data.columns: data[c] = data[c].astype(str).fillna('')
                        if 'COLOR_FAMILY' not in data.columns: data['COLOR_FAMILY'] = ""

                        data_hash = df_hash(data) + country_validator.code
                        final_report, _ = cached_validate_products(data_hash, data, support_files, country_validator.code, data_has_warranty)

                        st.session_state.final_report = final_report
                        st.session_state.all_data_map = data
                        st.session_state.last_processed_files = process_signature

                        save_df_parquet(data, f"{sig_hash}_data.parquet")
                        save_df_parquet(final_report, f"{sig_hash}_report.parquet")
                    else:
                        for e in errors: st.error(e)
                        st.session_state.last_processed_files = "error"
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.code(traceback.format_exc())
                st.session_state.last_processed_files = "error"

_bridge_val = st.text_input(
    "jtbridge", value="",
    placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE",
    key=f"main_bridge_{st.session_state.main_bridge_counter}",
    label_visibility="collapsed",
)
if _bridge_val:
    try:
        _msg = json.loads(_bridge_val)
        if _msg.get("action") == "reject":
            _payload = _msg.get("payload", {})
            if isinstance(_payload, dict) and _payload:
                _rgroups: dict = {}
                for _sid, _rkey in _payload.items():
                    _rgroups.setdefault(_rkey, []).append(_sid)
                _total = 0
                for _rkey, _sids in _rgroups.items():
                    _flag = REASON_MAP.get(_rkey, "Other Reason (Custom)")
                    _rinfo = support_files["flags_mapping"].get(_flag, {'reason': "1000007 - Other Reason", 'en': "Manual rejection"})
                    _code = _rinfo['reason']
                    _cmt_lang = 'fr' if st.session_state.selected_country == "Morocco" else 'en'
                    _cmt = _rinfo.get(_cmt_lang, _rinfo.get('en'))
                    st.session_state.final_report.loc[
                        st.session_state.final_report["ProductSetSid"].isin(_sids),
                        ["Status", "Reason", "Comment", "FLAG"]
                    ] = ["Rejected", _code, _cmt, _flag]
                    for _s in _sids:
                        st.session_state.quick_rejections[_s] = _flag
                    _total += len(_sids)
                st.session_state.exports_cache.clear()
                st.session_state.display_df_cache.clear()
                st.session_state.main_toasts.append((f"Rejected {_total} product(s)", ":material/block:"))
                st.session_state.main_bridge_counter += 1
                st.session_state.do_scroll_top = False
                st.rerun()

        elif _msg.get("action") == "undo":
            _payload = _msg.get("payload", {})
            _total_restored = 0
            if isinstance(_payload, dict):
                for _sid in _payload.keys():
                    restore_single_item(_sid)
                    _total_restored += 1
            if _total_restored > 0:
                st.session_state.main_bridge_counter += 1
                st.session_state.do_scroll_top = False
                st.rerun()

    except Exception as _e:
        logger.error(f"Bridge parse error: {_e}")

# ==========================================
# RESULTS SECTION
# ==========================================
if _files_for_processing and not st.session_state.final_report.empty and st.session_state.file_mode != 'post_qc':
    fr = st.session_state.final_report
    data = st.session_state.all_data_map
    app_df = fr[fr['Status'] == 'Approved']
    rej_df = fr[fr['Status'] == 'Rejected']

    st.header(f":material/bar_chart: {_t('val_results')}", anchor=False)

    with st.container(border=True):
        cols = st.columns(5 if st.session_state.layout_mode == "wide" else 3)
        is_nigeria = st.session_state.get('selected_country') == 'Nigeria'
        multi_count = int(data['_IS_MULTI_COUNTRY'].sum()) if '_IS_MULTI_COUNTRY' in data.columns else 0

        metrics_config = [
            (_t("total_prod"),  len(data),                                                                                                   JUMIA_COLORS['dark_gray']),
            (_t("approved"),    len(app_df),                                                                                                 JUMIA_COLORS['success_green']),
            (_t("rejected"),    len(rej_df),                                                                                                 JUMIA_COLORS['jumia_red']),
            (_t("rej_rate"),    f"{(len(rej_df)/len(data)*100) if len(data)>0 else 0:.1f}%",                                                 JUMIA_COLORS['primary_orange']),
            (_t("multi_skus") if is_nigeria else _t("common_skus"), multi_count if is_nigeria else st.session_state.intersection_count, JUMIA_COLORS['warning_yellow'] if is_nigeria else JUMIA_COLORS['medium_gray']),
        ]
        for i, (label, value, color) in enumerate(metrics_config):
            with cols[i % len(cols)]:
                st.markdown(
                    f"<div style='height:5px;background:{color};border-radius:6px 6px 0 0;'></div>",
                    unsafe_allow_html=True
                )
                st.metric(label=label, value=value)

    st.subheader(f":material/flag: {_t('flags_breakdown')}", anchor=False)
    if not rej_df.empty:
        if not st.session_state.flags_expanded_initialized and not rej_df.empty:
            top_flag = rej_df['FLAG'].value_counts().index[0]
            st.session_state[f"exp_{top_flag}"] = True
            st.session_state.flags_expanded_initialized = True

        for title in rej_df['FLAG'].unique():
            df_flagged = rej_df[rej_df['FLAG'] == title]
            with st.expander(f"{title} ({len(df_flagged)})", key=f"exp_{title}"):
                render_flag_expander(title, df_flagged, data, all(c in data.columns for c in ['PRODUCT_WARRANTY', 'WARRANTY_DURATION']), support_files, country_validator)
    else:
        st.success("All products passed validation — no rejections found.")


# ==========================================
# SECTION 2: MANUAL IMAGE REVIEW
# ==========================================
@st.fragment
def render_image_grid():
    if st.session_state.final_report.empty or st.session_state.file_mode == "post_qc":
        return

    st.markdown("---")
    st.header(f":material/pageview: {_t('manual_review')}", anchor=False)

    fr   = st.session_state.final_report
    data = st.session_state.all_data_map

    committed_rej_sids = set(st.session_state.quick_rejections.keys())
    mask          = (fr["Status"] == "Approved") | (fr["ProductSetSid"].isin(committed_rej_sids))
    valid_grid_df = fr[mask]

    c1, c2, c3 = st.columns([1.5, 1.5, 2])
    with c1: search_n  = st.text_input("Search by Name", placeholder="Product name…")
    with c2: search_sc = st.text_input("Search by Seller/Category", placeholder="Seller or Category…")
    with c3:
        st.session_state.grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.grid_items_per_page,
        )

    if 'MAIN_IMAGE' not in data.columns: data['MAIN_IMAGE'] = ''
    available_cols = [c for c in GRID_COLS if c in data.columns]
    if 'CATEGORY_CODE' in data.columns and 'CATEGORY_CODE' not in available_cols:
        available_cols = available_cols + ['CATEGORY_CODE']
    review_data = pd.merge(
        valid_grid_df[["ProductSetSid"]],
        data[available_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )

    _code_to_path = support_files.get('code_to_path', {})
    if _code_to_path and 'CATEGORY_CODE' in review_data.columns:
        review_data = review_data.copy()
        review_data['CATEGORY'] = review_data['CATEGORY_CODE'].apply(
            lambda c: _code_to_path.get(str(c).strip(), str(c)) if pd.notna(c) else ''
        )

    if search_n:
        review_data = review_data[review_data["NAME"].astype(str).str.contains(search_n, case=False, na=False)]
    if search_sc:
        mc = (review_data["CATEGORY"].astype(str).str.contains(search_sc, case=False, na=False) if "CATEGORY" in review_data.columns else pd.Series(False, index=review_data.index))
        ms = review_data["SELLER_NAME"].astype(str).str.contains(search_sc, case=False, na=False)
        review_data = review_data[mc | ms]

    ipp         = st.session_state.grid_items_per_page
    total_pages = max(1, (len(review_data) + ipp - 1) // ipp)
    if st.session_state.grid_page >= total_pages: st.session_state.grid_page = 0

    pg_cols = st.columns([1, 2, 1], vertical_alignment="center")
    with pg_cols[0]:
        if st.button("◀ Prev Page", use_container_width=True, disabled=st.session_state.grid_page == 0):
            st.session_state.grid_page = max(0, st.session_state.grid_page - 1)
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")
    with pg_cols[1]:
        new_page = st.number_input(
            f"Jump to Page (Total: {total_pages} | {len(review_data)} items)",
            min_value=1, max_value=max(1, total_pages),
            value=st.session_state.grid_page + 1, step=1
        )
        if new_page - 1 != st.session_state.grid_page:
            st.session_state.grid_page = new_page - 1
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")
    with pg_cols[2]:
        if st.button("Next Page ▶", use_container_width=True, disabled=st.session_state.grid_page >= total_pages - 1):
            st.session_state.grid_page += 1
            st.session_state.do_scroll_top = True
            st.rerun(scope="fragment")

    page_start = st.session_state.grid_page * ipp
    page_data  = review_data.iloc[page_start : page_start + ipp]

    page_warnings: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        future_to_sid = {
            ex.submit(analyze_image_quality_cached, str(r.get("MAIN_IMAGE", "")).strip()): str(r["PRODUCT_SET_SID"])
            for _, r in page_data.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_sid):
            warns = future.result()
            if warns: page_warnings[future_to_sid[future]] = warns

    rejected_state = {
        sid: st.session_state.quick_rejections[sid]
        for sid in page_data["PRODUCT_SET_SID"].astype(str)
        if sid in st.session_state.quick_rejections
    }

    cols_per_row = 3 if st.session_state.layout_mode == "centered" else 4

    grid_html = build_fast_grid_html(
        page_data, support_files["flags_mapping"],
        st.session_state.selected_country, page_warnings, rejected_state, cols_per_row,
    )
    components.html(grid_html, height=800, scrolling=True)

    if st.session_state.get("do_scroll_top", False):
        components.html(
            "<script>window.parent.document.querySelector('.main').scrollTo({top:0,behavior:'smooth'});</script>",
            height=0,
        )
        st.session_state.do_scroll_top = False


# ==========================================
# SECTION 3: EXPORTS
# ==========================================
@st.fragment
def render_exports_section():
    if st.session_state.final_report.empty or st.session_state.file_mode == 'post_qc':
        return

    fr      = st.session_state.final_report
    data    = st.session_state.all_data_map
    app_df  = fr[fr['Status'] == 'Approved']
    rej_df  = fr[fr['Status'] == 'Rejected']
    c_code  = st.session_state.selected_country[:2].upper()
    date_str = datetime.now().strftime('%Y-%m-%d')
    reasons_df = support_files.get('reasons', pd.DataFrame())

    st.markdown("---")
    st.markdown(f"""<div style='background: linear-gradient(135deg, {JUMIA_COLORS['primary_orange']}, {JUMIA_COLORS['secondary_orange']}); padding: 20px 24px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(246, 139, 30, 0.25);'><h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>{_t('download_reports')}</h2><p style='color: rgba(255,255,255,0.9); margin: 6px 0 0 0; font-size: 13px;'>Export validation results in Excel or ZIP format</p></div>""", unsafe_allow_html=True)

    exports_config = [
        ("PIM Export",    fr,     'Complete validation report with all statuses', lambda df: generate_smart_export(df, f"{c_code}_PIM_Export_{date_str}", 'simple', reasons_df)),
        ("Rejected Only", rej_df, 'Products that failed validation',              lambda df: generate_smart_export(df, f"{c_code}_Rejected_{date_str}", 'simple', reasons_df)),
        ("Approved Only", app_df, 'Products that passed validation',              lambda df: generate_smart_export(df, f"{c_code}_Approved_{date_str}", 'simple', reasons_df)),
        ("Full Data",     data,   'Complete dataset with validation flags',       lambda df: generate_smart_export(prepare_full_data_merged(df, fr), f"{c_code}_Full_{date_str}", 'full')),
    ]

    all_cached = all(title in st.session_state.exports_cache for title, _, _, _ in exports_config)

    if all_cached:
        st.success("All reports generated and ready to download.", icon=":material/check_circle:")
    else:
        if st.button("Generate All Reports", type="primary", icon=":material/download:", use_container_width=True):
            with st.spinner("Generating all reports…"):
                for t2, d2, _desc2, f2 in exports_config:
                    if t2 not in st.session_state.exports_cache:
                        res, fname, mime = f2(d2)
                        st.session_state.exports_cache[t2] = {"data": res.getvalue(), "fname": fname, "mime": mime}
            st.rerun()

    cols_count = 4 if st.session_state.layout_mode == "wide" else 2
    for i in range(0, len(exports_config), cols_count):
        cols = st.columns(cols_count)
        for j, col in enumerate(cols):
            if i + j < len(exports_config):
                title, df, desc, func = exports_config[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"""<div style='text-align:center;margin-bottom:15px;'><div style='font-size:18px;font-weight:700;'>{title}</div><div style='font-size:11px;margin-top:4px;opacity:0.7;'>{desc}</div><div style='background:{JUMIA_COLORS['light_gray']};color:{JUMIA_COLORS['primary_orange']};padding:8px;border-radius:6px;margin-top:12px;font-weight:600;'>{len(df):,} rows</div></div>""", unsafe_allow_html=True)

                        if title not in st.session_state.exports_cache:
                            if st.button("Generate", key=f"gen_{title}", type="primary", use_container_width=True, icon=":material/download:"):
                                with st.spinner("Generating all reports…"):
                                    for t2, d2, _desc2, f2 in exports_config:
                                        if t2 not in st.session_state.exports_cache:
                                            res, fname, mime = f2(d2)
                                            st.session_state.exports_cache[t2] = {"data": res.getvalue(), "fname": fname, "mime": mime}
                                st.rerun()
                        else:
                            cache = st.session_state.exports_cache[title]
                            st.download_button("Download", data=cache["data"], file_name=cache["fname"], mime=cache["mime"], use_container_width=True, type="primary", icon=":material/file_download:", key=f"dl_{title}")
                            if st.button("Clear", key=f"clr_{title}", use_container_width=True):
                                del st.session_state.exports_cache[title]
                                st.rerun()

# ==========================================
# CALL FRAGMENTS
# ==========================================
render_image_grid()
render_exports_section()
