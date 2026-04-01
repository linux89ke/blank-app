import sys
import os
import re
import hashlib
import traceback
import logging
import json
import concurrent.futures
from io import BytesIO

# ------------------------------------------------------------------
# PATH FIX — must be first, before any local imports.
# ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    import requests as _requests
    from PIL import Image as _Image
    _IMAGE_CHECK_OK = True
except ImportError:
    _IMAGE_CHECK_OK = False

# ------------------------------------------------------------------
# LOCAL IMPORTS
# ------------------------------------------------------------------
try:
    from translations import LANGUAGES, get_translation
    _TRANSLATIONS_OK = True
except ImportError:
    _TRANSLATIONS_OK = False
    def get_translation(lang, key): return key

try:
    from postqc import (
        detect_file_type,
        normalize_post_qc,
        load_category_map,
    )
    _POSTQC_OK = True
except Exception as _postqc_err:
    _POSTQC_OK = False
    _postqc_err_msg = str(_postqc_err)

try:
    import _preqc_registry  # noqa: F401
except ImportError:
    pass

try:
    import importlib as _importlib
    _main_mod = (
        sys.modules.get("streamlit_app")
        or _importlib.import_module("streamlit_app")
    )
    validate_products        = _main_mod.validate_products
    load_all_support_files   = _main_mod.load_all_support_files
    CountryValidator         = _main_mod.CountryValidator
    PRODUCTSETS_COLS         = _main_mod.PRODUCTSETS_COLS
    REJECTION_REASONS_COLS   = _main_mod.REJECTION_REASONS_COLS
    clean_category_code      = _main_mod.clean_category_code
    df_hash                  = _main_mod.df_hash
    cached_validate_products = _main_mod.cached_validate_products
    # Image grid helpers
    build_fast_grid_html            = _main_mod.build_fast_grid_html
    analyze_image_quality_cached    = _main_mod.analyze_image_quality_cached
    format_local_price              = _main_mod.format_local_price
    GRID_COLS                       = _main_mod.GRID_COLS
    REASON_MAP                      = _main_mod.REASON_MAP
    _FULL_VALIDATION_OK    = True
except Exception as _fv_err:
    _FULL_VALIDATION_OK = False
    _fv_err_msg = str(_fv_err)
    REASON_MAP = {
        "REJECT_POOR_IMAGE": "Poor images",
        "REJECT_WRONG_CAT":  "Wrong Category",
        "REJECT_FAKE":       "Suspected Fake product",
        "REJECT_BRAND":      "Restricted brands",
        "REJECT_PROHIBITED": "Prohibited products",
        "REJECT_COLOR":      "Missing COLOR",
        "REJECT_WRONG_BRAND":"Generic branded products with genuine brands",
        "OTHER_CUSTOM":      "Other Reason (Custom)",
    }
    GRID_COLS = ['PRODUCT_SET_SID','NAME','BRAND','CATEGORY','SELLER_NAME',
                 'MAIN_IMAGE','GLOBAL_SALE_PRICE','GLOBAL_PRICE','COLOR']

# ── Post-QC aware brand-check overrides ───────────────────────────────────────
if _FULL_VALIDATION_OK:
    import pandas as _pd_patch

    def _pq_check_fashion_brand_issues(data: "pd.DataFrame",
                                        valid_category_codes_fas: list) -> "pd.DataFrame":
        if not {"CATEGORY_CODE", "BRAND"}.issubset(data.columns):
            return _pd_patch.DataFrame(columns=data.columns)
        fas_set = set(clean_category_code(c) for c in valid_category_codes_fas)
        brand_is_fashion = data["BRAND"].astype(str).str.strip().str.lower() == "fashion"
        code_not_fas     = ~data["CATEGORY_CODE"].apply(clean_category_code).isin(fas_set)
        if "CATEGORY" in data.columns:
            path_is_fashion = data["CATEGORY"].astype(str).str.contains(
                r"\bfashion\b", case=False, na=False, regex=True
            )
        else:
            path_is_fashion = _pd_patch.Series(False, index=data.index)
        flagged = data[brand_is_fashion & code_not_fas & ~path_is_fashion].copy()
        return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

    def _pq_check_generic_brand_issues(data: "pd.DataFrame",
                                        valid_category_codes_fas: list) -> "pd.DataFrame":
        if not {"CATEGORY_CODE", "BRAND"}.issubset(data.columns):
            return _pd_patch.DataFrame(columns=data.columns)
        fas_set = set(clean_category_code(c) for c in valid_category_codes_fas)
        brand_is_generic = data["BRAND"].astype(str).str.lower() == "generic"
        code_in_fas      = data["CATEGORY_CODE"].apply(clean_category_code).isin(fas_set)
        if "CATEGORY" in data.columns:
            path_is_fashion = data["CATEGORY"].astype(str).str.contains(
                r"\bfashion\b", case=False, na=False, regex=True
            )
        else:
            path_is_fashion = _pd_patch.Series(False, index=data.index)
        flagged = data[brand_is_generic & (code_in_fas | path_is_fashion)].copy()
        return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

    _main_mod.check_fashion_brand_issues  = _pq_check_fashion_brand_issues
    _main_mod.check_generic_brand_issues  = _pq_check_generic_brand_issues


# ── READ-ONLY Expander logic (No Firebase Learning) ───────────────────────────
@st.dialog("Confirm Bulk Approval (Post-QC)")
def pq_bulk_approve_dialog(sids_to_process, title, subset_data, data_has_warranty_cols_check, support_files, country_validator):
    st.warning(f"You are about to approve **{len(sids_to_process)}** items from `{title}`.")
    if st.button("Approve", type="primary", use_container_width=True):
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

            if msg_approved > 0: st.session_state.main_toasts.append(f"{msg_approved} items successfully Approved!")
            for flag, count in msg_moved.items(): st.session_state.main_toasts.append(f"{count} items re-flagged as: {flag}")
            st.session_state.exports_cache.clear()
            st.session_state.display_df_cache.clear()
            _key = "pqexp_" + re.sub(r"[^a-zA-Z0-9]", "_", title)
            st.session_state[_key] = True
            if f"df_{title}" in st.session_state: del st.session_state[f"df_{title}"]
        st.rerun()

def pq_render_flag_expander(title, df_flagged_sids, data, data_has_warranty_cols_check, support_files, country_validator):
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
    with c1: search_term = st.text_input("Search", placeholder="Name, Brand...", key=f"s_{title}")
    with c2: seller_filter = st.multiselect("Filter by Seller", sorted(df_display['SELLER_NAME'].astype(str).unique()), key=f"f_{title}")

    df_view = df_display.copy()
    if search_term: df_view = df_view[df_view.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)]
    if seller_filter: df_view = df_view[df_view['SELLER_NAME'].isin(seller_filter)]
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

    _fm = support_files.get('flags_mapping', {})
    _reason_options = [
        "Wrong Category", "Restricted brands", "Suspected Fake product", "Seller Not approved to sell Refurb",
        "Product Warranty", "Seller Approve to sell books", "Seller Approved to Sell Perfume", "Counterfeit Sneakers",
        "Suspected counterfeit Jerseys", "Prohibited products", "Unnecessary words in NAME", "Single-word NAME",
        "Generic BRAND Issues", "Fashion brand issues", "BRAND name repeated in NAME", "Wrong Variation",
        "Generic branded products with genuine brands", "Missing COLOR", "Missing Weight/Volume",
        "Incomplete Smartphone Name", "Duplicate product", "Poor images", "Perfume Tester", "Other Reason (Custom)",
    ]

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Approve", key=f"approve_sel_{title}", type="primary", use_container_width=True, disabled=not has_selection):
            sids_to_process = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
            subset = data[data['PRODUCT_SET_SID'].isin(sids_to_process)]
            if f"df_{title}" in st.session_state: del st.session_state[f"df_{title}"]
            pq_bulk_approve_dialog(sids_to_process, title, subset, data_has_warranty_cols_check, support_files, country_validator)

    with btn_col2:
        with st.popover("Reject As", use_container_width=True, disabled=not has_selection):
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
                    _key = "pqexp_" + re.sub(r"[^a-zA-Z0-9]", "_", title)
                    st.session_state[_key] = True
                    if f"df_{title}" in st.session_state: del st.session_state[f"df_{title}"]
                    st.rerun()
            else:
                _rinfo = _fm.get(chosen_reason, {'reason': '1000007 - Other Reason', 'en': chosen_reason})
                _rcode = _rinfo['reason']
                _cmt = _rinfo.get('en', chosen_reason)
                st.caption(f"Code: {_rcode[:40]}...")
                if st.button("Apply", key=f"apply_dd_{title}", type="primary", use_container_width=True, disabled=not has_selection):
                    to_reject = df_view.iloc[selected_indices]['PRODUCT_SET_SID'].tolist()
                    st.session_state.final_report.loc[st.session_state.final_report['ProductSetSid'].isin(to_reject), ['Status', 'Reason', 'Comment', 'FLAG']] = ['Rejected', _rcode, _cmt, chosen_reason]
                    st.session_state.main_toasts.append(f"{len(to_reject)} items rejected as '{chosen_reason}'.")
                    st.session_state.exports_cache.clear()
                    st.session_state.display_df_cache.clear()
                    _key = "pqexp_" + re.sub(r"[^a-zA-Z0-9]", "_", title)
                    st.session_state[_key] = True
                    if f"df_{title}" in st.session_state: del st.session_state[f"df_{title}"]
                    st.rerun()
# ─────────────────────────────────────────────────────────────────────────────

try:
    from jumia_scraper import (
        enrich_post_qc_df,
        SCRAPABLE_FIELDS,
        scrape_single_sku,
        COUNTRY_URLS,
    )
    _SCRAPER_OK = True
except ImportError:
    _SCRAPER_OK = False

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Post-QC Validation",
        page_icon="QC",
        layout=st.session_state.get("layout_mode", "wide"),
    )
except Exception:
    pass

# ------------------------------------------------------------------
# THEME — Enterprise SaaS palette
# Orange used only as accent; neutral dark/light base; Inter font.
# ------------------------------------------------------------------
ORANGE  = "#F68B1E"
ORANGE2 = "#E07A10"
RED     = "#D93025"
DARK    = "#111827"
MED     = "#6B7280"
LIGHT   = "#F9FAFB"
BORDER  = "#E5E7EB"
GREEN   = "#16A34A"
BLUE    = "#1D4ED8"

# Lucide-style SVG icons (inline, 16×16, currentColor)
_ICON = {
    "search":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>',
    "shield":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "file":    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    "alert":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    "globe":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "upload":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/><path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/></svg>',
    "download":'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "check":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    "x":       '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    "trash":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
    "image":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>',
    "flag":    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>',
    "bar":     '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, .stMarkdown, .stButton > button, .stSelectbox, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"], .stTextInput input, .stTextArea textarea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
    border: 1px solid {BORDER};
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"] {{
    background-color: {ORANGE} !important;
    border-color: {ORANGE} !important;
    color: white !important;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {ORANGE2} !important;
    border-color: {ORANGE2} !important;
}}

/* ── Metrics ── */
div[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px 20px;
}}
div[data-testid="stMetricValue"] {{
    color: {DARK};
    font-weight: 700;
    font-size: 28px !important;
    letter-spacing: -0.5px;
}}
div[data-testid="stMetricLabel"] {{
    color: {MED};
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Expanders ── */
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    box-shadow: none;
}}
div[data-testid="stExpander"] summary {{
    background-color: {LIGHT};
    padding: 12px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 13px;
}}

/* ── Headings ── */
h1 {{ color: {DARK} !important; font-size: 22px !important; font-weight: 700 !important; }}
h2 {{ color: {DARK} !important; font-size: 18px !important; font-weight: 600 !important; }}
h3 {{ color: {DARK} !important; font-size: 15px !important; font-weight: 600 !important; }}

/* ── Enrich badge ── */
.enrich-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    background: {GREEN};
    color: white;
    margin-left: 8px;
    vertical-align: middle;
}}

/* ── Field chips ── */
.field-chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    background: #F0FDF4;
    color: #166534;
    border: 1px solid #BBF7D0;
    margin: 2px 3px 2px 0;
}}
.field-chip.missing {{
    background: #FFF7ED;
    color: #9A3412;
    border-color: #FED7AA;
}}

/* ── Section headers ── */
.pq-section-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 14px 18px;
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-left: 3px solid {ORANGE};
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
    color: {DARK};
    font-size: 14px;
    font-weight: 600;
}}
.pq-section-header svg {{ color: {ORANGE}; flex-shrink: 0; }}

/* ── Metric accent bar ── */
.metric-accent {{
    height: 3px;
    border-radius: 3px 3px 0 0;
    margin-bottom: -1px;
}}

@media (prefers-color-scheme: dark) {{
    div[data-testid="stMetricValue"] {{ color: #F9FAFB !important; }}
    div[data-testid="stMetric"]      {{ background: #1F2937 !important; border-color: #374151 !important; }}
    h1, h2, h3                       {{ color: #F9FAFB !important; }}
    .pq-section-header               {{ background: #1F2937; border-color: #374151; color: #F9FAFB; }}
}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# SESSION STATE DEFAULTS
# ------------------------------------------------------------------
_SS_DEFAULTS = {
    "ui_lang":            "en",
    "selected_country":   "Kenya",
    "layout_mode":        "wide",
    "pq_country":         "Kenya",
    "pq_data":            pd.DataFrame(),
    "pq_last_sig":        None,
    "enriched_df":        None,
    "enrichment_summary": {},
    "enrichment_done":    False,
    "pq_val_report":      pd.DataFrame(),
    "pq_val_results":     {},
    "pq_val_exports":     {},
    "pq_flags_init":      False,
    "display_df_cache":   {},
    "exports_cache":      {},
    "final_report":       pd.DataFrame(),
    "main_toasts":        [],
    "flags_expanded_initialized": False,
    # image grid state
    "pq_grid_page":          0,
    "pq_grid_items_per_page": 50,
    "pq_bridge_counter":     0,
    "pq_do_scroll_top":      False,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ------------------------------------------------------------------
# CANONICAL FIELD LIST
# ------------------------------------------------------------------
_ALL_FIELDS = [
    "PRICE", "DISCOUNT", "STOCK_STATUS",
    "MODEL", "GTIN", "BRAND", "CATEGORY_PATH", "IS_OFFICIAL_STORE",
    "MAIN_IMAGE", "WEIGHT",
    "COLOR", "COLOR_IN_TITLE", "COUNT_VARIATIONS", "SIZES_AVAILABLE",
    "PRODUCT_WARRANTY", "WARRANTY_DURATION",
    "RATING", "REVIEW_COUNT",
    "DESCRIPTION", "KEY_FEATURES", "KEY_SPECS", "SPECIFICATIONS",
    "WHATS_IN_BOX",
]

# ------------------------------------------------------------------
# COLUMN MAP
# ------------------------------------------------------------------
_PQ_COL_MAP = {
    "sku":               "PRODUCT_SET_SID",
    "name":              "NAME",
    "brand":             "BRAND",
    "category":          "CATEGORY",
    "categories":        "CATEGORY",
    "price":             "GLOBAL_SALE_PRICE",
    "newprice":          "GLOBAL_SALE_PRICE",
    "old price":         "GLOBAL_PRICE",
    "oldprice":          "GLOBAL_PRICE",
    "seller":            "SELLER_NAME",
    "sellername":        "SELLER_NAME",
    "seller name":       "SELLER_NAME",
    "image url":         "MAIN_IMAGE",
    "image":             "MAIN_IMAGE",
    "main image":        "MAIN_IMAGE",
    "url":               "PRODUCT_URL",
    "product url":       "PRODUCT_URL",
    "stock":             "STOCK_STATUS",
    "stock status":      "STOCK_STATUS",
    "rating":            "RATING",
    "averagerating":     "RATING",
    "average rating":    "RATING",
    "total ratings":     "REVIEW_COUNT",
    "totalratings":      "REVIEW_COUNT",
    "review count":      "REVIEW_COUNT",
    "discount":          "DISCOUNT",
    "tags":              "TAGS",
    "jumia express":     "JUMIA_EXPRESS",
    "isexpress":         "JUMIA_EXPRESS",
    "shop global":       "SHOP_GLOBAL",
    "isglobal":          "SHOP_GLOBAL",
    "color":             "COLOR",
    "colour":            "COLOR",
    "product warranty":  "PRODUCT_WARRANTY",
    "warranty":          "PRODUCT_WARRANTY",
    "warranty duration": "WARRANTY_DURATION",
    "count variations":  "COUNT_VARIATIONS",
    "variations":        "COUNT_VARIATIONS",
    "seller sku":        "SELLER_SKU",
    "parentsku":         "PARENTSKU",
    "parent sku":        "PARENTSKU",
    "description":       "DESCRIPTION",
}

def _standardise_pq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        mapped = _PQ_COL_MAP.get(col.lower().strip())
        if mapped:
            rename[col] = mapped
    return df.rename(columns=rename)

COUNTRY_CODES = {
    "Kenya":   "KE",
    "Uganda":  "UG",
    "Nigeria": "NG",
    "Ghana":   "GH",
    "Morocco": "MA",
}
COUNTRIES = list(COUNTRY_CODES.keys())

def _t(key): return get_translation(st.session_state.ui_lang, key)
def _code(name: str) -> str: return COUNTRY_CODES.get(name, "KE")

def _load_support_files() -> dict:
    if "support_files" in st.session_state:
        return st.session_state.support_files
    if _FULL_VALIDATION_OK:
        try:
            sf = load_all_support_files()
            st.session_state.support_files = sf
            return sf
        except Exception:
            pass
    cat_map = load_category_map() if _POSTQC_OK else {}
    return {"category_map": cat_map}

def _reset_results():
    st.session_state.pq_data            = pd.DataFrame()
    st.session_state.pq_last_sig        = None
    st.session_state.enriched_df        = None
    st.session_state.enrichment_summary = {}
    st.session_state.enrichment_done    = False
    st.session_state.pq_val_report      = pd.DataFrame()
    st.session_state.pq_val_results     = {}
    st.session_state.pq_val_exports     = {}
    st.session_state.pq_flags_init      = False
    st.session_state.final_report       = pd.DataFrame()
    # image grid
    st.session_state.pq_grid_page       = 0
    st.session_state.pq_bridge_counter  = 0
    st.session_state.pq_do_scroll_top   = False
    # clear per-product rejection keys from previous run
    _keys_to_del = [k for k in st.session_state.keys()
                    if k.startswith("pq_qrej_")]
    for _k in _keys_to_del:
        del st.session_state[_k]

def _count_filled(series: pd.Series) -> int:
    return int(
        series.astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .sum()
    )

def _build_enrichment_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    summary = {}
    for col in SCRAPABLE_FIELDS:
        b_count = _count_filled(before[col]) if col in before.columns else 0
        a_count = _count_filled(after[col])  if col in after.columns  else 0
        summary[col] = {
            "before": b_count,
            "after":  a_count,
            "filled": max(0, a_count - b_count),
        }
    return summary

def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Post QC Results")
    return buf.getvalue()

def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def _clean_sku(raw: str) -> str:
    return raw.split("-")[0].strip()

def _extract_sku_from_url(url: str) -> str | None:
    slug = re.sub(r"\.html.*$", "", url.rstrip("/").split("/")[-1])
    parts = slug.split("-")
    for part in reversed(parts):
        if not (12 <= len(part) <= 30): continue
        if not re.search(r"[A-Za-z]", part) or not re.search(r"[0-9]", part): continue
        if re.fullmatch(r"[\dA-Z]*(ML|KG|GB|TB|CM|MM|GM|MG|PC|PCS|L)+[\dA-Z]*", part, re.IGNORECASE): continue
        if re.search(r"(NAFAMZ|GANAFAMZ|HANAFAMZ|FANAFAMZ)$", part, re.IGNORECASE):
            return part.upper()
    for part in reversed(parts):
        if not (12 <= len(part) <= 30): continue
        if not re.search(r"[A-Za-z]", part) or not re.search(r"[0-9]", part): continue
        if re.fullmatch(r"[\dA-Z]*(ML|KG|GB|TB|CM|MM|GM|MG|PC|PCS|L)+[\dA-Z]*", part, re.IGNORECASE): continue
        return part.upper()
    return None

def _auto_detect_country_from_url(url: str) -> str | None:
    _domain_map = {
        "jumia.co.ke": "KE", "jumia.ug": "UG", "jumia.com.ng": "NG",
        "jumia.com.gh": "GH", "jumia.ma": "MA",
    }
    for domain, code in _domain_map.items():
        if domain in url:
            return code
    return None

def _parse_lookup_inputs(raw_text: str) -> list[dict]:
    entries = []
    seen_skus: set[str] = set()
    for line in raw_text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("http"):
            sku = _extract_sku_from_url(line)
            if not sku:
                slug = re.sub(r"\.html.*$", "", line.rstrip("/").split("/")[-1])
                num_m = re.search(r"(\d{6,})$", slug)
                sku = num_m.group(1) if num_m else slug[:20]
            hint = _auto_detect_country_from_url(line)
            direct_url = line
        else:
            sku = _clean_sku(line)
            hint = None
            direct_url = None
        if sku and sku not in seen_skus:
            seen_skus.add(sku)
            entries.append({
                "input": line, "sku": sku, "url": direct_url, "country_hint": hint,
            })
    return entries

# ------------------------------------------------------------------
# NEW HELPERS: data-quality checks
# ------------------------------------------------------------------

def _derive_official_store_from_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive IS_OFFICIAL_STORE from the TAGS column when it is absent.
    Any product tagged JMALL is an official-store listing.
    Only writes to IS_OFFICIAL_STORE if the column is fully absent or all-blank.
    """
    if "TAGS" not in df.columns:
        return df
    col_present = (
        "IS_OFFICIAL_STORE" in df.columns
        and df["IS_OFFICIAL_STORE"].astype(str).str.strip()
            .replace({"nan": "", "None": ""}).ne("").any()
    )
    if col_present:
        return df
    df = df.copy()
    df["IS_OFFICIAL_STORE"] = df["TAGS"].str.contains(
        "JMALL", case=False, na=False
    ).map({True: "Yes", False: "No"})
    return df


def _calc_discount_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    For Data-Grab uploads (and any file missing DISCOUNT), calculate the
    discount percentage from GLOBAL_PRICE and GLOBAL_SALE_PRICE.
    Skips rows that already have a value.
    """
    if "DISCOUNT" not in df.columns:
        df = df.copy()
        df["DISCOUNT"] = ""
    else:
        df = df.copy()

    missing = df["DISCOUNT"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("")
    if not missing.any():
        return df

    op = pd.to_numeric(
        df.get("GLOBAL_PRICE", pd.Series(dtype=float)), errors="coerce"
    )
    sp = pd.to_numeric(
        df.get("GLOBAL_SALE_PRICE", pd.Series(dtype=float)), errors="coerce"
    )
    mask = missing & (op > 0) & (sp < op)
    df.loc[mask, "DISCOUNT"] = (
        ((op - sp) / op * 100).round(0).astype("Int64").astype(str) + "%"
    )[mask]
    return df


def _flag_corrupted_old_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark OLD_PRICE_CORRUPTED = True when GLOBAL_PRICE is implausibly large
    relative to GLOBAL_SALE_PRICE (>500×). This catches concatenated-number
    artefacts like 82848717 when the real price is ~5000.
    The flag column is used by the discount-integrity check below.
    """
    df = df.copy()
    op = pd.to_numeric(df.get("GLOBAL_PRICE",  pd.Series(dtype=float)), errors="coerce")
    sp = pd.to_numeric(df.get("GLOBAL_SALE_PRICE", pd.Series(dtype=float)), errors="coerce")
    df["_OLD_PRICE_CORRUPTED"] = (op > 0) & (sp > 0) & (op > sp * 500)
    return df


def _flag_discount_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _DISCOUNT_MISMATCH = True when the stated DISCOUNT diverges from
    the value calculated from GLOBAL_PRICE / GLOBAL_SALE_PRICE by more than
    5 percentage points. Rows with corrupted old prices are excluded.
    """
    df = df.copy()
    if "DISCOUNT" not in df.columns:
        df["_DISCOUNT_MISMATCH"] = False
        return df

    op  = pd.to_numeric(df.get("GLOBAL_PRICE",      pd.Series(dtype=float)), errors="coerce")
    sp  = pd.to_numeric(df.get("GLOBAL_SALE_PRICE",  pd.Series(dtype=float)), errors="coerce")
    stated = pd.to_numeric(
        df["DISCOUNT"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    calc = ((op - sp) / op * 100).where((op > 0) & (sp < op))
    corrupted = df.get("_OLD_PRICE_CORRUPTED", pd.Series(False, index=df.index))
    df["_DISCOUNT_MISMATCH"] = (
        stated.notna() & calc.notna() & ~corrupted & (abs(calc - stated) > 5)
    )
    return df


def _flag_brand_in_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _BRAND_IN_TITLE = True when the brand name appears 2+ times in NAME.
    """
    df = df.copy()
    if "BRAND" not in df.columns or "NAME" not in df.columns:
        df["_BRAND_IN_TITLE"] = False
        return df

    def _check(row):
        brand = str(row.get("BRAND", "")).lower().strip()
        name  = str(row.get("NAME",  "")).lower()
        return bool(brand) and name.count(brand) >= 2

    df["_BRAND_IN_TITLE"] = df.apply(_check, axis=1)
    return df


def _flag_rating_no_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark _RATING_NO_REVIEWS = True when a product has a non-zero RATING but
    REVIEW_COUNT is 0 or missing.
    """
    df = df.copy()
    rating  = pd.to_numeric(df.get("RATING",       pd.Series(dtype=float)), errors="coerce")
    reviews = pd.to_numeric(df.get("REVIEW_COUNT",  pd.Series(dtype=float)), errors="coerce")
    df["_RATING_NO_REVIEWS"] = rating.gt(0) & (reviews.isna() | reviews.eq(0))
    return df


def _render_data_quality_flags(df: pd.DataFrame) -> None:
    """
    Render expandable sections for each internal data-quality flag.
    Called after pipeline processing, before the main validation block.
    """
    flag_cfg = [
        (
            "_OLD_PRICE_CORRUPTED",
            "Corrupted original price",
            "Old price is 500× or more the current price — likely two values concatenated by the export system.",
            ["PRODUCT_SET_SID", "NAME", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "DISCOUNT"],
        ),
        (
            "_DISCOUNT_MISMATCH",
            "Discount / price mismatch",
            "Stated discount differs from the calculated (old − new) / old by more than 5 percentage points.",
            ["PRODUCT_SET_SID", "NAME", "GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "DISCOUNT"],
        ),
        (
            "_BRAND_IN_TITLE",
            "Brand duplicated in title",
            "The brand name appears two or more times in the product title.",
            ["PRODUCT_SET_SID", "NAME", "BRAND"],
        ),
        (
            "_RATING_NO_REVIEWS",
            "Rating present, review count missing",
            "Product has a star rating but zero (or missing) review count.",
            ["PRODUCT_SET_SID", "NAME", "RATING", "REVIEW_COUNT"],
        ),
    ]

    shown_any = False
    for flag_col, title, desc, show_cols in flag_cfg:
        if flag_col not in df.columns:
            continue
        flagged = df[df[flag_col] == True]  # noqa: E712
        if flagged.empty:
            continue
        shown_any = True
        display_cols = [c for c in show_cols if c in flagged.columns]
        with st.expander(f"{title} ({len(flagged)} product{'s' if len(flagged) != 1 else ''})", expanded=False):
            st.caption(desc)
            st.dataframe(
                flagged[display_cols].fillna("").replace("nan", ""),
                use_container_width=True,
                hide_index=True,
            )

    if shown_any:
        st.info(
            "Data quality notices detected in the uploaded file. "
            "These are advisory — they do not affect pass/fail status but should be "
            "reviewed before publishing.",
        )


def _restore_pq_item(sid: str):
    """Mark a manually-rejected post-QC item as approved again."""
    st.session_state.final_report.loc[
        st.session_state.final_report["ProductSetSid"] == sid,
        ["Status", "Reason", "Comment", "FLAG"],
    ] = ["Approved", "", "", "Approved by User"]
    st.session_state.pop(f"pq_qrej_{sid}", None)
    st.session_state.pop(f"pq_qrej_reason_{sid}", None)
    st.session_state.exports_cache.clear()
    st.session_state.display_df_cache.clear()
    st.session_state.main_toasts.append("Restored item to approved.")


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px;'
        f'color:{DARK};margin-bottom:12px;">'
        f'{_ICON["search"]} Post-QC Settings</div>',
        unsafe_allow_html=True,
    )

    pq_country = st.selectbox(
        "Country", COUNTRIES,
        index=COUNTRIES.index(st.session_state.pq_country),
        key="pq_country_select",
    )
    if pq_country != st.session_state.pq_country:
        st.session_state.pq_country = pq_country
        _reset_results()

    country_code = _code(pq_country)
    st.session_state.pq_country = pq_country
    st.session_state._pq_country_code = country_code
    st.markdown("---")

    if _SCRAPER_OK:
        st.caption("Jumia enrichment active — missing fields are auto-filled on upload.")
    else:
        st.caption("jumia_scraper.py not found. Auto-enrichment disabled.")

    st.markdown("---")
    if st.button("Clear Results", use_container_width=True, type="secondary"):
        _reset_results()
        st.rerun()

# ------------------------------------------------------------------
# HEADER BANNER & MAIN UI
# ------------------------------------------------------------------
st.markdown(
    f'<div style="border-left:4px solid {ORANGE};padding:16px 20px;margin-bottom:24px;'
    f'background:#FFFFFF;border-radius:0 8px 8px 0;border-top:1px solid {BORDER};'
    f'border-right:1px solid {BORDER};border-bottom:1px solid {BORDER};">'
    f'<div style="display:flex;align-items:center;gap:10px;">'
    f'{_ICON["shield"].replace("16", "20")}'
    f'<span style="font-size:18px;font-weight:700;color:{DARK};">Post-QC Validation</span>'
    f'</div>'
    f'<p style="color:{MED};margin:6px 0 0;font-size:12px;">'
    f'Validate Jumia product data, enrich missing fields via scraper, and generate audit reports.</p>'
    f'</div>',
    unsafe_allow_html=True,
)

if not _POSTQC_OK:
    st.error(f"**postqc.py could not be imported.**\nError: `{_postqc_err_msg}`")
    st.stop()
if not _FULL_VALIDATION_OK:
    st.warning(f"Validation engine unavailable: `{_fv_err_msg}`. Validations skipped.")

st.markdown(
    f'<div class="pq-section-header">{_ICON["upload"]} Input Source</div>',
    unsafe_allow_html=True,
)
input_mode = st.radio(
    "Input method",
    ["Upload Post-QC File", "Data Grab Upload", "Paste SKUs / URLs"],
    horizontal=True,
    key="input_mode",
)

# ------------------------------------------------------------------
# INPUT FETCHING
# ------------------------------------------------------------------
all_dfs = []
sig = None

if input_mode in ["Upload Post-QC File", "Data Grab Upload"]:
    uploaded = st.file_uploader(
        f"Drop your {input_mode} here",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key=f"pq_uploader_{input_mode}",
    )
    if uploaded:
        files_data = [{"name": f.name, "bytes": f.read()} for f in uploaded]
        sig = hashlib.md5(
            (
                str(sorted(f["name"] + str(len(f["bytes"])) for f in files_data))
                + country_code
                + input_mode
            ).encode()
        ).hexdigest()

        if st.session_state.pq_last_sig != sig:
            for uf in files_data:
                buf = BytesIO(uf["bytes"])
                if uf["name"].endswith(".xlsx"):
                    raw = pd.read_excel(buf, engine="openpyxl", dtype=str)
                else:
                    if input_mode == "Data Grab Upload":
                        try:
                            raw = pd.read_csv(buf, sep="|", dtype=str)
                            if len(raw.columns) <= 1:
                                buf.seek(0)
                                raw = pd.read_csv(buf, sep=",", dtype=str)
                        except Exception:
                            buf.seek(0)
                            raw = pd.read_csv(buf, dtype=str)
                    else:
                        try:
                            raw = pd.read_csv(buf, dtype=str)
                            if len(raw.columns) <= 1:
                                buf.seek(0)
                                raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)
                        except Exception:
                            buf.seek(0)
                            raw = pd.read_csv(buf, sep=";", encoding="ISO-8859-1", dtype=str)

                _is_pq = _POSTQC_OK and detect_file_type(raw) == "post_qc"
                if not _is_pq:
                    raw = _standardise_pq(raw)
                all_dfs.append(raw)
    else:
        if st.session_state.pq_last_sig is not None:
            _reset_results()

elif input_mode == "Paste SKUs / URLs":
    if not _SCRAPER_OK:
        st.error("`jumia_scraper.py` is required for this input mode.")
        st.stop()

    lookup_input_raw = st.text_area(
        "SKUs or Jumia product URLs (one per line)",
        height=160,
        placeholder="GE840EA6C62GANAFAMZ\nhttps://www.jumia.co.ke/some-product.html"
    )

    if lookup_input_raw.strip():
        sig = hashlib.md5(
            (lookup_input_raw + country_code + input_mode).encode()
        ).hexdigest()

        if st.session_state.pq_last_sig != sig:
            do_process = st.button("Search & Process", type="primary")
            if do_process:
                entries = _parse_lookup_inputs(lookup_input_raw)
                if not entries:
                    st.warning("Please enter at least one valid SKU or URL.")
                else:
                    st.info(f"Looking up **{len(entries)}** item(s)...")
                    _sbar = st.progress(0, text="Starting…")
                    _stxt = st.empty()

                    rows = []
                    for idx, entry in enumerate(entries):
                        sku = entry["sku"]
                        direct_url = entry["url"]
                        eff_code = entry["country_hint"] or country_code

                        _sbar.progress(idx / len(entries), text=f"Fetching {idx + 1}/{len(entries)} — {sku}")
                        _stxt.caption(f"{'URL' if direct_url else 'SKU'}: **{entry['input'][:80]}**")

                        try:
                            if direct_url:
                                from jumia_scraper import _scrape_product_page
                                _base = (COUNTRY_URLS or {}).get(eff_code, "https://www.jumia.co.ke")
                                scraped = _scrape_product_page(direct_url, _base)
                            else:
                                scraped = scrape_single_sku(sku, country_code=eff_code)
                        except Exception as exc:
                            logger.warning("Lookup failed for %s: %s", sku, exc)
                            scraped = {}

                        row = {
                            "PRODUCT_SET_SID": sku,
                            "Input": entry["input"],
                            "Type": "URL" if direct_url else "SKU",
                            "Country": eff_code,
                            "Found": "Yes" if scraped else "No",
                        }
                        row.update(scraped)
                        rows.append(row)

                    _sbar.progress(1.0, text="Done!")
                    _stxt.empty()
                    _sbar.empty()

                    raw = pd.DataFrame(rows)
                    raw = _standardise_pq(raw)
                    all_dfs.append(raw)
        else:
            st.success("Inputs processed. Results are displayed below.")

# ------------------------------------------------------------------
# PIPELINE PROCESSING
# ------------------------------------------------------------------
if all_dfs and sig and st.session_state.pq_last_sig != sig:
    _reset_results()

    try:
        support_files = _load_support_files()
        cat_map = support_files.get("category_map", {})
        support_files_pq = dict(support_files)
        support_files_pq["country_code"] = country_code
        support_files_pq["country_name"] = pq_country

        import unicodedata as _ud
        def _nk(s: str) -> str:
            s = _ud.normalize("NFKD", str(s)).lower()
            return re.sub(r"[^a-z0-9]", "", s)
        def _norm_sep(s: str) -> str:
            return re.sub(r"\s*[>/|\\]\s*", " / ", str(s).strip()).lower()

        _cmap_by_path, _cmap_by_name, _cmap_by_seg = {}, {}, {}
        _cmap_xlsx_path = "category_map.xlsx"
        if os.path.exists(_cmap_xlsx_path):
            try:
                _cm_df = pd.read_excel(_cmap_xlsx_path, engine="openpyxl", dtype=str)
                _cm_df.columns = [c.strip().lower() for c in _cm_df.columns]
                _col_name = next((c for c in _cm_df.columns if "name" in c and "path" not in c), None)
                _col_code = next((c for c in _cm_df.columns if "code" in c), None)
                _col_path = next((c for c in _cm_df.columns if "path" in c), None)
                if _col_name and _col_code:
                    for _, _row in _cm_df.iterrows():
                        _cs = str(_row[_col_code]).split(".")[0].strip()
                        if not re.match(r"^\d+$", _cs): continue
                        _nm = str(_row[_col_name]).strip()
                        if _nm and _nm.lower() not in ("nan", "none", ""):
                            _cmap_by_name[_nm.lower()] = _cs
                            _cmap_by_seg.setdefault(_nm.lower(), _cs)
                        if _col_path:
                            _pt = str(_row[_col_path]).strip()
                            if _pt and _pt.lower() not in ("nan", "none", ""):
                                _cmap_by_path[_norm_sep(_pt)] = _cs
                                for _seg in _pt.split(" / "):
                                    _seg = _seg.strip()
                                    if _seg and _seg.lower() not in ("nan","none",""):
                                        _cmap_by_seg.setdefault(_seg.lower(), _cs)
            except Exception as _cm_err:
                logger.warning(f"category_map.xlsx load failed: {_cm_err}")
        else:
            for _n, _c in cat_map.items():
                _cs = str(_c).split(".")[0].strip()
                if re.match(r"^\d+$", _cs):
                    _cmap_by_name[_n.lower()] = _cs
                    _cmap_by_seg[_n.lower()]   = _cs

        _cmap_seg_sorted = sorted(_cmap_by_seg.items(), key=lambda x: len(x[0]), reverse=True)

        def _resolve_cat_code(category: str, existing: str = "") -> str:
            _existing = str(existing).strip()
            raw = str(category).strip()
            if not raw or raw.lower() in ("nan", "none", ""): return _existing
            def _valid(v: str) -> str:
                v = str(v).split(".")[0].strip()
                return v if re.match(r"^\d+$", v) else ""

            c = _valid(_cmap_by_path.get(_norm_sep(raw), ""))
            if c: return c

            segs = [s.strip() for s in re.split(r"\s*[>/|\\]\s*", raw) if s.strip()]
            for seg in reversed(segs):
                c = _valid(_cmap_by_name.get(seg.lower(), ""))
                if c: return c

            for seg in reversed(segs):
                sn = _nk(seg)
                if not sn: continue
                best_c, best_len = "", 0
                for k, v in _cmap_seg_sorted:
                    kn = _nk(k)
                    if not kn: continue
                    if len(kn) < 3: continue
                    if kn == sn or kn in sn or sn in kn:
                        c = _valid(v)
                        if c and len(kn) > best_len:
                            best_c, best_len = c, len(kn)
                if best_c: return best_c

            if re.match(r"^\d+$", _existing): return _existing
            return _existing

        # ── Normalise all loaded DataFrames ────────────────────
        norm_dfs = []
        for df in all_dfs:
            if "PRODUCT_SET_SID" in df.columns:
                ndf = df.copy()
                if "CATEGORY_CODE" not in ndf.columns:
                    ndf["CATEGORY_CODE"] = ""
            else:
                ndf = normalize_post_qc(df, category_map=cat_map)
                if "CATEGORY_CODE" not in ndf.columns:
                    ndf["CATEGORY_CODE"] = ""
            if "CATEGORY" in ndf.columns:
                ndf["CATEGORY_CODE"] = ndf.apply(
                    lambda r: _resolve_cat_code(r.get("CATEGORY", ""), r.get("CATEGORY_CODE", "")), axis=1
                )
            norm_dfs.append(ndf)

        merged = pd.concat(norm_dfs, ignore_index=True)
        # ── Dedup on SKU (keep first occurrence) ───────────────
        # Duplicate-SKU validation is intentionally removed from this page.
        # The dedup still runs so the rest of the pipeline works correctly,
        # but no flag or warning is surfaced to the user.
        merged_dedup = merged.drop_duplicates(subset=["PRODUCT_SET_SID"], keep="first")

        # ── Scraper enrichment ─────────────────────────────────
        if _SCRAPER_OK and input_mode != "Paste SKUs / URLs":
            _missing_cols = [
                c for c in SCRAPABLE_FIELDS
                if c not in merged_dedup.columns
                or merged_dedup[c].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").mean() > 0.5
            ]
            if _missing_cols:
                st.info(f"Enriching **{len(merged_dedup)}** products from Jumia ({', '.join(_missing_cols)})...")
                _bar = st.progress(0, text="Starting enrichment...")
                _txt = st.empty()
                _before_df = merged_dedup.copy()

                def _cb(done, total, sku, bar=_bar, txt=_txt):
                    bar.progress(done / max(total, 1), text=f"Scraped {done}/{total} — {sku}")
                    txt.caption(f"Last scraped: **{sku}**")

                merged_dedup = enrich_post_qc_df(
                    merged_dedup, country_code=country_code, progress_callback=_cb
                )
                _bar.empty()
                _txt.empty()

                enrich_summary = _build_enrichment_summary(_before_df, merged_dedup)
                total_filled = sum(v["filled"] for v in enrich_summary.values())

                st.session_state.enriched_df        = merged_dedup.copy()
                st.session_state.enrichment_summary = enrich_summary
                st.session_state.enrichment_done    = True

                if total_filled:
                    st.toast(f"{total_filled} cell(s) filled from Jumia")
                else:
                    st.toast("No new data found on Jumia.")
            else:
                st.toast("All enrichable columns already populated — no scraping needed.")

        elif input_mode == "Paste SKUs / URLs":
            st.session_state.enriched_df = merged_dedup.copy()
            st.session_state.enrichment_done = True

        # ── Prepare data for full validation ───────────────────
        ready = merged_dedup.copy()

        if "CATEGORY_PATH" in ready.columns:
            if "CATEGORY" not in ready.columns:
                ready["CATEGORY"] = ready["CATEGORY_PATH"]
            else:
                ready["CATEGORY"] = ready["CATEGORY"].where(ready["CATEGORY"].ne(""), ready["CATEGORY_PATH"])

        ready["ACTIVE_STATUS_COUNTRY"] = country_code

        if "CATEGORY" in ready.columns:
            if "CATEGORY_CODE" not in ready.columns:
                ready["CATEGORY_CODE"] = ""
            ready["CATEGORY_CODE"] = ready.apply(
                lambda r: _resolve_cat_code(r.get("CATEGORY", ""), r.get("CATEGORY_CODE", "")), axis=1
            )
            _unresolved = ready[~ready["CATEGORY_CODE"].astype(str).str.match(r"^\d+$")]["CATEGORY"].dropna().unique()
            if len(_unresolved):
                st.warning(
                    f"**{len(_unresolved)}** category path(s) could not be resolved to a numeric code.\n\n"
                    + "\n".join(f"- `{c}`" for c in _unresolved[:10])
                    + ("\n- *(and more...)*" if len(_unresolved) > 10 else ""),
                )

        if "LIST_VARIATIONS" not in ready.columns or (
            ready["LIST_VARIATIONS"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").all()
        ):
            _var_parts = []
            for _vcol in ["COLOR", "SIZES_AVAILABLE"]:
                if _vcol in ready.columns:
                    _var_parts.append(ready[_vcol].astype(str).str.strip().replace({"nan": "", "None": ""}))
            if _var_parts:
                import functools as _ft
                ready["LIST_VARIATIONS"] = _ft.reduce(
                    lambda a, b: a.where(b.eq(""), a + " | " + b).where(a.ne(""), b), _var_parts
                ).replace({"": pd.NA})
            else:
                ready["LIST_VARIATIONS"] = ""

        def _parse_price_str(s) -> str:
            try:
                clean = re.sub(r"[^\d.]", "", str(s))
                val = float(clean) if clean else None
                if val and val > 0: return str(val)
            except (ValueError, TypeError): pass
            return ""

        _file_price_candidates = [c for c in ["GLOBAL_SALE_PRICE", "GLOBAL_PRICE", "SALE_PRICE"] if c in ready.columns]
        _file_price_col = _file_price_candidates[0] if _file_price_candidates else None

        if _file_price_col and _file_price_col != "GLOBAL_SALE_PRICE":
            ready["GLOBAL_SALE_PRICE"] = ready[_file_price_col].astype(str).apply(_parse_price_str)
        elif "GLOBAL_SALE_PRICE" not in ready.columns:
            ready["GLOBAL_SALE_PRICE"] = ""

        _sp_empty = ready["GLOBAL_SALE_PRICE"].astype(str).str.strip().eq("")
        if "PRICE" in ready.columns and _sp_empty.any():
            ready.loc[_sp_empty, "GLOBAL_SALE_PRICE"] = (
                ready.loc[_sp_empty, "PRICE"].astype(str).apply(_parse_price_str)
            )

        if "GLOBAL_PRICE" not in ready.columns or (
            ready["GLOBAL_PRICE"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("").all()
        ):
            ready["GLOBAL_PRICE"] = ready["GLOBAL_SALE_PRICE"]

        if "PARENTSKU" not in ready.columns: ready["PARENTSKU"] = ""

        for _col_needed in [
            "NAME", "BRAND", "COLOR", "SELLER_NAME",
            "PRODUCT_WARRANTY", "WARRANTY_DURATION", "COUNT_VARIATIONS", "COLOR_FAMILY",
        ]:
            if _col_needed not in ready.columns: ready[_col_needed] = ""
        for _col_str in ["NAME", "BRAND", "COLOR", "SELLER_NAME"]:
            if _col_str in ready.columns:
                ready[_col_str] = ready[_col_str].astype(str).fillna("")

        # ── NEW: derive IS_OFFICIAL_STORE from TAGS ────────────
        ready = _derive_official_store_from_tags(ready)

        # ── NEW: calculate DISCOUNT when missing ───────────────
        ready = _calc_discount_if_missing(ready)

        # ── COLOR: always derive from title (COLOR_IN_TITLE) ───────────
        # For Post-QC, COLOR is authoritative only when it comes from the
        # product title.  Pill-scraped colors are unreliable (wrong DOM
        # elements, spec strings, share buttons, etc.).
        # Rule:
        #   • COLOR_IN_TITLE has a real color word → use that as COLOR
        #   • COLOR_IN_TITLE is "None" or blank     → set COLOR = "None"
        #   • COLOR_IN_TITLE column absent entirely → leave COLOR as-is
        if "COLOR_IN_TITLE" in ready.columns:
            if "COLOR" not in ready.columns:
                ready["COLOR"] = ""

            _cit = ready["COLOR_IN_TITLE"].astype(str).str.strip()

            # Rows where COLOR_IN_TITLE has a real value (not blank / "None")
            _has_title_color = (
                _cit.str.lower().ne("none")
                & _cit.replace({"nan": "", "None": "", "none": ""}).ne("")
            )
            # Overwrite COLOR from title wherever scraper found a color word
            ready.loc[_has_title_color, "COLOR"] = _cit[_has_title_color]

            # Where title has no color → set COLOR = "None" explicitly
            ready.loc[~_has_title_color, "COLOR"] = "None"

        # ── NEW: internal quality flags (not pass/fail — advisory) ─
        ready = _flag_corrupted_old_price(ready)
        ready = _flag_discount_mismatch(ready)
        ready = _flag_brand_in_title(ready)
        ready = _flag_rating_no_reviews(ready)

        # ── Run full validation pipeline ───────────────────────
        if _FULL_VALIDATION_OK:
            # Strip internal flag columns before passing to validator
            # (validator doesn't know about them and may choke on extras)
            _internal_flags = [c for c in ready.columns if c.startswith("_")]
            _ready_for_val  = ready.drop(columns=_internal_flags, errors="ignore")

            with st.spinner("Running validations…"):
                _cv = CountryValidator(pq_country)
                try:
                    _sf_full = load_all_support_files()
                except Exception:
                    _sf_full = support_files_pq

                data_has_warranty = all(
                    c in _ready_for_val.columns
                    for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"]
                )
                _val_report, _val_results = validate_products(
                    _ready_for_val, _sf_full, _cv, data_has_warranty
                )

            st.session_state.pq_val_report  = _val_report
            st.session_state.pq_val_results = _val_results
            st.session_state.pq_flags_init  = False

        st.session_state.pq_data     = ready   # keep flags for advisory display
        st.session_state.pq_last_sig = sig
        if not st.session_state.enrichment_done:
            st.session_state.enriched_df = ready.copy()

        # ── Ensure MAIN_IMAGE is always in pq_data for the image grid ──
        # For upload modes where the file had no image column, the scraper
        # fills MAIN_IMAGE into enriched_df / merged_dedup, but pq_data is
        # set from ready (= merged_dedup) so it should already have it.
        # This guard handles the edge case where enrichment was skipped
        # (all cols populated) but enriched_df still has images from a
        # previous run — we copy them across so the grid never shows blanks.
        if (
            "MAIN_IMAGE" not in st.session_state.pq_data.columns
            or st.session_state.pq_data["MAIN_IMAGE"]
               .astype(str).str.strip()
               .replace({"nan": "", "None": ""}).eq("").all()
        ):
            _edf_ref = st.session_state.get("enriched_df")
            if _edf_ref is not None and "MAIN_IMAGE" in _edf_ref.columns:
                _img_map = (
                    _edf_ref.set_index("PRODUCT_SET_SID")["MAIN_IMAGE"]
                    .astype(str).str.strip()
                    .replace({"nan": "", "None": ""})
                )
                if "MAIN_IMAGE" not in st.session_state.pq_data.columns:
                    st.session_state.pq_data["MAIN_IMAGE"] = ""
                st.session_state.pq_data["MAIN_IMAGE"] = (
                    st.session_state.pq_data["PRODUCT_SET_SID"]
                    .map(_img_map)
                    .fillna(st.session_state.pq_data["MAIN_IMAGE"])
                )

    except Exception as exc:
        st.error(f"Processing error: {exc}")
        st.code(traceback.format_exc())

# ------------------------------------------------------------------
# ENRICHMENT PANEL
# ------------------------------------------------------------------
if st.session_state.enrichment_done and st.session_state.enriched_df is not None:
    edf      = st.session_state.enriched_df
    esummary = st.session_state.enrichment_summary

    st.markdown("---")

    if esummary:
        total_cells_filled = sum(v["filled"] for v in esummary.values())
        cols_filled        = [c for c, v in esummary.items() if v["filled"] > 0]

        st.markdown(
            f'<div class="pq-section-header">{_ICON["globe"]} Jumia Enrichment Results '
            f'<span class="enrich-badge">+{total_cells_filled} cells filled</span></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Products Processed", len(edf))
        m2.metric("Fields Attempted",   len(SCRAPABLE_FIELDS))
        m3.metric("Fields Enriched",    len(cols_filled))
        m4.metric("Total Cells Filled", total_cells_filled)

        with st.expander("Field-by-field breakdown", expanded=True):
            chip_html = ""
            for col in SCRAPABLE_FIELDS:
                info  = esummary.get(col, {"before": 0, "after": 0, "filled": 0})
                label = col.replace("_", " ").title()
                if info["filled"] > 0:
                    chip_html += f'<span class="field-chip">{label} (+{info["filled"]})</span>'
                else:
                    status = "—" if info["after"] == 0 else f'{info["after"]} rows'
                    chip_html += f'<span class="field-chip missing">{label} ({status})</span>'
            st.markdown(chip_html, unsafe_allow_html=True)

    else:
        st.markdown(
            f'<div class="pq-section-header">{_ICON["globe"]} Data Generated '
            f'<span class="enrich-badge">{len(edf)} Products Loaded</span></div>',
            unsafe_allow_html=True,
        )

    with st.expander(f"Data preview — enriched ({len(edf)} rows)", expanded=False):
        _preview_cols = [
            c for c in edf.columns
            if not c.startswith("_")
            and edf[c].astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()
        ]
        st.dataframe(
            edf[_preview_cols].fillna("").replace("nan", ""),
            use_container_width=True, height=400,
        )

# ------------------------------------------------------------------
# DATA QUALITY FLAGS (advisory — shown before validation results)
# ------------------------------------------------------------------
_pq_data = st.session_state.pq_data
if not _pq_data.empty:
    _has_any_flag = any(
        c in _pq_data.columns and _pq_data[c].any()
        for c in ["_OLD_PRICE_CORRUPTED", "_DISCOUNT_MISMATCH",
                  "_BRAND_IN_TITLE", "_RATING_NO_REVIEWS"]
    )
    if _has_any_flag:
        st.markdown("---")
        st.markdown(
            f'<div class="pq-section-header">{_ICON["alert"]} Data Quality Notices</div>',
            unsafe_allow_html=True,
        )
        _render_data_quality_flags(_pq_data)

# ------------------------------------------------------------------
# FULL VALIDATION RESULTS
# _val_report must always be a DataFrame (never undefined) so the
# image-grid fragment at the bottom can safely reference it too.
# ------------------------------------------------------------------
_val_report = st.session_state.get("pq_val_report", pd.DataFrame())
if not isinstance(_val_report, pd.DataFrame):
    _val_report = pd.DataFrame()

if _FULL_VALIDATION_OK and not _val_report.empty and not _pq_data.empty:
    _app_df = _val_report[_val_report["Status"] == "Approved"]
    _rej_df = _val_report[_val_report["Status"] == "Rejected"]
    _sf     = _load_support_files()
    _cv     = CountryValidator(pq_country)

    st.markdown("---")
    st.markdown(
        f'<div class="pq-section-header">{_ICON["shield"]} Audit Summary</div>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    _total = len(_pq_data[[c for c in _pq_data.columns if not c.startswith("_")]])
    for _col, _lbl, _val, _color in [
        (mc1, "Total Products",  _total,           DARK),
        (mc2, "Approved",        len(_app_df),     GREEN),
        (mc3, "Rejected",        len(_rej_df),     RED),
        (mc4, "Rejection Rate",  f"{(len(_rej_df)/_total*100) if _total > 0 else 0:.1f}%", ORANGE),
    ]:
        with _col:
            st.markdown(f"<div class='metric-accent' style='background:{_color};'></div>", unsafe_allow_html=True)
            st.metric(_lbl, _val)

    if st.session_state.final_report.empty:
        st.session_state.final_report = st.session_state.pq_val_report.copy()

    if not _rej_df.empty:
        st.markdown(
            f'<div class="pq-section-header" style="margin-top:16px;">{_ICON["flag"]} Rejection Flags</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.pq_flags_init:
            _top = _rej_df["FLAG"].value_counts().index[0]
            _top_key = "pqexp_" + re.sub(r"[^a-zA-Z0-9]", "_", _top)
            st.session_state[_top_key] = True
            st.session_state.pq_flags_init = True

        st.session_state.final_report = st.session_state.pq_val_report.copy()
        _data_has_w = all(c in _pq_data.columns for c in ["PRODUCT_WARRANTY", "WARRANTY_DURATION"])

        for _flag_title in _rej_df["FLAG"].unique():
            _flag_df = _rej_df[_rej_df["FLAG"] == _flag_title]
            _flag_key = "pqexp_" + re.sub(r"[^a-zA-Z0-9]", "_", _flag_title)
            with st.expander(f"{_flag_title} ({len(_flag_df)})", key=_flag_key):
                pq_render_flag_expander(_flag_title, _flag_df, _pq_data, _data_has_w, _sf, _cv)

        st.session_state.pq_val_report = st.session_state.final_report.copy()
    else:
        st.success("All products passed validation — no rejections found.")

    st.markdown("---")
    st.markdown(
        f'<div class="pq-section-header">{_ICON["download"]} Download Post-QC Results</div>',
        unsafe_allow_html=True,
    )
    _date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    
    # ── CONSOLIDATED SINGLE-FILE EXPORT ──────────────────────────
    _export_df = _pq_data.copy()
    
    _status_map = st.session_state.final_report.set_index("ProductSetSid")["Status"].to_dict()
    _flag_map = st.session_state.final_report.set_index("ProductSetSid")["FLAG"].to_dict()
    _comment_map = st.session_state.final_report.set_index("ProductSetSid")["Comment"].to_dict()
    
    def _get_final_flag(sku):
        stat = _status_map.get(sku, "Approved")
        if stat == "Approved":
            return "ok"
        return _flag_map.get(sku, "Rejected")
        
    def _get_final_comment(sku):
        stat = _status_map.get(sku, "Approved")
        if stat == "Approved":
            return "ok"
        return _comment_map.get(sku, "")
        
    _export_df["FINAL_FLAG"] = _export_df["PRODUCT_SET_SID"].apply(_get_final_flag)
    _export_df["FLAG_REASON"] = _export_df["PRODUCT_SET_SID"].apply(_get_final_comment)
    
    # Clean up advisory internal flags from export
    _export_df = _export_df.drop(columns=[c for c in _export_df.columns if c.startswith("_")], errors="ignore")
    
    st.download_button(
        label="Download Final Results (.xlsx)",
        data=_to_excel_bytes(_export_df),
        file_name=f"Post_QC_Results_{pq_country}_{country_code}_{_date_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

# ------------------------------------------------------------------
# JTBRIDGE — receives reject / undo messages from the iframe
# Uses pq_qrej_ prefixed keys to avoid colliding with main app state.
# ------------------------------------------------------------------
_pq_bridge_val = st.text_input(
    "jtbridge_pq", value="",
    placeholder="JTBRIDGE_UNIQUE_DO_NOT_USE",
    key=f"pq_bridge_{st.session_state.pq_bridge_counter}",
    label_visibility="collapsed",
)

if _pq_bridge_val:
    try:
        _pq_msg = json.loads(_pq_bridge_val)

        if _pq_msg.get("action") == "reject":
            _pq_payload = _pq_msg.get("payload", {})
            if isinstance(_pq_payload, dict) and _pq_payload:
                _pq_sf = _load_support_files()
                _pq_rgroups: dict = {}
                for _sid, _rkey in _pq_payload.items():
                    _pq_rgroups.setdefault(_rkey, []).append(_sid)
                _pq_total = 0
                for _rkey, _sids in _pq_rgroups.items():
                    _flag    = REASON_MAP.get(_rkey, "Poor images")
                    _rinfo   = _pq_sf.get("flags_mapping", {}).get(
                        _flag,
                        {"reason": "1000042 - Kindly follow our product image upload guideline.",
                         "en": "Poor Image Quality"},
                    )
                    _rcode = _rinfo["reason"]
                    _rcmt  = _rinfo.get("en", "Manual image rejection")
                    st.session_state.final_report.loc[
                        st.session_state.final_report["ProductSetSid"].isin(_sids),
                        ["Status", "Reason", "Comment", "FLAG"],
                    ] = ["Rejected", _rcode, _rcmt, _flag]
                    for _s in _sids:
                        st.session_state[f"pq_qrej_{_s}"]        = True
                        st.session_state[f"pq_qrej_reason_{_s}"] = _flag
                    _pq_total += len(_sids)
                st.session_state.exports_cache.clear()
                st.session_state.display_df_cache.clear()
                st.session_state.pq_val_exports.clear()
                # Sync final_report back into pq_val_report so the flags
                # breakdown above updates on re-render
                st.session_state.pq_val_report = st.session_state.final_report.copy()
                st.session_state.main_toasts.append(
                    (f"Rejected {_pq_total} product(s) via image review", "")
                )
                st.session_state.pq_bridge_counter += 1
                st.session_state.pq_do_scroll_top = False
                st.rerun()

        elif _pq_msg.get("action") == "undo":
            _pq_payload = _pq_msg.get("payload", {})
            if isinstance(_pq_payload, dict):
                for _sid in _pq_payload.keys():
                    _restore_pq_item(_sid)
                st.session_state.pq_val_report = st.session_state.final_report.copy()
                st.session_state.pq_bridge_counter += 1
                st.session_state.pq_do_scroll_top = False
                st.rerun()

    except Exception as _pq_bridge_err:
        logger.error(f"PQ bridge parse error: {_pq_bridge_err}")


# ------------------------------------------------------------------
# IMAGE REVIEW GRID
# Shown after validation results when data is available.
# ------------------------------------------------------------------
@st.fragment
def _render_pq_image_grid():
    _fr   = st.session_state.get("final_report", pd.DataFrame())
    _data = st.session_state.pq_data

    if _fr.empty or _data.empty:
        return

    # Read country from session state — pq_country is a local var in the outer
    # script and is NOT available during fragment-scoped reruns.
    _pq_country      = st.session_state.get("pq_country", "Kenya")

    _sf = _load_support_files()

    st.markdown("---")
    st.markdown(
        f'<div class="pq-section-header">{_ICON["image"]} Manual Image Review</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Inspect product images below. Use 'Poor Image' or the dropdown to reject individual cards, "
        "then 'Batch Reject' to commit. Rejections sync back to the audit summary above."
    )

    # Show approved + already-manually-rejected (so undo overlay works)
    _committed_sids = {
        k.replace("pq_qrej_", "")
        for k in st.session_state.keys()
        if k.startswith("pq_qrej_") and "reason" not in k
    }
    _approved_mask = (
        (_fr["Status"] == "Approved") |
        (_fr["ProductSetSid"].isin(_committed_sids))
    )
    _grid_fr = _fr[_approved_mask]

    # ── filters ──────────────────────────────────────────────────────
    _gc1, _gc2, _gc3 = st.columns([1.5, 1.5, 2])
    with _gc1:
        _gsearch_n = st.text_input("Search by Name", placeholder="Product name\u2026",
                                   key="pq_grid_search_n")
    with _gc2:
        _gsearch_sc = st.text_input("Search by Seller/Category",
                                    placeholder="Seller or Category\u2026",
                                    key="pq_grid_search_sc")
    with _gc3:
        st.session_state.pq_grid_items_per_page = st.select_slider(
            "Items per page", options=[20, 50, 100, 200],
            value=st.session_state.pq_grid_items_per_page,
            key="pq_grid_ipp",
        )

    # ── merge product data ────────────────────────────────────────────
    # Always include MAIN_IMAGE explicitly — it is the core of this grid
    # and must not be dropped if GRID_COLS from the main app doesn't list it.
    _must_have = ["PRODUCT_SET_SID", "MAIN_IMAGE"]
    _avail_cols = list(dict.fromkeys(
        [c for c in GRID_COLS if c in _data.columns]
        + [c for c in _must_have if c in _data.columns]
    ))
    _review_data = pd.merge(
        _grid_fr[["ProductSetSid"]],
        _data[_avail_cols],
        left_on="ProductSetSid", right_on="PRODUCT_SET_SID", how="left",
    )

    # Resolve MAIN_IMAGE: prefer the scraped URL already in pq_data;
    # fall back to enriched_df in case the grid is showing data before
    # pq_data was updated (e.g. fragment re-run before full rerun).
    if "MAIN_IMAGE" not in _review_data.columns:
        _review_data["MAIN_IMAGE"] = ""
    _img_blank = _review_data["MAIN_IMAGE"].astype(str).str.strip().replace({"nan": "", "None": ""}).eq("")
    if _img_blank.any():
        _edf_img = st.session_state.get("enriched_df")
        if _edf_img is not None and "MAIN_IMAGE" in _edf_img.columns:
            _img_lookup = (
                _edf_img.set_index("PRODUCT_SET_SID")["MAIN_IMAGE"]
                .astype(str).str.strip()
                .replace({"nan": "", "None": ""})
            )
            _review_data.loc[_img_blank, "MAIN_IMAGE"] = (
                _review_data.loc[_img_blank, "ProductSetSid"].map(_img_lookup).fillna("")
            )

    if _gsearch_n:
        _review_data = _review_data[
            _review_data["NAME"].astype(str).str.contains(_gsearch_n, case=False, na=False)
        ]
    if _gsearch_sc:
        _mc = (
            _review_data["CATEGORY"].astype(str).str.contains(_gsearch_sc, case=False, na=False)
            if "CATEGORY" in _review_data.columns
            else pd.Series(False, index=_review_data.index)
        )
        _ms = _review_data["SELLER_NAME"].astype(str).str.contains(
            _gsearch_sc, case=False, na=False
        )
        _review_data = _review_data[_mc | _ms]

    if _review_data.empty:
        st.info("No products to display in the image grid.")
        return

    # ── pagination ────────────────────────────────────────────────────
    _ipp         = st.session_state.pq_grid_items_per_page
    _total_pages = max(1, (len(_review_data) + _ipp - 1) // _ipp)
    if st.session_state.pq_grid_page >= _total_pages:
        st.session_state.pq_grid_page = 0

    _pg_c1, _pg_c2, _pg_c3 = st.columns([1, 2, 1], vertical_alignment="center")
    with _pg_c1:
        if st.button("\u25c4 Prev", use_container_width=True,
                     disabled=st.session_state.pq_grid_page == 0,
                     key="pq_grid_prev"):
            st.session_state.pq_grid_page = max(0, st.session_state.pq_grid_page - 1)
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")
    with _pg_c2:
        _new_page = st.number_input(
            f"Page (of {_total_pages} \u00b7 {len(_review_data)} items)",
            min_value=1, max_value=max(1, _total_pages),
            value=st.session_state.pq_grid_page + 1, step=1,
            key="pq_grid_page_input",
        )
        if _new_page - 1 != st.session_state.pq_grid_page:
            st.session_state.pq_grid_page = _new_page - 1
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")
    with _pg_c3:
        if st.button("Next \u25ba", use_container_width=True,
                     disabled=st.session_state.pq_grid_page >= _total_pages - 1,
                     key="pq_grid_next"):
            st.session_state.pq_grid_page += 1
            st.session_state.pq_do_scroll_top = True
            st.rerun(scope="fragment")

    _page_start = st.session_state.pq_grid_page * _ipp
    _page_data  = _review_data.iloc[_page_start : _page_start + _ipp]

    # ── image quality warnings (async) ───────────────────────────────
    _page_warnings: dict = {}
    if _IMAGE_CHECK_OK:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as _ex:
            _fut_map = {
                _ex.submit(
                    _analyze_image_quality,
                    str(_r.get("MAIN_IMAGE", "")).strip()
                ): str(_r["PRODUCT_SET_SID"])
                for _, _r in _page_data.iterrows()
            }
            for _fut in concurrent.futures.as_completed(_fut_map):
                _warns = _fut.result()
                if _warns:
                    _page_warnings[_fut_map[_fut]] = _warns

    # ── committed rejections for this page ───────────────────────────
    _rejected_state = {
        _sid: st.session_state[f"pq_qrej_reason_{_sid}"]
        for _sid in _page_data["PRODUCT_SET_SID"].astype(str)
        if st.session_state.get(f"pq_qrej_{_sid}")
    }

    _cols_per_row = 3 if st.session_state.get("layout_mode") == "centered" else 4

    # ── build and render the grid iframe ─────────────────────────────
    # Try the main app grid builder first; fall back to self-contained version.
    # Use _pq_country (from session state) NOT the outer-scope pq_country.
    _grid_html = None
    try:
        _grid_html = build_fast_grid_html(
            _page_data, _sf.get("flags_mapping", {}),
            _pq_country, _page_warnings, _rejected_state, _cols_per_row,
        )
    except Exception as _grid_err:
        logger.warning("build_fast_grid_html failed (%s), using fallback", _grid_err)

    if _grid_html is None:
        _grid_html = _build_pq_grid_html(
            _page_data, _page_warnings, _rejected_state, _cols_per_row,
        )

    components.html(_grid_html, height=800, scrolling=True)

    if st.session_state.get("pq_do_scroll_top", False):
        components.html(
            "<script>"
            "window.parent.document.querySelector('.main')"
            ".scrollTo({top:0,behavior:'smooth'});"
            "</script>",
            height=0,
        )
        st.session_state.pq_do_scroll_top = False


# ------------------------------------------------------------------
# Lightweight image quality checker (no dependency on main app cache)
# ------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def _analyze_image_quality(url: str):
    if not url or not url.startswith("http"):
        return []
    if not _IMAGE_CHECK_OK:
        return []
    warnings = []
    try:
        resp = _requests.get(url, timeout=1, stream=True)
        if resp.status_code == 200:
            img = _Image.open(resp.raw)
            w, h = img.size
            if w < 300 or h < 300:
                warnings.append("Low Resolution")
            ratio = h / w if w > 0 else 1
            if ratio > 1.5:
                warnings.append("Tall (Screenshot?)")
            elif ratio < 0.6:
                warnings.append("Wide Aspect")
    except Exception:
        pass
    return warnings


# ------------------------------------------------------------------
# Fallback grid builder (used if build_fast_grid_html isn't importable)
# Mirrors the main app's grid but scoped to post-QC page keys.
# ------------------------------------------------------------------
def _build_pq_grid_html(page_data, page_warnings, rejected_state, cols_per_row):
    import json as _json

    O = ORANGE
    R = RED
    G = GREEN

    committed_json = _json.dumps(rejected_state)

    cards_data = []
    for _, row in page_data.iterrows():
        sid     = str(row["PRODUCT_SET_SID"])
        img_url = str(row.get("MAIN_IMAGE", "")).strip()
        if img_url.startswith("http://"):
            img_url = img_url.replace("http://", "https://")
        if not img_url.startswith("http"):
            img_url = ""

        # Detect offline: scraper found nothing — all key content fields blank
        def _blank(v):
            return not str(v).strip() or str(v).strip().lower() in ("nan", "none", "")
        is_offline = all(_blank(row.get(f)) for f in ["NAME", "BRAND", "MAIN_IMAGE", "GLOBAL_SALE_PRICE"])

        if not img_url:
            img_url = "https://via.placeholder.com/150?text=Offline" if is_offline else "https://via.placeholder.com/150?text=No+Image"

        try:
            sp   = row.get("GLOBAL_SALE_PRICE")
            rp   = row.get("GLOBAL_PRICE")
            uval = sp if pd.notna(sp) and str(sp).strip() not in ("", "nan") else rp
            price_str = f"KSh {float(str(uval).replace(',','')):.0f}" if pd.notna(uval) else ""
        except Exception:
            price_str = ""

        name   = str(row.get("NAME",  "")).strip()
        brand  = str(row.get("BRAND", "")).strip()
        seller = str(row.get("SELLER_NAME", "")).strip()

        cards_data.append({
            "sid":      sid,
            "img":      img_url,
            "name":     name if name and name.lower() not in ("nan", "none") else "",
            "brand":    brand if brand and brand.lower() not in ("nan", "none") else "",
            "cat":      str(row.get("CATEGORY", "")).strip(),
            "seller":   seller if seller and seller.lower() not in ("nan", "none") else "",
            "warnings": page_warnings.get(sid, []),
            "price":    price_str,
            "offline":  is_offline,
        })
    cards_json = _json.dumps(cards_data)



    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:sans-serif;}}
body{{background:#f5f5f5;padding:8px;}}
.ctrl-bar{{
  position:sticky;top:0;z-index:999;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:8px 12px;background:rgba(255,255,255,0.96);
  border-bottom:2px solid {O};border-radius:4px;margin-bottom:12px;
  box-shadow:0 4px 16px rgba(0,0,0,0.12);
}}
.sel-count{{font-weight:700;color:{O};font-size:13px;min-width:80px;}}
.reason-sel{{flex:1;min-width:160px;padding:6px 10px;border:1px solid #ccc;border-radius:4px;font-size:12px;background:#fff;}}
.batch-btn{{padding:7px 14px;background:{O};color:#fff;border:none;border-radius:4px;font-weight:700;font-size:12px;cursor:pointer;}}
.batch-btn:hover{{opacity:.88;}}
.desel-btn{{padding:7px 12px;background:#fff;color:#555;border:1px solid #ccc;border-radius:4px;font-size:12px;cursor:pointer;}}
.grid{{display:grid;grid-template-columns:repeat({cols_per_row},1fr);gap:12px;}}
.card{{border:2px solid #e0e0e0;border-radius:8px;padding:10px;background:#fff;position:relative;transition:border-color .15s;z-index:1;}}
.card.selected{{border-color:{G};box-shadow:0 0 0 3px rgba(76,175,80,.2);background:rgba(76,175,80,.04);}}
.card.staged-rej{{border-color:{R};box-shadow:0 0 0 3px rgba(231,60,23,.2);background:rgba(231,60,23,.04);}}
.card.committed-rej{{border-color:#bbb;opacity:.6;}}
.img-wrap{{position:relative;cursor:pointer;height:180px;display:flex;align-items:center;justify-content:center;background:#fff;border-radius:6px;}}
.card-img{{width:100%;height:180px;object-fit:contain;border-radius:6px;transition:transform .2s;}}
.card.committed-rej .card-img{{filter:grayscale(80%);}}
.card-img.zoomed{{transform:scale(2.3);box-shadow:0 15px 50px rgba(0,0,0,0.6);border:2px solid {O};z-index:9999;position:relative;border-radius:8px;}}
.zoom-btn{{position:absolute;bottom:6px;left:6px;width:28px;height:28px;background:rgba(255,255,255,.95);border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.3);z-index:100;font-size:14px;}}
.tick{{position:absolute;bottom:6px;right:6px;width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,.18);display:flex;align-items:center;justify-content:center;color:transparent;font-size:13px;font-weight:900;pointer-events:none;z-index:10;}}
.card.selected .tick{{background:{G};color:#fff;}}
.card.staged-rej .tick{{background:{R};color:#fff;}}
.warn-wrap{{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;z-index:5;pointer-events:none;}}
.warn-badge{{background:rgba(255,193,7,.95);color:#313133;font-size:9px;font-weight:800;padding:3px 7px;border-radius:10px;}}
.price-badge{{position:absolute;top:6px;left:6px;background:rgba(76,175,80,.95);color:#fff;font-size:10px;font-weight:800;padding:3px 7px;border-radius:10px;z-index:5;pointer-events:none;}}
.rej-overlay{{display:none;position:absolute;inset:0;background:rgba(255,255,255,.90);border-radius:6px;flex-direction:column;align-items:center;justify-content:center;z-index:20;gap:5px;padding:8px;text-align:center;}}
.card.committed-rej .rej-overlay{{display:flex;}}
.card.staged-rej .rej-overlay.staged{{display:flex;}}
.rej-badge{{background:{R};color:#fff;padding:3px 10px;border-radius:10px;font-size:11px;font-weight:700;}}
.rej-badge.pending{{background:{O};}}
.rej-label{{font-size:10px;color:{R};font-weight:600;max-width:120px;}}
.undo-btn{{margin-top:8px;padding:6px 12px;background:#313133;color:#fff;border:none;border-radius:4px;font-size:11px;font-weight:bold;cursor:pointer;}}
.meta{{font-size:11px;margin-top:8px;line-height:1.4;}}
.meta .nm{{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
.meta .br{{color:{O};font-weight:700;margin:2px 0;}}
.meta .ct{{color:#666;font-size:10px;}}
.meta .sl{{color:#999;font-size:9px;margin-top:4px;border-top:1px dashed #eee;padding-top:4px;}}
.acts{{display:flex;gap:4px;margin-top:8px;}}
.act-btn{{flex:1;padding:6px;font-size:11px;border:none;border-radius:4px;cursor:pointer;font-weight:700;color:#fff;background:{O};}}
.act-more{{flex:1;font-size:11px;border:1px solid #ccc;border-radius:4px;outline:none;cursor:pointer;background:#fff;}}
</style>
</head>
<body>
<div class="ctrl-bar">
  <span class="sel-count" id="sel-count">0 pending</span>
  <select class="reason-sel" id="batch-reason">
    <option value="REJECT_POOR_IMAGE">Poor Image</option>
    <option value="REJECT_WRONG_CAT">Wrong Category</option>
    <option value="REJECT_FAKE">Fake Product</option>
    <option value="REJECT_BRAND">Restricted Brand</option>
    <option value="REJECT_WRONG_BRAND">Wrong Brand</option>
    <option value="REJECT_PROHIBITED">Prohibited</option>
    <option value="REJECT_COLOR">Missing Color</option>
  </select>
  <button class="batch-btn" onclick="doBatchReject()">Batch Reject</button>
  <button class="desel-btn" onclick="doSelectAll()">Select All</button>
  <button class="desel-btn" onclick="doDeselAll()">Deselect All</button>
</div>
<div class="grid" id="card-grid"></div>
<script>
function esc(s){{return(s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}}
var CARDS={cards_json};
var COMMITTED={committed_json};
window._pqSel=window._pqSel||{{}};
window._pqStaged=window._pqStaged||{{}};
var sel=window._pqSel,staged=window._pqStaged;
function sendMsg(type,payload){{
  try{{
    var par=window.parent;
    var inputs=par.document.querySelectorAll('input[type="text"]');
    var bridge=null;
    for(var i=0;i<inputs.length;i++){{
      if(inputs[i].placeholder==='JTBRIDGE_UNIQUE_DO_NOT_USE'){{bridge=inputs[i];break;}}
    }}
    if(!bridge)return;
    var msg=JSON.stringify({{action:type,payload:payload}});
    bridge.focus({{preventScroll:true}});
    Object.getOwnPropertyDescriptor(par.HTMLInputElement.prototype,'value').set.call(bridge,msg);
    bridge.dispatchEvent(new par.Event('input',{{bubbles:true}}));
    setTimeout(function(){{bridge.blur();bridge.dispatchEvent(new par.KeyboardEvent('keydown',{{bubbles:true,cancelable:true,key:'Enter',keyCode:13}}));}},150);
  }}catch(e){{console.error('bridge error:',e);}}
}}
function upd(){{
  var n=Object.keys(sel).length+Object.keys(staged).length;
  document.getElementById('sel-count').textContent=n+' pending';
}}
function renderCard(c){{
  var sid=c.sid,img=esc(c.img);
  var isC=sid in COMMITTED,isS=sid in staged,isSel=!isC&&!isS&&(sid in sel);
  var cls='card'+(isC?' committed-rej':isS?' staged-rej':isSel?' selected':'');
  var warns=(c.warnings||[]).map(w=>'<span class="warn-badge">'+esc(w)+'</span>').join('');
  var price=c.price?'<div class="price-badge">'+esc(c.price)+'</div>':'';
  var zoom='<div class="zoom-btn" onclick="event.stopPropagation();toggleZoom(\''+sid+'\')">&#128269;</div>';
  var overlay='',acts='';
  if(isC){{
    overlay='<div class="rej-overlay"><div class="rej-badge">REJECTED</div><div class="rej-label">'+esc((COMMITTED[sid]||'').replace(/_/g,' '))+'</div><button class="undo-btn" onclick="event.stopPropagation();undoR(\''+sid+'\')">Undo</button></div>';
  }}else if(isS){{
    overlay='<div class="rej-overlay staged"><div class="rej-badge pending">PENDING</div><div class="rej-label">'+esc((staged[sid]||'').replace(/_/g,' '))+'</div><button class="undo-btn" onclick="event.stopPropagation();clrS(\''+sid+'\')">Clear</button></div>';
  }}else{{
    acts='<div class="acts"><button class="act-btn" onclick="event.stopPropagation();stageR(\''+sid+'\',\'REJECT_POOR_IMAGE\')">Poor Image</button><select class="act-more" onchange="if(this.value){{event.stopPropagation();stageR(\''+sid+'\',this.value);this.value=\'\'}}"><option value="">More…</option><option value="REJECT_WRONG_CAT">Wrong Category</option><option value="REJECT_FAKE">Fake</option><option value="REJECT_BRAND">Restricted Brand</option><option value="REJECT_PROHIBITED">Prohibited</option><option value="REJECT_COLOR">Missing Color</option><option value="REJECT_WRONG_BRAND">Wrong Brand</option></select></div>';
  }}
  var shortName=c.name.length>38?esc(c.name.slice(0,38))+'…':esc(c.name);
  return '<div class="'+cls+'" id="card-'+sid+'"><div class="img-wrap" onclick="toggleSel(\''+sid+'\',event)">'+price+'<div class="warn-wrap">'+warns+'</div><img class="card-img" src="'+img+'" onerror="this.src=\'https://via.placeholder.com/150?text=No+Image\'">'+zoom+overlay+'<div class="tick">&#10003;</div></div><div class="meta"><div class="nm" title="'+esc(c.name)+'">'+shortName+'</div><div class="br">'+esc(c.brand)+'</div><div class="ct">'+esc(c.cat)+'</div><div class="sl">'+esc(c.seller)+'</div></div>'+acts+'</div>';
}}
function repl(sid){{var el=document.getElementById('card-'+sid);if(!el)return;var c=CARDS.find(x=>x.sid===sid);if(c){{var t=document.createElement('div');t.innerHTML=renderCard(c);el.replaceWith(t.firstElementChild);}}}}
function toggleZoom(sid){{var img=document.querySelector('#card-'+sid+' .card-img');if(!img)return;if(img.classList.contains('zoomed')){{img.classList.remove('zoomed');img.closest('.card').style.zIndex='1';}}else{{document.querySelectorAll('.zoomed').forEach(e=>{{e.classList.remove('zoomed');if(e.closest('.card'))e.closest('.card').style.zIndex='1';}});img.classList.add('zoomed');img.closest('.card').style.zIndex='999';}}}}
function toggleSel(sid,e){{var img=document.querySelector('#card-'+sid+' .card-img');if(img&&img.classList.contains('zoomed')){{img.classList.remove('zoomed');img.closest('.card').style.zIndex='1';return;}}if(sid in COMMITTED)return;if(sid in staged){{delete staged[sid];}}else if(sid in sel){{delete sel[sid];}}else{{sel[sid]=true;}}repl(sid);upd();}}
function stageR(sid,r){{if(sid in sel)delete sel[sid];staged[sid]=r;repl(sid);upd();}}
function clrS(sid){{delete staged[sid];repl(sid);upd();}}
function undoR(sid){{sendMsg('undo',{{[sid]:true}});delete COMMITTED[sid];repl(sid);upd();}}
function doBatchReject(){{
  var br=document.getElementById('batch-reason').value,payload={{}},count=0;
  for(var s in staged){{payload[s]=staged[s];count++;}}
  for(var s in sel){{payload[s]=br;count++;}}
  if(!count)return;
  for(var s in payload){{COMMITTED[s]=payload[s];delete sel[s];delete staged[s];}}
  sendMsg('reject',payload);
  renderAll();upd();
}}
function doSelectAll(){{CARDS.forEach(c=>{{if(!(c.sid in COMMITTED)&&!(c.sid in staged))sel[c.sid]=true;}});renderAll();upd();}}
function doDeselAll(){{for(var k in sel)delete sel[k];for(var k in staged)delete staged[k];renderAll();upd();}}
function renderAll(){{document.getElementById('card-grid').innerHTML=CARDS.map(renderCard).join('');upd();}}
renderAll();
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# CALL THE IMAGE GRID
# Render whenever product data is available — works with or without
# validation results, and whether products are approved or rejected.
# ------------------------------------------------------------------
_grid_data_ready  = not st.session_state.pq_data.empty
_grid_report_ready = not st.session_state.pq_val_report.empty or not st.session_state.final_report.empty

if _grid_data_ready and _grid_report_ready:
    # Ensure final_report is populated even if validation block was skipped
    # (e.g. _FULL_VALIDATION_OK is False but enrichment ran)
    if st.session_state.final_report.empty and not st.session_state.pq_val_report.empty:
        st.session_state.final_report = st.session_state.pq_val_report.copy()
    _render_pq_image_grid()
