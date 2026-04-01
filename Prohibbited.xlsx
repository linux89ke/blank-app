#!/usr/bin/env python3
"""
generate_qc_report.py
─────────────────────
Auto-generate a weekly QC PowerPoint report from the Jumia validation app's
output data (final_report + all_data_map DataFrames, or exported Excel files).

USAGE
─────
  python generate_qc_report.py \
      --pim        "KE_PIM_Export_2026-03-31.xlsx" \
      --full       "KE_Full_2026-03-31.xlsx"  \
      --template   "QC_Report_8_2026.pptx"   \
      --output     "QC_Report_9_2026.pptx"

Or call generate_report() directly from code:
  from generate_qc_report import generate_report
  generate_report(final_report_df, all_data_df, template_path, output_path)

INPUTS accepted
───────────────
  final_report_df  ← ProductSets sheet columns:
      ProductSetSid, ParentSKU, Status, Reason, Comment, FLAG, SellerName
  all_data_df      ← Full Data sheet columns (superset of above):
      PRODUCT_SET_SID, ACTIVE_STATUS_COUNTRY, NAME, BRAND, CATEGORY,
      CATEGORY_CODE, FLAG, SELLER_NAME, … (all FULL_DATA_COLS)

WHAT GETS FILLED
────────────────
  Slide 1  – Title: "QC REPORT week N"
  Slide 2  – Seller Center: always 0s; prev-week row label only (Week N-1)
  Slide 3  – PIM QC: KE & UG Approved/Rejected/Total; prev-week label only
  Slide 4  – Total Work: same logic (PIM + SC)
  Slide 5  – Takeaways: auto-generated from KE data
  Slide 6  – Top Rejection Categories: KE Wk-N with counts; Wk N-1 label only
  Slide 7  – Top Rejection Reasons: KE & UG Wk-N; Wk N-1 label only
  Slide 8  – Insights: pattern-based paragraph
  Slide 9  – Top Rejected Sellers: KE top 5 / UG top 5
  Slide 10 – Backlog: "No backlog for week N" + blank Repeat Offenders table
"""

import re
import sys
import shutil
import zipfile
import argparse
import datetime
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# WEEK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def current_week_number() -> int:
    """ISO week number of today."""
    return datetime.date.today().isocalendar()[1]


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _country_rows(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Filter DataFrame to a specific country code (KE / UG)."""
    if "ACTIVE_STATUS_COUNTRY" not in df.columns:
        return df
    col = df["ACTIVE_STATUS_COUNTRY"].astype(str).str.strip().str.upper()
    col = col.str.replace(r"^JUMIA-", "", regex=True)
    return df[col == country_code].copy()


def _safe_int(val) -> int:
    try:
        return int(val)
    except Exception:
        return 0


def get_ke_ug_stats(final_report: pd.DataFrame, all_data: pd.DataFrame):
    """
    Returns dict with:
      ke_approved, ke_rejected, ke_total
      ug_approved, ug_rejected, ug_total
    Works whether ACTIVE_STATUS_COUNTRY is on final_report or all_data.
    """
    stats = {}

    for code, prefix in [("KE", "ke"), ("UG", "ug")]:
        # Try merging country info from all_data
        if "ACTIVE_STATUS_COUNTRY" in all_data.columns:
            country_df = _country_rows(all_data, code)
            sids = set(country_df["PRODUCT_SET_SID"].astype(str).str.strip())
            sub = final_report[
                final_report["ProductSetSid"].astype(str).str.strip().isin(sids)
            ]
        elif "ACTIVE_STATUS_COUNTRY" in final_report.columns:
            sub = _country_rows(final_report, code)
        else:
            sub = final_report  # fallback – treat everything as one country

        approved = int((sub["Status"] == "Approved").sum())
        rejected = int((sub["Status"] == "Rejected").sum())
        stats[f"{prefix}_approved"] = approved
        stats[f"{prefix}_rejected"] = rejected
        stats[f"{prefix}_total"]    = approved + rejected

    return stats


def get_top_rejection_categories(
    final_report: pd.DataFrame,
    all_data: pd.DataFrame,
    country_code: str,
    top_n: int = 5,
) -> list[tuple[str, int]]:
    """Top N rejection categories for a country, as [(category_name, count)]."""
    sids_rejected = set(
        final_report[final_report["Status"] == "Rejected"]["ProductSetSid"]
        .astype(str).str.strip()
    )
    if not sids_rejected:
        return []

    country_df = _country_rows(all_data, country_code)
    rej = country_df[
        country_df["PRODUCT_SET_SID"].astype(str).str.strip().isin(sids_rejected)
    ]

    cat_col = next(
        (c for c in ["CATEGORY", "CATEGORY_CODE"] if c in rej.columns), None
    )
    if cat_col is None:
        return []

    counts = (
        rej[cat_col]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(top_n)
    )
    return [(cat, int(cnt)) for cat, cnt in counts.items()]


def get_top_rejection_reasons(
    final_report: pd.DataFrame,
    all_data: pd.DataFrame,
    country_code: str,
    top_n: int = 5,
) -> list[str]:
    """Top N rejection FLAGs for a country."""
    sids_rejected = set(
        final_report[final_report["Status"] == "Rejected"]["ProductSetSid"]
        .astype(str).str.strip()
    )
    if not sids_rejected:
        return []

    country_df = _country_rows(all_data, country_code)
    country_sids = set(country_df["PRODUCT_SET_SID"].astype(str).str.strip())

    rej = final_report[
        final_report["ProductSetSid"].astype(str).str.strip().isin(
            sids_rejected & country_sids
        )
    ]

    flag_col = "FLAG" if "FLAG" in rej.columns else None
    if flag_col is None:
        return []

    counts = (
        rej[flag_col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
        .head(top_n)
    )
    return list(counts.index)


def get_top_rejected_sellers(
    final_report: pd.DataFrame,
    all_data: pd.DataFrame,
    country_code: str,
    top_n: int = 5,
) -> list[str]:
    """Top N sellers by rejection count for a country."""
    sids_rejected = set(
        final_report[final_report["Status"] == "Rejected"]["ProductSetSid"]
        .astype(str).str.strip()
    )
    if not sids_rejected:
        return []

    country_df = _country_rows(all_data, country_code)
    rej = country_df[
        country_df["PRODUCT_SET_SID"].astype(str).str.strip().isin(sids_rejected)
    ]

    seller_col = next(
        (c for c in ["SELLER_NAME", "SellerName"] if c in rej.columns), None
    )
    if seller_col is None:
        # Try from final_report
        rej2 = final_report[
            final_report["ProductSetSid"].astype(str).str.strip().isin(sids_rejected)
        ]
        seller_col2 = next(
            (c for c in ["SellerName", "SELLER_NAME"] if c in rej2.columns), None
        )
        if seller_col2 is None:
            return []
        counts = rej2[seller_col2].astype(str).str.strip().value_counts().head(top_n)
        return list(counts.index)

    counts = rej[seller_col].astype(str).str.strip().value_counts().head(top_n)
    return list(counts.index)


def build_takeaways(stats: dict, week_n: int) -> tuple[str, str]:
    """
    Build KE and UG takeaway sentences.
    Returns (ke_sentence, ug_sentence).
    Week N-1 figures are unknown (from prev run), so we phrase around week N data only.
    """
    ke_app  = stats["ke_approved"]
    ke_rej  = stats["ke_rejected"]
    ke_tot  = stats["ke_total"]
    ke_rate = round(ke_app / ke_tot * 100, 1) if ke_tot > 0 else 0.0

    ug_app  = stats["ug_approved"]
    ug_rej  = stats["ug_rejected"]
    ug_tot  = stats["ug_total"]
    ug_rate = round(ug_app / ug_tot * 100, 1) if ug_tot > 0 else 0.0

    ke_sent = (
        f"Volume for week {week_n}: {ke_tot:,} products reviewed "
        f"with an approval rate of {ke_rate}%"
    )
    ug_sent = (
        f"Volume for week {week_n}: {ug_tot:,} products reviewed "
        f"with an approval rate of {ug_rate}%"
    )
    return ke_sent, ug_sent


def build_insights(
    ke_reasons: list[str],
    ug_reasons: list[str],
    ke_categories: list[tuple[str, int]],
) -> str:
    """Build an insights paragraph from top rejection data."""
    recurring = []
    ug_set = {r.lower() for r in ug_reasons}
    for r in ke_reasons:
        if r.lower() in ug_set:
            recurring.append(r)

    if recurring:
        common = ", ".join(recurring[:3])
        insight = (
            f"{common} remain a persistent issue across both Kenya and Uganda "
            f"this week, continuing a trend seen in recent reporting periods."
        )
    elif ke_reasons:
        top = ke_reasons[0]
        insight = (
            f"{top} continues to be the leading rejection driver in Kenya "
            f"this week. Sellers should review listing guidelines to reduce rejections."
        )
    else:
        insight = (
            "Quality issues continue to require attention. "
            "Sellers are encouraged to review listing guidelines."
        )

    if ke_categories:
        top_cat = ke_categories[0][0]
        insight += (
            f" The category '{top_cat}' had the highest rejection volume in Kenya."
        )

    return insight


# ─────────────────────────────────────────────────────────────────────────────
# XML EDITING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    """XML-escape a plain string."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _plain_run(text: str, sz: int = 1800, bold: bool = False,
               color: Optional[str] = None,
               font: str = "Calibri") -> str:
    """Return a single <a:r> XML run."""
    b_attr   = ' b="1"' if bold else ""
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        if color else
        '<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
    )
    return (
        f'<a:r>'
        f'<a:rPr lang="en-GB" sz="{sz}"{b_attr} dirty="0">'
        f'{fill_xml}'
        f'<a:latin typeface="{font}" panose="020F0502020204030204"/>'
        f'<a:cs typeface="{font}" panose="020F0502020204030204"/>'
        f'</a:rPr>'
        f'<a:t>{_esc(text)}</a:t>'
        f'</a:r>'
    )


def _bullet_para(content_xml: str, sz: int = 1800) -> str:
    """Wrap content_xml in a bulleted paragraph."""
    return (
        '<a:p>'
        '<a:pPr marL="298450" indent="-285750">'
        '<a:spcBef><a:spcPts val="1720"/></a:spcBef>'
        '<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/>'
        '<a:buChar char="&#x2022;"/>'
        '</a:pPr>'
        + content_xml +
        f'<a:endParaRPr lang="en-GB" sz="{sz}" dirty="0">'
        '<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        '<a:cs typeface="Trebuchet MS" panose="020B0603020202020204"/>'
        '</a:endParaRPr>'
        '</a:p>'
    )


def _table_cell_text(text: str, sz: int = 1800, bold: bool = False,
                     color: Optional[str] = None, align: str = "l") -> str:
    """
    Return a <a:tc> XML element with single text run.
    align: 'l' | 'r' | 'ctr'
    """
    algn_attr = f' algn="{align}"' if align != "l" else ""
    b_attr    = ' b="1"' if bold else ""
    fill_xml  = (
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        if color else ""
    )
    font = "Arial" if bold else "Calibri"
    panose = "020B0604020202020204" if bold else "020F0502020204030204"
    return (
        '<a:tc>'
        '<a:txBody><a:bodyPr/><a:lstStyle/>'
        f'<a:p><a:pPr{algn_attr} fontAlgn="b"/>'
        f'<a:r>'
        f'<a:rPr lang="en-US" altLang="zh-CN" sz="{sz}"{b_attr} dirty="0">'
        f'{fill_xml}'
        f'<a:latin typeface="{font}" panose="{panose}"/>'
        f'<a:ea typeface="{font}" panose="{panose}"/>'
        f'</a:rPr>'
        f'<a:t>{_esc(text)}</a:t>'
        f'</a:r>'
        f'</a:p>'
        '</a:txBody>'
        '<a:tcPr marL="28892" marR="28892" marT="19367" marB="19367" anchor="b" anchorCtr="0">'
        '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
        '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
        '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
        '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
        '</a:tcPr>'
        '</a:tc>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE EDITORS
# ─────────────────────────────────────────────────────────────────────────────

def edit_slide1(xml: str, week_n: int) -> str:
    """Replace week number in title."""
    # The last <a:r> before </a:p> in the title contains "8" → replace with week_n
    return re.sub(
        r'(<a:t>)\s*8\s*(</a:t>)',
        rf'\g<1>{week_n}\2',
        xml,
        count=1,
    )


def edit_slide2(xml: str, week_n: int) -> str:
    """
    Seller Center – always all zeros.
    Replace 'Week 7' label with 'Week N-1'.
    The KE/UG data cells keep 0.
    """
    prev = week_n - 1
    xml = xml.replace(">Week 7<", f">Week {prev}<")
    xml = xml.replace(">Week 7 <", f">Week {prev} <")
    # Ensure the week-N-1 KE and UG cells show 0 (they already do in template)
    return xml


def edit_slide3(xml: str, week_n: int, stats: dict) -> str:
    """
    PIM QC table:
      Row 2 (Approved): KE={ke_approved}  UG={ug_approved}
      Row 3 (Rejected): KE={ke_rejected}  UG={ug_rejected}
      Row 5 (Total):    KE={ke_total}     UG={ug_total}
      Row 6 (prev wk):  label = "Week N-1" + BLANK cells (no figures)
    """
    prev = week_n - 1

    # Replace existing numbers with new values using targeted replacements
    # The template has: 19618, 8003, 5640, 2540, 25248, 10543 (and prev week 31598, 5883)
    replacements = {
        ">19618<": f">{stats['ke_approved']}<",
        ">8003<":  f">{stats['ug_approved']}<",
        ">5640<":  f">{stats['ke_rejected']}<",
        ">2540<":  f">{stats['ug_rejected']}<",
        ">25248<": f">{stats['ke_total']}<",
        ">10543<": f">{stats['ug_total']}<",
    }
    for old, new in replacements.items():
        xml = xml.replace(old, new)

    # Replace previous-week label
    xml = re.sub(r">Week\s+7<", f">Week {prev}<", xml)

    # Clear previous-week KE/UG figure cells (31598 and 5883 → blank)
    xml = xml.replace(">31598<", "><")
    xml = xml.replace(">5883<", "><")

    return xml


def edit_slide4(xml: str, week_n: int, stats: dict) -> str:
    """
    Total Work table:
      PIM row: KE={ke_total}  UG={ug_total}
      SC  row: 0  0
      Total:   KE={ke_total}  UG={ug_total}  (SC=0 so same as PIM)
      prev wk: label "Week N-1" + blank cells
    """
    prev = week_n - 1

    # Template values: PIM KE=25258, UG=10543; Total KE=25258, UG=10543;
    # prev week KE=31598, UG=5880
    replacements = {
        ">25258<": f">{stats['ke_total']}<",
        ">10543<": f">{stats['ug_total']}<",
    }
    for old, new in replacements.items():
        xml = xml.replace(old, new)

    xml = re.sub(r">Week\s+7<", f">Week {prev}<", xml)
    xml = xml.replace(">31598<", "><")
    xml = xml.replace(">5880<", "><")

    return xml


def edit_slide5(xml: str, week_n: int, ke_sentence: str, ug_sentence: str) -> str:
    """
    Takeaways slide – replace the body text box content.
    The text box (id=3) contains bulleted paragraphs for Kenya and Uganda.
    We rebuild the whole txBody content.
    """
    prev = week_n - 1

    # Build two bullet paragraphs
    ke_content = (
        _plain_run("Kenya : ", sz=1800, bold=True) +
        _plain_run(ke_sentence, sz=1800, bold=False)
    )
    ug_content = (
        _plain_run("Uganda : ", sz=1800, bold=True) +
        _plain_run(ug_sentence, sz=1800, bold=False)
    )

    new_body = (
        '<a:bodyPr vert="horz" wrap="square" lIns="0" tIns="139700" rIns="0" bIns="0" rtlCol="0">'
        '<a:noAutofit/>'
        '</a:bodyPr>'
        '<a:lstStyle/>'
        # empty first paragraph
        '<a:p><a:pPr marL="12700"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        '<a:spcBef><a:spcPts val="1100"/></a:spcBef></a:pPr>'
        '<a:endParaRPr sz="1800" dirty="0">'
        '<a:cs typeface="Trebuchet MS" panose="020B0603020202020204"/>'
        '</a:endParaRPr></a:p>'
        # empty bullet placeholder
        '<a:p><a:pPr marL="298450" indent="-285750">'
        '<a:spcBef><a:spcPts val="1720"/></a:spcBef>'
        '<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/>'
        '<a:buChar char="&#x2022;"/>'
        '</a:pPr>'
        '<a:endParaRPr lang="en-US" altLang="en-GB" sz="1800" b="1" dirty="0">'
        '<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        '<a:cs typeface="Trebuchet MS" panose="020B0603020202020204"/>'
        '</a:endParaRPr></a:p>'
        + _bullet_para(ke_content)
        + _bullet_para(ug_content)
    )

    # Replace the txBody of shape id=3 (the text content box, NOT the title)
    # Match the second <p:sp> txBody in slide5 (the content box)
    # Use a pattern that targets the specific shape by matching content
    pattern = (
        r'(<p:sp>.*?<p:cNvPr id="3".*?<p:txBody>)'
        r'.*?'
        r'(</p:txBody>\s*</p:sp>)'
    )
    replacement = r'\g<1>' + new_body + r'\g<2>'
    new_xml = re.sub(pattern, replacement, xml, count=1, flags=re.DOTALL)
    if new_xml == xml:
        # Fallback: replace the long takeaway paragraph block
        # Find between first and second bullet blocks in the content txBody
        pass

    return new_xml


def _build_category_table_rows(
    prev_week_n: int,
    curr_week_n: int,
    ke_cats: list[tuple[str, int]],
    ug_cats: list[tuple[str, int]],
) -> str:
    """
    Build the XML rows for slide 6 LEFT table (Wk N-1) — label only, no data.
    Returns full <a:tbl> XML content for both left (Wk N-1) and right (Wk N) tables.
    """
    # Left table: Wk N-1 header, Kenya label, 5 blank rows, spacer, Uganda label, 5 blank rows
    def blank_row(h=289560):
        return (
            f'<a:tr h="{h}">'
            + _table_cell_text("", sz=1400) + _table_cell_text("", sz=1400) +
            '</a:tr>'
        )

    def data_row(cat_name: str, count: int, h: int = 349885):
        return (
            f'<a:tr h="{h}">'
            + _table_cell_text(cat_name, sz=1400, bold=False) +
            _table_cell_text(str(count), sz=1400, bold=False, align="r") +
            '</a:tr>'
        )

    def header_row(label: str, color: str = "FF0000", h: int = 403859):
        cell = (
            '<a:tc>'
            '<a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr marL="57150">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="30"/></a:spcBef>'
            '</a:pPr>'
            f'<a:r><a:rPr sz="2400" spc="-5" dirty="0" err="1">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            '<a:latin typeface="Arial MT"/><a:cs typeface="Arial MT"/>'
            f'</a:rPr><a:t>{_esc(label)}</a:t></a:r>'
            '</a:p></a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="3810" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr>'
            '</a:tc>'
        )
        empty_cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
            '<a:endParaRPr sz="1300"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr></a:tc>'
        )
        return f'<a:tr h="{h}">' + cell + empty_cell + '</a:tr>'

    def country_subheader(label: str, h: int = 403860):
        cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            f'<a:p><a:pPr marL="57150">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="530"/></a:spcBef>'
            '</a:pPr>'
            f'<a:r><a:rPr sz="2000" spc="-5" dirty="0">'
            '<a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>'
            '<a:latin typeface="Arial MT"/><a:cs typeface="Arial MT"/>'
            f'</a:rPr><a:t>{_esc(label)}</a:t></a:r>'
            '</a:p></a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="67310" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/>'
            '<a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/>'
            '</a:lnB></a:tcPr></a:tc>'
        )
        empty_cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
            '<a:endParaRPr sz="1300" dirty="0"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/>'
            '<a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/>'
            '</a:lnB></a:tcPr></a:tc>'
        )
        return f'<a:tr h="{h}">' + cell + empty_cell + '</a:tr>'

    # LEFT TABLE: Wk N-1 (label only, no figures)
    left_rows = (
        header_row(f"Wk {prev_week_n}")
        + country_subheader("Kenya")
        + "".join(blank_row() for _ in range(5))
        + blank_row()  # spacer
        + country_subheader("Uganda")
        + "".join(blank_row() for _ in range(5))
    )

    # RIGHT TABLE: Wk N (with actual data)
    ke_rows_xml = ""
    for i, (cat, cnt) in enumerate(ke_cats[:5]):
        h = 349885 if i == 0 else 289560
        ke_rows_xml += data_row(cat, cnt, h)
    # Pad to 5 rows
    for _ in range(5 - len(ke_cats[:5])):
        ke_rows_xml += blank_row()

    ug_rows_xml = ""
    for i, (cat, cnt) in enumerate(ug_cats[:5]):
        h = 289560
        ug_rows_xml += data_row(cat, cnt, h)
    for _ in range(5 - len(ug_cats[:5])):
        ug_rows_xml += blank_row()

    right_header_cell = (
        '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr marL="57150">'
        '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        '<a:spcBef><a:spcPts val="30"/></a:spcBef>'
        '</a:pPr>'
        f'<a:r><a:rPr sz="2400" spc="-5" dirty="0" err="1">'
        '<a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>'
        '<a:latin typeface="Arial MT"/><a:cs typeface="Arial MT"/>'
        f'</a:rPr><a:t>Wk {curr_week_n}</a:t></a:r>'
        '</a:p></a:txBody>'
        '<a:tcPr marL="0" marR="0" marT="3810" marB="0">'
        '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
        '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
        '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
        '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
        '</a:tcPr>'
        '</a:tc>'
    )
    right_empty_cell = (
        '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
        '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
        '<a:endParaRPr sz="1300"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
        '</a:txBody>'
        '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
        '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
        '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
        '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
        '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
        '</a:tcPr></a:tc>'
    )

    right_rows = (
        f'<a:tr h="403859">{right_header_cell}{right_empty_cell}</a:tr>'
        + country_subheader("Kenya")
        + ke_rows_xml
        + blank_row()   # spacer
        + country_subheader("Uganda")
        + ug_rows_xml
    )

    return left_rows, right_rows


def edit_slide6(
    xml: str,
    week_n: int,
    ke_cats: list[tuple[str, int]],
    ug_cats: list[tuple[str, int]],
) -> str:
    """Top Rejection Categories – replace table content."""
    prev = week_n - 1
    left_rows, right_rows = _build_category_table_rows(prev, week_n, ke_cats, ug_cats)

    # Replace LEFT table (graphicFrame id=3) tbl content
    left_tbl = (
        '<a:tblPr firstRow="1" bandRow="1">'
        '<a:tableStyleId>{2D5ABB26-0587-4C30-8999-92F81FD0307C}</a:tableStyleId>'
        '</a:tblPr>'
        '<a:tblGrid>'
        '<a:gridCol w="1913890"/>'
        '<a:gridCol w="850900"/>'
        '</a:tblGrid>'
        + left_rows
    )
    right_tbl = (
        '<a:tblPr firstRow="1" bandRow="1">'
        '<a:tableStyleId>{2D5ABB26-0587-4C30-8999-92F81FD0307C}</a:tableStyleId>'
        '</a:tblPr>'
        '<a:tblGrid>'
        '<a:gridCol w="1913889"/>'
        '<a:gridCol w="850900"/>'
        '</a:tblGrid>'
        + right_rows
    )

    # Replace first <a:tbl>…</a:tbl> with left_tbl, second with right_tbl
    tables = re.findall(r'<a:tbl>.*?</a:tbl>', xml, flags=re.DOTALL)
    if len(tables) >= 2:
        xml = xml.replace(tables[0], f'<a:tbl>{left_tbl}</a:tbl>', 1)
        xml = xml.replace(tables[1], f'<a:tbl>{right_tbl}</a:tbl>', 1)

    return xml


def _build_reasons_table_rows(
    prev_week_n: int,
    curr_week_n: int,
    ke_reasons: list[str],
    ug_reasons: list[str],
    is_right: bool,
) -> str:
    """Build rows for slide 7 tables."""

    def reason_row(text: str, h: int = 289560):
        cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            f'<a:p><a:r>'
            f'<a:rPr lang="en-US" altLang="zh-CN" sz="1100"/>'
            f'<a:t>{_esc(text)}</a:t>'
            '</a:r></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0" anchor="ctr" anchorCtr="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr></a:tc>'
        )
        empty = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
            '<a:endParaRPr sz="1200"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
            '<a:lnL w="9525" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/>'
            '<a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/>'
            '</a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr></a:tc>'
        )
        return f'<a:tr h="{h}">' + cell + empty + '</a:tr>'

    def blank_reason_row(h=289560):
        return reason_row("", h)

    def header_row_reasons(label: str, h=403859):
        cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr marL="57150">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="30"/></a:spcBef></a:pPr>'
            f'<a:r><a:rPr sz="2400" spc="-5" dirty="0" err="1">'
            '<a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>'
            '<a:latin typeface="Arial MT"/><a:cs typeface="Arial MT"/>'
            f'</a:rPr><a:t>{_esc(label)}</a:t></a:r>'
            '</a:p></a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="3810" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr></a:tc>'
        )
        empty = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
            '<a:endParaRPr sz="1200"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnB>'
            '</a:tcPr></a:tc>'
        )
        return f'<a:tr h="{h}">' + cell + empty + '</a:tr>'

    def country_sub(label, h=403860):
        cell = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr marL="57150">'
            '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="530"/></a:spcBef></a:pPr>'
            f'<a:r><a:rPr sz="2000" spc="-5" dirty="0">'
            '<a:solidFill><a:srgbClr val="FF0000"/></a:solidFill>'
            '<a:latin typeface="Arial MT"/><a:cs typeface="Arial MT"/>'
            f'</a:rPr><a:t>{_esc(label)}</a:t></a:r>'
            '</a:p></a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="67310" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/>'
            '<a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/>'
            '</a:lnB></a:tcPr></a:tc>'
        )
        empty = (
            '<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>'
            '<a:p><a:pPr><a:lnSpc><a:spcPct val="100000"/></a:lnSpc></a:pPr>'
            '<a:endParaRPr sz="1200"><a:latin typeface="Times New Roman"/></a:endParaRPr></a:p>'
            '</a:txBody>'
            '<a:tcPr marL="0" marR="0" marT="0" marB="0">'
            '<a:lnL w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnL>'
            '<a:lnR w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnR>'
            '<a:lnT w="9525"><a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/></a:lnT>'
            '<a:lnB w="9525" cap="flat" cmpd="sng" algn="ctr">'
            '<a:solidFill><a:srgbClr val="CCCCCC"/></a:solidFill><a:prstDash val="solid"/>'
            '<a:round/><a:headEnd type="none" w="med" len="med"/><a:tailEnd type="none" w="med" len="med"/>'
            '</a:lnB></a:tcPr></a:tc>'
        )
        return f'<a:tr h="{h}">' + cell + empty + '</a:tr>'

    if not is_right:
        # Left = Wk N-1, label only, blanks
        rows = (
            header_row_reasons(f"Wk {prev_week_n}")
            + country_sub("Kenya")
            + "".join(blank_reason_row() for _ in range(5))
            + blank_reason_row()  # spacer
            + country_sub("Uganda")
            + "".join(blank_reason_row() for _ in range(5))
        )
    else:
        # Right = Wk N, actual data
        ke_xml = ""
        for i, r in enumerate(ke_reasons[:5]):
            h = 373380 if i < 2 else 289560
            ke_xml += reason_row(r, h)
        for _ in range(5 - len(ke_reasons[:5])):
            ke_xml += blank_reason_row()

        ug_xml = ""
        for r in ug_reasons[:5]:
            ug_xml += reason_row(r)
        for _ in range(5 - len(ug_reasons[:5])):
            ug_xml += blank_reason_row()

        rows = (
            header_row_reasons(f"Wk {curr_week_n}")
            + country_sub("Kenya")
            + ke_xml
            + blank_reason_row()
            + country_sub("Uganda")
            + ug_xml
        )

    return rows


def edit_slide7(
    xml: str,
    week_n: int,
    ke_reasons: list[str],
    ug_reasons: list[str],
) -> str:
    """Top Rejection Reasons."""
    prev = week_n - 1

    left_rows  = _build_reasons_table_rows(prev, week_n, ke_reasons, ug_reasons, is_right=False)
    right_rows = _build_reasons_table_rows(prev, week_n, ke_reasons, ug_reasons, is_right=True)

    left_tbl = (
        '<a:tblPr firstRow="1" bandRow="1">'
        '<a:tableStyleId>{2D5ABB26-0587-4C30-8999-92F81FD0307C}</a:tableStyleId>'
        '</a:tblPr>'
        '<a:tblGrid>'
        '<a:gridCol w="2885402"/>'
        '<a:gridCol w="306426"/>'
        '</a:tblGrid>'
        + left_rows
    )
    right_tbl = (
        '<a:tblPr firstRow="1" bandRow="1">'
        '<a:tableStyleId>{2D5ABB26-0587-4C30-8999-92F81FD0307C}</a:tableStyleId>'
        '</a:tblPr>'
        '<a:tblGrid>'
        '<a:gridCol w="3607827"/>'
        '<a:gridCol w="293931"/>'
        '</a:tblGrid>'
        + right_rows
    )

    tables = re.findall(r'<a:tbl>.*?</a:tbl>', xml, flags=re.DOTALL)
    if len(tables) >= 2:
        xml = xml.replace(tables[0], f'<a:tbl>{left_tbl}</a:tbl>', 1)
        xml = xml.replace(tables[1], f'<a:tbl>{right_tbl}</a:tbl>', 1)

    return xml


def edit_slide8(xml: str, insights_text: str) -> str:
    """Replace insights bullet text."""
    # The bullet run is in the first (and only) content text box
    # Replace existing text between <a:t> tags in the bullet paragraph
    new_run = (
        f'<a:r>'
        f'<a:rPr lang="en-US" altLang="en-GB" sz="1800" dirty="0">'
        f'<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        f'<a:cs typeface="Trebuchet MS" panose="020B0603020202020204"/>'
        f'</a:rPr>'
        f'<a:t>{_esc(insights_text)}</a:t>'
        f'</a:r>'
    )

    # Replace the entire paragraph that has the bullet content
    new_para = (
        '<a:p>'
        '<a:pPr marL="298450" indent="-285750">'
        '<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        '<a:buFont typeface="Arial" panose="020B0604020202020204" pitchFamily="34" charset="0"/>'
        '<a:buChar char="&#x2022;"/>'
        '<a:tabLst>'
        '<a:tab pos="297815" algn="l"/>'
        '<a:tab pos="298450" algn="l"/>'
        '</a:tabLst>'
        '</a:pPr>'
        + new_run +
        '<a:endParaRPr lang="en-GB" altLang="en-US" sz="1800" dirty="0">'
        '<a:solidFill><a:schemeClr val="tx1"/></a:solidFill>'
        '<a:cs typeface="Trebuchet MS" panose="020B0603020202020204"/>'
        '</a:endParaRPr>'
        '</a:p>'
    )

    # Replace the second <a:p> (the bullet paragraph) in the content textbox
    # Identify the text box (id=2, not the title which is id=3)
    # Use regex to find the content shape and replace its second paragraph
    pattern = (
        r'(<p:sp>.*?<p:cNvPr id="2".*?<p:txBody>.*?</a:p>)'  # up to end of first para
        r'(<a:p>.*?</a:p>)'                                    # the bullet para to replace
        r'(.*?</p:txBody>)'
    )
    replacement = r'\g<1>' + new_para + r'\g<3>'
    new_xml = re.sub(pattern, replacement, xml, count=1, flags=re.DOTALL)
    if new_xml == xml:
        # Simple fallback: replace first bullet text
        xml = re.sub(
            r'Missing COLOR.*?remain a major issue.*?year\s*',
            _esc(insights_text) + ' ',
            xml,
            count=1,
            flags=re.DOTALL,
        )
        return xml

    return new_xml


def edit_slide9(
    xml: str,
    ke_sellers: list[str],
    ug_sellers: list[str],
) -> str:
    """Top Rejected Sellers – fill KE and UG columns."""
    # Pad to 5
    ke = (ke_sellers + [""] * 5)[:5]
    ug = (ug_sellers + [""] * 5)[:5]

    # The template has 5 data rows after the header
    # KE old values: Schola, Jovic Optometry Center, Blesteve Solutions, Aksoft Technologies, Anticipater
    # UG old values: ml gadgets, JANAMUz collection, First Goods, Ephraim Medical And Laboratory Supplies, Bena enterprises
    ke_old = [
        "Schola", "Jovic Optometry Center", "Blesteve Solutions",
        "Aksoft Technologies", "Anticipater"
    ]
    ug_old = [
        "ml gadgets", "JANAMUz collection", "First Goods",
        "Ephraim Medical And Laboratory Supplies", "Bena enterprises"
    ]

    for old, new in zip(ke_old, ke):
        xml = xml.replace(f">{_esc(old)}<", f">{_esc(new)}<", 1)
    for old, new in zip(ug_old, ug):
        xml = xml.replace(f">{_esc(old)}<", f">{_esc(new)}<", 1)

    return xml


def edit_slide10(xml: str, week_n: int) -> str:
    """Backlog: replace week number in bullet."""
    xml = re.sub(
        r'No backlog for week \d+',
        f'No backlog for week {week_n}',
        xml,
    )
    # Also clear the repeat offenders table rows (leave headers, blank data)
    # The table has sellers from prev weeks — clear them
    old_entries = [
        "Alex Mwailu (Week 5 &amp; 7)",
        "Karl Wilhelm (Week 5 &amp; 6)",
        "Aksoft Technologies (Week 7 &amp; 8)",
        "Blesteve Solutions (Week 7 &amp; 8)",
        "ml gadgets (Week 5 &amp; 8)",
        "Mi gadgets (Week 3 &amp; 6)",
        "MD Investments (Week 4 &amp; 6)",
    ]
    for entry in old_entries:
        xml = xml.replace(f">{entry}<", "><")

    return xml


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    final_report: pd.DataFrame,
    all_data: pd.DataFrame,
    template_path: str,
    output_path: str,
    week_n: Optional[int] = None,
):
    """
    Main entry point.

    Parameters
    ----------
    final_report  : DataFrame with columns ProductSetSid, Status, FLAG, SellerName, ...
    all_data      : DataFrame with columns PRODUCT_SET_SID, ACTIVE_STATUS_COUNTRY,
                    NAME, BRAND, CATEGORY, SELLER_NAME, ...
    template_path : Path to the Week 8 PPTX template
    output_path   : Where to save the new PPTX
    week_n        : Override week number (defaults to current ISO week)
    """
    if week_n is None:
        week_n = current_week_number()

    print(f"Generating QC Report for Week {week_n}...")

    # ── Compute stats ──────────────────────────────────────────────────────
    stats = get_ke_ug_stats(final_report, all_data)
    print(f"  KE: {stats['ke_approved']} approved / {stats['ke_rejected']} rejected "
          f"/ {stats['ke_total']} total")
    print(f"  UG: {stats['ug_approved']} approved / {stats['ug_rejected']} rejected "
          f"/ {stats['ug_total']} total")

    ke_cats    = get_top_rejection_categories(final_report, all_data, "KE")
    ug_cats    = get_top_rejection_categories(final_report, all_data, "UG")
    ke_reasons = get_top_rejection_reasons(final_report, all_data, "KE")
    ug_reasons = get_top_rejection_reasons(final_report, all_data, "UG")
    ke_sellers = get_top_rejected_sellers(final_report, all_data, "KE")
    ug_sellers = get_top_rejected_sellers(final_report, all_data, "UG")

    ke_takeaway, ug_takeaway = build_takeaways(stats, week_n)
    insights = build_insights(ke_reasons, ug_reasons, ke_cats)

    print(f"  Top KE categories : {[c for c, _ in ke_cats]}")
    print(f"  Top UG categories : {[c for c, _ in ug_cats]}")
    print(f"  Top KE reasons    : {ke_reasons}")
    print(f"  Top UG reasons    : {ug_reasons}")
    print(f"  Top KE sellers    : {ke_sellers}")
    print(f"  Top UG sellers    : {ug_sellers}")

    # ── Unpack PPTX ────────────────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(template_path, 'r') as z:
        z.extractall(tmp_dir)

    slides_dir = tmp_dir / "ppt" / "slides"

    editors = {
        1: lambda x: edit_slide1(x, week_n),
        2: lambda x: edit_slide2(x, week_n),
        3: lambda x: edit_slide3(x, week_n, stats),
        4: lambda x: edit_slide4(x, week_n, stats),
        5: lambda x: edit_slide5(x, week_n, ke_takeaway, ug_takeaway),
        6: lambda x: edit_slide6(x, week_n, ke_cats, ug_cats),
        7: lambda x: edit_slide7(x, week_n, ke_reasons, ug_reasons),
        8: lambda x: edit_slide8(x, insights),
        9: lambda x: edit_slide9(x, ke_sellers, ug_sellers),
        10: lambda x: edit_slide10(x, week_n),
    }

    for slide_num, editor in editors.items():
        slide_path = slides_dir / f"slide{slide_num}.xml"
        if not slide_path.exists():
            print(f"  WARNING: slide{slide_num}.xml not found — skipping")
            continue
        content = slide_path.read_text(encoding="utf-8")
        content = editor(content)
        slide_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Slide {slide_num} edited")

    # ── Repack PPTX ────────────────────────────────────────────────────────
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path_obj, 'w', zipfile.ZIP_DEFLATED) as zout:
        for file_path in sorted(tmp_dir.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(tmp_dir)
                zout.write(file_path, arcname)

    shutil.rmtree(tmp_dir)
    print(f"\n✅ Report saved to: {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def load_dataframes(pim_path: str, full_path: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PIM Export + Full Data from Excel files."""
    print(f"Loading PIM export: {pim_path}")
    pim_df = pd.read_excel(pim_path, sheet_name="ProductSets", dtype=str)

    if full_path:
        print(f"Loading Full data: {full_path}")
        full_df = pd.read_excel(full_path, sheet_name="ProductSets", dtype=str)
        # Try to also get the Full Data sheet
        try:
            full_df2 = pd.read_excel(full_path, dtype=str)
            all_data = full_df2
        except Exception:
            all_data = full_df
    else:
        # Use PIM export as both
        all_data = pim_df.copy()
        # Rename columns to match all_data schema if needed
        col_map = {
            "ProductSetSid": "PRODUCT_SET_SID",
            "SellerName": "SELLER_NAME",
        }
        all_data = all_data.rename(columns=col_map)

    return pim_df, all_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate weekly QC PowerPoint report from Jumia app exports."
    )
    parser.add_argument("--pim",      required=True,  help="PIM Export Excel file (ProductSets sheet)")
    parser.add_argument("--full",     default=None,   help="Full Data Excel file (optional, for category/seller detail)")
    parser.add_argument("--template", required=True,  help="Template PPTX (Week 8 fixed)")
    parser.add_argument("--output",   required=True,  help="Output PPTX path")
    parser.add_argument("--week",     type=int, default=None, help="Override week number (default: auto-detect)")

    args = parser.parse_args()

    final_report, all_data = load_dataframes(args.pim, args.full)

    generate_report(
        final_report=final_report,
        all_data=all_data,
        template_path=args.template,
        output_path=args.output,
        week_n=args.week,
    )


if __name__ == "__main__":
    main()
