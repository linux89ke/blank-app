import streamlit as st
import pandas as pd
import io
import copy
import requests
from PIL import Image
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

st.set_page_config(page_title="Decathlon Product Lookup", page_icon="🏅", layout="wide")

st.markdown("""
<style>
h1 { color: #0082C3; }
.tag {
    display: inline-block; background: #0082C3; color: white;
    border-radius: 4px; padding: 2px 8px; font-size: 12px; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏅 Decathlon Product Lookup")
st.markdown("Search by model number or product name — view details, images, and **download a filled upload template**.")

IMAGE_COLS = ["OG_image"] + [f"picture_{i}" for i in range(1, 11)]
TEMPLATE_PATH = "product-creation-template.xlsx"

# ── Mapping: master file columns → template English columns ──────────────────
# Template English columns (no _AR / _FR):
# Name, Description, SellerSKU, ParentSKU, Brand, PrimaryCategory, AdditionalCategory,
# GTIN_Barcode, Price_KES, Sale_Price_KES, Sale_Price_Start_At, Sale_Price_End_At,
# Stock, variation, certifications, color, color_family, machine_type, main_material,
# manufacturer_txt, material_family, model, note, package_content, product_line,
# product_measures, product_warranty, product_weight, production_country,
# short_description, sport_type, warranty_address, warranty_duration,
# warranty_type, youtube_id, MainImage, Image2..Image8

MASTER_TO_TEMPLATE = {
    "product_name":      "Name",
    "designed_for":      "Description",
    "sku_num_sku_r3":    "SellerSKU",
    "model_code":        "ParentSKU",
    "brand_name":        "Brand",
    "department_label":  "PrimaryCategory",
    "nature_label":      "AdditionalCategory",
    "bar_code":          "GTIN_Barcode",
    "color":             "color",
    "color":             "color",
    "model_label":       "model",
    "keywords":          "note",
    "weight":            "product_weight",
    "OG_image":          "MainImage",
    "picture_1":         "Image2",
    "picture_2":         "Image3",
    "picture_3":         "Image4",
    "picture_4":         "Image5",
    "picture_5":         "Image6",
    "picture_6":         "Image7",
    "picture_7":         "Image8",
}

# All template English columns in order (no _AR/_FR)
TEMPLATE_ENG_COLS = [
    "Name", "Description", "SellerSKU", "ParentSKU", "Brand",
    "PrimaryCategory", "AdditionalCategory", "GTIN_Barcode",
    "Price_KES", "Sale_Price_KES", "Sale_Price_Start_At", "Sale_Price_End_At",
    "Stock", "variation", "certifications", "color", "color_family",
    "machine_type", "main_material", "manufacturer_txt", "material_family",
    "model", "note", "package_content", "product_line", "product_measures",
    "product_warranty", "product_weight", "production_country",
    "short_description", "sport_type", "warranty_address",
    "warranty_duration", "warranty_type", "youtube_id",
    "MainImage", "Image2", "Image3", "Image4", "Image5",
    "Image6", "Image7", "Image8",
]

# ── Load master data ──────────────────────────────────────────────────────────
@st.cache_data
def load_master(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes), dtype=str)

with st.sidebar:
    st.header("📂 Master Data File")
    uploaded_master = st.file_uploader("Replace bundled file (.xlsx)", type=["xlsx"])
    st.markdown("---")
    st.header("🔎 Search Fields")
    search_fields = st.multiselect(
        "Match terms against",
        ["model_code", "model_label", "product_name", "sku_num_sku_r3", "Jumia SKU", "bar_code"],
        default=["model_code", "model_label", "product_name"]
    )
    st.markdown("---")
    show_images = st.checkbox("Show product images", value=True)
    max_images  = st.slider("Max images per product", 1, 11, 5)

MASTER_PATH = "Decathlon_Working_File_Split.xlsx"
if uploaded_master:
    df_master = load_master(uploaded_master.read())
    st.sidebar.success(f"✅ {len(df_master):,} rows loaded")
else:
    try:
        df_master = load_master(open(MASTER_PATH, "rb").read())
        st.sidebar.info(f"📋 Bundled file · {len(df_master):,} rows")
    except FileNotFoundError:
        st.error("⚠️ No master file found. Upload one in the sidebar.")
        st.stop()

img_cols_present = [c for c in IMAGE_COLS if c in df_master.columns]
data_cols = [c for c in df_master.columns if c not in img_cols_present]

# ── Build filled template ─────────────────────────────────────────────────────
def build_template(results_df: pd.DataFrame) -> bytes:
    """
    Opens the original template, preserves all sheets & styling,
    then fills in 'Upload Template' with English-only data from results_df.
    """
    wb = load_workbook(TEMPLATE_PATH)
    ws = wb["Upload Template"]

    # Read header row to get column positions
    header_map = {}  # col_name -> col_index (1-based)
    for col_idx in range(1, ws.max_column + 1):
        val = ws.cell(row=1, column=col_idx).value
        if val:
            header_map[val] = col_idx

    # Copy header row style so data rows match
    header_font   = ws.cell(row=1, column=1).font
    data_font     = Font(name=header_font.name or "Calibri", size=header_font.size or 11)
    data_alignment = Alignment(vertical="center")

    # Write data rows starting at row 2
    for row_idx, (_, src_row) in enumerate(results_df.iterrows(), start=2):
        # Build a dict of template_col -> value from master row
        row_data = {}
        for master_col, tmpl_col in MASTER_TO_TEMPLATE.items():
            val = src_row.get(master_col, "")
            if pd.notna(val) and str(val).strip():
                row_data[tmpl_col] = str(val).strip()

        # Write each mapped value into the correct template column
        for tmpl_col, value in row_data.items():
            if tmpl_col in header_map:
                cell = ws.cell(row=row_idx, column=header_map[tmpl_col])
                cell.value = value
                cell.font = data_font
                cell.alignment = data_alignment

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()

# ── Input ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload a List", "⌨️ Manual Entry"])
queries = []

with tab1:
    uploaded_list = st.file_uploader(
        "Upload file with model numbers / product names",
        type=["xlsx", "csv", "txt"],
        help="One value per row. For Excel/CSV, values must be in column A."
    )
    if uploaded_list:
        ext = uploaded_list.name.rsplit(".", 1)[-1].lower()
        if ext == "txt":
            queries = [l.strip() for l in uploaded_list.read().decode().splitlines() if l.strip()]
        elif ext == "csv":
            q_df = pd.read_csv(uploaded_list, header=None, dtype=str)
            queries = q_df.iloc[:, 0].dropna().str.strip().tolist()
        else:
            q_df = pd.read_excel(uploaded_list, header=None, dtype=str)
            queries = q_df.iloc[:, 0].dropna().str.strip().tolist()
        st.success(f"Loaded **{len(queries)}** search terms")

with tab2:
    manual = st.text_area(
        "Enter one model number or product name per line",
        height=160,
        placeholder="8641696\nKEEPDRY DDY LS BLACK\n4271703"
    )
    if manual.strip():
        queries = [q.strip() for q in manual.strip().splitlines() if q.strip()]

# ── Search ────────────────────────────────────────────────────────────────────
def search(q: str) -> pd.DataFrame:
    mask = pd.Series(False, index=df_master.index)
    for field in search_fields:
        if field in df_master.columns:
            mask |= df_master[field].fillna("").str.lower().str.contains(q.lower(), regex=False)
    return df_master[mask].copy()

if queries:
    st.markdown("---")

    all_result_frames = []
    no_match = []

    for q in queries:
        res = search(q)
        if res.empty:
            no_match.append(q)
        else:
            res.insert(0, "Search Term", q)
            all_result_frames.append((q, res))

    if no_match:
        st.warning(f"No matches found for: **{', '.join(no_match)}**")

    if all_result_frames:
        total_rows = sum(len(r) for _, r in all_result_frames)
        st.success(f"**{total_rows} rows** matched across **{len(all_result_frames)}** query(ies)")

        combined = pd.concat([r for _, r in all_result_frames], ignore_index=True)

        # ── Download buttons side by side ──
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            raw_out = io.BytesIO()
            with pd.ExcelWriter(raw_out, engine="openpyxl") as writer:
                combined.to_excel(writer, index=False, sheet_name="Results")
            st.download_button(
                "⬇️ Download Raw Results (.xlsx)",
                data=raw_out.getvalue(),
                file_name="decathlon_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col_dl2:
            try:
                tpl_bytes = build_template(combined)
                st.download_button(
                    "📋 Download Filled Upload Template (.xlsx)",
                    data=tpl_bytes,
                    file_name="decathlon_upload_template_filled.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary"
                )
            except FileNotFoundError:
                st.warning("Template file not found. Place `product-creation-template.xlsx` in the app folder.")

        st.markdown("---")

        # ── Per-query results ──
        for q, res in all_result_frames:
            with st.expander(f"🔍 **{q}**  —  {len(res)} row(s)", expanded=True):

                show_cols = ["Search Term"] + [c for c in data_cols if c in res.columns]
                st.dataframe(
                    res[show_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "keywords":     st.column_config.TextColumn("keywords", width="large"),
                        "product_name": st.column_config.TextColumn("product_name", width="large"),
                        "designed_for": st.column_config.TextColumn("designed_for", width="large"),
                    }
                )

                # Category tags
                cats = set()
                for _, row in res.iterrows():
                    for col in ["department_label", "nature_label", "family", "type"]:
                        val = row.get(col, "")
                        if pd.notna(val) and str(val).strip():
                            cats.add(str(val).strip())
                if cats:
                    tags = " ".join(f'<span class="tag">{c}</span>' for c in sorted(cats))
                    st.markdown(f"**Categories & Types:** {tags}", unsafe_allow_html=True)

                # Images
                if show_images and img_cols_present:
                    first_row = res.iloc[0]
                    img_urls = [
                        str(first_row[c]) for c in img_cols_present
                        if pd.notna(first_row.get(c)) and str(first_row.get(c, "")).startswith("http")
                    ][:max_images]
                    if img_urls:
                        st.markdown("**🖼 Product Images**")
                        cols = st.columns(len(img_urls))
                        for i, url in enumerate(img_urls):
                            try:
                                resp = requests.get(url, timeout=6)
                                img = Image.open(io.BytesIO(resp.content))
                                cols[i].image(img, caption="Main" if i == 0 else f"View {i}", use_container_width=True)
                            except Exception:
                                cols[i].markdown(f"[🔗 Image {i+1}]({url})")
else:
    st.info("👆 Upload a list or type search terms above to get started.")

st.markdown("---")
st.caption("Decathlon Product Lookup · Powered by your Decathlon working file")
