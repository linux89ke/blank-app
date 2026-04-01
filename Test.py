import streamlit as st
import pandas as pd
import datetime
import re
import zipfile
from io import BytesIO
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PIM Weekly Analysis Dashboard", page_icon=":material/analytics:", layout="wide")

# --- CONSTANTS & MAPPINGS ---
COUNTRY_MAP = {
    "KE": "Kenya", "UG": "Uganda", "NG": "Nigeria", "GH": "Ghana",
    "MA": "Morocco", "MO": "Morocco", "EG": "Egypt", "CI": "Ivory Coast",
    "SN": "Senegal", "ZA": "South Africa"
}

# --- HELPER FUNCTIONS ---
def parse_file_metadata(filename):
    prefix = filename[:2].upper()
    country = COUNTRY_MAP.get(prefix, "Unknown Country")
    
    date_obj = None
    week_num = None
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        date_obj = datetime.datetime.strptime(match.group(), '%Y-%m-%d')
        week_num = date_obj.isocalendar()[1]
        
    return country, date_obj, week_num

def get_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def generate_excel_report(daily_summary, seller_stats, top_reasons, top_categories, metadata):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        cover_df = pd.DataFrame(list(metadata.items()), columns=['Metric', 'Value'])
        cover_df.to_excel(writer, sheet_name='Cover Sheet', index=False)
        
        worksheet = writer.sheets['Cover Sheet']
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 35)

        daily_summary.to_excel(writer, sheet_name='Daily & Weekly Summary')
        seller_stats.to_excel(writer, sheet_name='Top Rejected Sellers')
        top_reasons.to_excel(writer, sheet_name='Top Rejection Reasons')
        top_categories.to_excel(writer, sheet_name='Top Rejected Categories')
        
    return output.getvalue()

def process_dataframe(df, filename, all_data_list):
    country, file_date, week_num = parse_file_metadata(filename)
    if file_date:
        df['Date'] = file_date
        df['Day'] = file_date.strftime('%A')
        df['Country'] = country
        all_data_list.append(df)
    return country, week_num

def highlight_weekends(row):
    if row.name in ['Saturday', 'Sunday']:
        return ['background-color: rgba(128, 128, 128, 0.2)'] * len(row)
    elif row.name == 'Weekly Total':
        return ['font-weight: bold; background-color: rgba(0, 150, 255, 0.1)'] * len(row)
    return [''] * len(row)

# --- MAIN UI ---
st.title(":material/monitoring: PIM Weekly Export Analyzer")
st.markdown("Upload your `ProductSets` files (**CSV, Excel, or ZIP archives**) to generate a professional performance report.")

uploaded_files = st.file_uploader("Select files or ZIP archives to process", type=["csv", "xlsx", "xls", "zip"], accept_multiple_files=True)

# --- DATA PROCESSING ---
master_df = pd.DataFrame()
primary_country, primary_week = "Unknown", "N/A"
total_files_merged = 0

if uploaded_files:
    all_data = []
    
    with st.spinner(":material/hourglass_empty: Extracting and merging files..."):
        for file in uploaded_files:
            if file.name.endswith('.zip'):
                with zipfile.ZipFile(file, 'r') as z:
                    for zip_filename in z.namelist():
                        if zip_filename.endswith(('.csv', '.xlsx', '.xls')) and not zip_filename.startswith('__MACOSX'):
                            with z.open(zip_filename) as f:
                                if zip_filename.endswith('.csv'):
                                    df = pd.read_csv(f, low_memory=False)
                                else:
                                    df = pd.read_excel(f)
                                c, w = process_dataframe(df, zip_filename.split('/')[-1], all_data)
                                if primary_country == "Unknown":
                                    primary_country, primary_week = c, w
                                total_files_merged += 1
            else:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file, low_memory=False)
                else:
                    df = pd.read_excel(file)
                c, w = process_dataframe(df, file.name, all_data)
                if primary_country == "Unknown":
                    primary_country, primary_week = c, w
                total_files_merged += 1
        
        if all_data:
            master_df = pd.concat(all_data, ignore_index=True)
            st.success(f"Successfully merged **{total_files_merged} file(s)** for **{primary_country}** (Week {primary_week})", icon=":material/library_add_check:")
        else:
            st.error("Could not find valid data files.", icon=":material/error:")

# --- DASHBOARD RENDERING ---
if not master_df.empty:
    status_col = get_col(master_df, ['Status', 'STATUS', 'status', 'LISTING_STATUS']) 
    seller_col = get_col(master_df, ['SellerName', 'SELLER_NAME', 'seller_name', 'Seller'])
    flag_col = get_col(master_df, ['FLAG', 'Flag', 'flag', 'Reason'])
    cat_col = get_col(master_df, ['CATEGORY', 'Category', 'category'])
    
    if status_col:
        
        # --- PRE-CALCULATE CORE METRICS ---
        daily_summary = master_df.groupby(['Day', status_col]).size().unstack(fill_value=0)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_summary = daily_summary.reindex(days_order).fillna(0).astype(int)
        daily_summary['Daily Total'] = daily_summary.sum(axis=1)
        
        weekly_approved = daily_summary.get('Approved', pd.Series(0)).sum()
        weekly_rejected = daily_summary.get('Rejected', pd.Series(0)).sum()
        weekly_total = daily_summary['Daily Total'].sum()
        rejection_rate = (weekly_rejected / weekly_total * 100) if weekly_total > 0 else 0
        
        daily_summary.loc['Weekly Total'] = daily_summary.sum()
        rejected_df = master_df[master_df[status_col] == 'Rejected']

        # Pre-calculate Seller Stats
        if seller_col:
            seller_stats = master_df.groupby(seller_col)[status_col].value_counts().unstack(fill_value=0)
            seller_stats['Total Submitted'] = seller_stats.sum(axis=1)
            seller_stats['Rejected'] = seller_stats.get('Rejected', 0)
            seller_stats['Rejection Rate (%)'] = (seller_stats['Rejected'] / seller_stats['Total Submitted'] * 100).round(1)
        else:
            seller_stats = pd.DataFrame()

        # Create Tabs
        tab_exec, tab_deepdive, tab_data = st.tabs([
            ":material/summarize: Executive Summary", 
            ":material/troubleshoot: Rejection Deep-Dive", 
            ":material/table_chart: Data Explorer"
        ])

        # === TAB 1: EXECUTIVE SUMMARY & INSIGHTS ===
        with tab_exec:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Processed", f"{int(weekly_total):,}")
            c2.metric("Total Approved", f"{int(weekly_approved):,}")
            c3.metric("Total Rejected", f"{int(weekly_rejected):,}", f"{rejection_rate:.1f}% Rate", delta_color="inverse")

            # --- AUTOMATED INSIGHTS GENERATION ---
            st.markdown("### :material/lightbulb: Automated Insights")
            insights_container = st.container(border=True)
            
            with insights_container:
                # 1. Health Insight (Updated)
                if rejection_rate > 20:
                    st.error(f"**Action Required:** The overall rejection rate is high at **{rejection_rate:.1f}%**.", icon=":material/report:")
                elif rejection_rate > 0:
                    st.success(f"**Healthy Quality:** The approval rate is strong at **{100-rejection_rate:.1f}%**.", icon=":material/health_and_safety:")
                
                # 2. Top Reason Insight
                if flag_col and not rejected_df.empty:
                    top_reason = rejected_df[flag_col].value_counts().index[0]
                    top_reason_count = rejected_df[flag_col].value_counts().iloc[0]
                    reason_pct = (top_reason_count / weekly_rejected) * 100
                    st.warning(f"**Primary Bottleneck:** **'{top_reason}'** is the leading cause of rejections, accounting for **{reason_pct:.1f}%** of all rejected products.", icon=":material/warning:")
                
                # 3. Seller Focus Insight (Updated)
                if not seller_stats.empty and not rejected_df.empty:
                    worst_seller = seller_stats.sort_values(by='Rejected', ascending=False).index[0]
                    worst_seller_rejections = seller_stats.loc[worst_seller, 'Rejected']
                    if worst_seller_rejections > 0:
                        st.info(f"**Seller Focus:** **{worst_seller}** generated the highest number of rejections ({int(worst_seller_rejections)} items).", icon=":material/person_search:")
                
                # 4. Activity Insight (Updated)
                if weekly_total > 0:
                    busiest_day = daily_summary['Daily Total'].drop('Weekly Total').idxmax()
                    st.info(f"**Peak Activity:** **{busiest_day}** saw the highest volume of SKUs.", icon=":material/trending_up:")

            st.divider()

            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                st.markdown("#### :material/show_chart: Daily Processing Trend")
                trend_data = master_df.groupby(['Date', status_col]).size().reset_index(name='Count')
                fig_trend = px.line(trend_data, x='Date', y='Count', color=status_col, markers=True, 
                                    color_discrete_map={"Approved": "#2e7d32", "Rejected": "#d32f2f"})
                fig_trend.update_layout(xaxis_title="", yaxis_title="Products Processed", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_trend, use_container_width=True)

            with col_table:
                st.markdown("#### :material/calendar_today: Daily Breakdown")
                styled_daily_summary = daily_summary.style.apply(highlight_weekends, axis=1)
                st.dataframe(styled_daily_summary, use_container_width=True)

        # === TAB 2: REJECTION DEEP-DIVE ===
        with tab_deepdive:
            if rejected_df.empty:
                st.info("No rejected products found in this dataset.", icon=":material/check_circle:")
            else:
                c_pie, c_bar = st.columns(2)
                
                with c_pie:
                    st.markdown("#### :material/donut_large: Rejection Reasons Breakdown")
                    if flag_col:
                        reason_counts = rejected_df[flag_col].value_counts().reset_index()
                        reason_counts.columns = ['Reason', 'Count']
                        fig_pie = px.pie(reason_counts, values='Count', names='Reason', hole=0.4)
                        fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)

                with c_bar:
                    st.markdown("#### :material/storefront: Top 5 Rejected Sellers")
                    if not seller_stats.empty:
                        top_5_sellers = seller_stats.sort_values(by='Rejected', ascending=False).head(5).reset_index()
                        fig_seller = px.bar(top_5_sellers, x='Rejected', y=seller_col, orientation='h', 
                                            text='Rejection Rate (%)', color_discrete_sequence=['#ef5350'])
                        fig_seller.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
                        fig_seller.update_traces(texttemplate='%{text}% Rate', textposition='outside')
                        st.plotly_chart(fig_seller, use_container_width=True)

                st.divider()
                c_cat, c_rea = st.columns(2)
                with c_cat:
                    st.markdown("#### :material/category: Top 5 Rejected Categories")
                    if cat_col:
                        top_categories = rejected_df[cat_col].value_counts().head(5).reset_index()
                        top_categories.columns = ['Category', 'Rejected Count']
                        st.dataframe(top_categories, use_container_width=True, hide_index=True)

                with c_rea:
                    st.markdown("#### :material/report: Top 5 Rejection Reasons (Data)")
                    if flag_col:
                        top_reasons = rejected_df[flag_col].value_counts().head(5).reset_index()
                        top_reasons.columns = ['Reason', 'Rejected Count']
                        st.dataframe(top_reasons, use_container_width=True, hide_index=True)

        # === TAB 3: DATA EXPLORER ===
        with tab_data:
            st.markdown("#### :material/filter_alt: Filter & Search Data")
            filter_c1, filter_c2 = st.columns(2)
            
            with filter_c1:
                selected_status = st.multiselect("Filter by Status", master_df[status_col].dropna().unique())
            with filter_c2:
                if seller_col:
                    seller_search = st.text_input("Search for a specific Seller")
            
            filtered_df = master_df.copy()
            if selected_status:
                filtered_df = filtered_df[filtered_df[status_col].isin(selected_status)]
            if seller_col and seller_search:
                filtered_df = filtered_df[filtered_df[seller_col].astype(str).str.contains(seller_search, case=False, na=False)]
                
            st.dataframe(filtered_df, use_container_width=True)

        # === DOWNLOAD REPORT ===
        st.divider()
        st.markdown("#### :material/download: Export Formal Report")
        
        report_metadata = {
            "Report Generated On": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Country / Region": primary_country,
            "Reporting Week": f"Week {primary_week}",
            "Total Files/Parts Merged": total_files_merged,
            "Total Products Processed": weekly_total,
            "Total Approved": weekly_approved,
            "Total Rejected": weekly_rejected,
            "Overall Rejection Rate": f"{rejection_rate:.1f}%"
        }
        
        safe_seller_stats = seller_stats if not seller_stats.empty else pd.DataFrame(["N/A"])
        safe_top_reasons = reason_counts if flag_col else pd.DataFrame(["N/A"])
        safe_top_cats = top_categories if cat_col else pd.DataFrame(["N/A"])
        
        report_data = generate_excel_report(daily_summary, safe_seller_stats, safe_top_reasons, safe_top_cats, report_metadata)
        download_filename = f"{primary_country}_Week{primary_week}_ExecutiveReport.xlsx"
        
        st.download_button(
            label="Download Complete Executive Report (Excel)",
            data=report_data,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            icon=":material/download:"
        )
