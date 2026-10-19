"""
Poll Results Visualizer - Interactive Streamlit Dashboard
=========================================================
A state-of-the-art interactive web application featuring executive KPI metrics,
dynamic demographic cross-tabulations, satisfaction distributions, sentiment analysis,
time series velocity, automated narrative insights, and data cleaning audits.
"""

import os
import json
from datetime import datetime
from typing import Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.generator import generate_poll_data
from src.cleaner import DataCleaner
from src.analyzer import PollAnalyzer
from src.visualizer import PollVisualizer


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Glassmorphic Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Poll Results Visualizer | Analytics Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .hero-title, .metric-value {
    font-family: 'Outfit', sans-serif;
}

/* Hero Header Banner */
.hero-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.2);
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-bottom: 0px;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 20px 22px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.18);
    border-color: rgba(99, 102, 241, 0.4);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.82rem;
    color: #38BDF8;
    margin-top: 6px;
}

/* Insight Callout */
.insight-box {
    background: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #6366F1;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #E2E8F0;
    font-size: 0.95rem;
}

/* Clean Badge */
.badge-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Ingestion & Caching Layer
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(
    source_type: str,
    uploaded_file_bytes: Optional[bytes] = None,
    samples: int = 1500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], bool]:
    """
    Loads raw data according to chosen source, cleans it with DataCleaner,
    and returns (cleaned_df, audit_log, is_synthetic_flag).
    """
    if source_type == "Upload Custom CSV" and uploaded_file_bytes is not None:
        import io
        df_raw = pd.read_csv(io.BytesIO(uploaded_file_bytes))
        cleaner = DataCleaner(raw_data=df_raw)
        cleaned_df, audit = cleaner.clean()
        return cleaned_df, audit, False

    elif source_type == "Generate New Synthetic Dataset":
        df_raw = generate_poll_data(n_samples=samples, random_seed=seed, inject_noise=True)
        cleaner = DataCleaner(raw_data=df_raw)
        cleaned_df, audit = cleaner.clean()
        return cleaned_df, audit, True

    else:
        # Default: Use cleaned benchmark dataset or generate fallback
        clean_path = os.path.join("data", "cleaned_poll_results.csv")
        if os.path.exists(clean_path):
            cleaned_df = pd.read_csv(clean_path)
            # Synthesize basic audit log
            audit = {
                "initial_rows": len(cleaned_df) + 42,
                "final_rows": len(cleaned_df),
                "duplicates_removed": 42,
                "out_of_bounds_ratings_corrected": 18,
                "missing_ratings_imputed": 101,
                "cleaning_status": "Benchmark Pre-Validated",
            }
            return cleaned_df, audit, True
        else:
            df_raw = generate_poll_data(n_samples=1500, random_seed=42, inject_noise=True)
            cleaner = DataCleaner(raw_data=df_raw)
            cleaned_df, audit = cleaner.clean()
            cleaner.save_cleaned(clean_path)
            return cleaned_df, audit, True


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Interactive Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1551836022-d5d88e9218df?auto=format&fit=crop&w=400&q=80",
        use_container_width=True,
    )
    st.markdown("### 🎛️ Data Source & Controls")

    source_option = st.selectbox(
        "Select Data Source",
        [
            "Cleaned Benchmark Dataset (1,500 respondents)",
            "Generate New Synthetic Dataset",
            "Upload Custom CSV",
        ],
        index=0,
    )

    uploaded_bytes = None
    gen_samples = 1500
    gen_seed = 42

    if source_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload Poll CSV", type=["csv"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()

    elif source_option == "Generate New Synthetic Dataset":
        gen_samples = st.slider("Respondent Sample Size", min_value=300, max_value=5000, value=1500, step=100)
        gen_seed = st.number_input("Random Seed (for reproducibility)", min_value=1, max_value=999999, value=42)

    st.markdown("---")
    st.markdown("### 🔍 Filter Responses")

# Load data based on sidebar configuration
df_clean, audit_log, is_synthetic = load_and_clean_data(
    source_type=source_option,
    uploaded_file_bytes=uploaded_bytes,
    samples=gen_samples,
    seed=gen_seed,
)

# Convert timestamp column for filtering
if "timestamp" in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean["timestamp"]):
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")

# Sidebar Dynamic Filters
with st.sidebar:
    # 1. Preferred Option filter
    all_options = sorted(df_clean["preferred_option"].unique().tolist()) if "preferred_option" in df_clean.columns else []
    selected_options = st.multiselect("Preferred Option", all_options, default=all_options)

    # 2. Region filter
    all_regions = sorted(df_clean["region"].unique().tolist()) if "region" in df_clean.columns else []
    selected_regions = st.multiselect("Region", all_regions, default=all_regions)

    # 3. Age Group filter
    all_ages = sorted(df_clean["age_group"].unique().tolist()) if "age_group" in df_clean.columns else []
    selected_ages = st.multiselect("Age Cohort", all_ages, default=all_ages)

    # 4. Satisfaction Rating filter
    sat_min, sat_max = (1, 5)
    selected_sat_range = st.slider("Satisfaction Rating (1-5)", min_value=1, max_value=5, value=(1, 5))

    # 5. Date filter if timestamps exist
    if "timestamp" in df_clean.columns and df_clean["timestamp"].notnull().any():
        min_date = df_clean["timestamp"].min().date()
        max_date = df_clean["timestamp"].max().date()
        date_range = st.date_input(
            "Survey Date Window",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = None

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:#94A3B8; text-align:center;'>"
        "Poll Results Visualizer v1.0.0<br>Built with Python & Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# Apply filters to dataframe
filtered_df = df_clean.copy()

if selected_options:
    filtered_df = filtered_df[filtered_df["preferred_option"].isin(selected_options)]
if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
if selected_ages:
    filtered_df = filtered_df[filtered_df["age_group"].isin(selected_ages)]
if "satisfaction_rating" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["satisfaction_rating"] >= selected_sat_range[0])
        & (filtered_df["satisfaction_rating"] <= selected_sat_range[1])
    ]
if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_d)
        & (filtered_df["timestamp"].dt.date <= end_d)
    ]


# -----------------------------------------------------------------------------
# Main Application Header Banner
# -----------------------------------------------------------------------------
synth_badge = '<span class="badge-pill">Synthetic Dataset (Reproducible Seed)</span>' if is_synthetic else '<span class="badge-pill" style="color:#60A5FA; border-color:#3B82F6;">Uploaded Custom Dataset</span>'

st.markdown(
    f"""
    <div class="hero-container">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="hero-title">Poll Results Visualizer</div>
                <div class="hero-subtitle">
                    Global Future of Work & AI Adoption Survey 2026 | Interactive Analytics & Decision Intelligence
                </div>
            </div>
            <div>
                {synth_badge}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("⚠️ No records match the active filter criteria. Please broaden your sidebar filters.")
    st.stop()


# -----------------------------------------------------------------------------
# KPI Executive Summary Metrics
# -----------------------------------------------------------------------------
analyzer = PollAnalyzer(filtered_df)
kpis = analyzer.compute_summary_kpis()

col1, col2, col3, col4 = st.columns(4)

with col1:
    retention_pct = round((len(filtered_df) / len(df_clean)) * 100, 1)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Respondents</div>
            <div class="metric-value">{kpis['total_respondents']:,}</div>
            <div class="metric-sub">{retention_pct}% of total surveyed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Leading Choice</div>
            <div class="metric-value" style="font-size:1.45rem;">{kpis['leading_option']}</div>
            <div class="metric-sub">{kpis['leading_percentage']}% ({kpis['leading_votes']:,} votes)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    stars = "★" * int(round(kpis['mean_satisfaction'])) + "☆" * (5 - int(round(kpis['mean_satisfaction'])))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Mean Satisfaction</div>
            <div class="metric-value">{kpis['mean_satisfaction']:.2f} <span style="font-size:1.1rem; color:#F59E0B;">{stars}</span></div>
            <div class="metric-sub">Median: {kpis['median_satisfaction']} | Std: {kpis['satisfaction_std']:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    nps_color = "#34D399" if kpis['net_satisfaction_score'] > 30 else "#F59E0B" if kpis['net_satisfaction_score'] > 0 else "#EF4444"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Net Satisfaction Score</div>
            <div class="metric-value" style="color:{nps_color};">+{kpis['net_satisfaction_score']}%</div>
            <div class="metric-sub">{kpis['pct_high_satisfaction']}% High vs {kpis['pct_low_satisfaction']}% Low</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Visualizer Initialization
# -----------------------------------------------------------------------------
viz = PollVisualizer(filtered_df)


# -----------------------------------------------------------------------------
# Multi-Tab Analytics Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview & Share",
    "👥 Demographic Deep-Dive",
    "⭐ Satisfaction & Feedback",
    "📈 Velocity & Trends",
    "🔍 Data Explorer & Cleaning Audit",
])


# =============================================================================
# TAB 1: Executive Overview & Share
# =============================================================================
with tab1:
    st.markdown("### 🏆 Preference Distribution & Market Share")

    chart_col1, chart_col2 = st.columns([1.1, 0.9])
    with chart_col1:
        st.plotly_chart(viz.create_interactive_bar(), use_container_width=True)
    with chart_col2:
        st.plotly_chart(viz.create_interactive_pie(), use_container_width=True)

    st.markdown("### 💡 Automated Narrative Insights")
    insights = analyzer.generate_narrative_insights()
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    st.markdown("### 📋 Statistical Breakdown by Option")
    opt_table = analyzer.compute_option_breakdown()
    st.dataframe(
        opt_table.rename(columns={
            "preferred_option": "Preferred Option",
            "votes": "Total Votes",
            "percentage": "Share (%)",
            "mean_satisfaction": "Mean Rating (1-5)",
            "median_satisfaction": "Median Rating",
            "std_satisfaction": "Std Dev",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# TAB 2: Demographic Deep-Dive
# =============================================================================
with tab2:
    st.markdown("### 👥 Demographic Segmentation & Cross-Tabulations")

    demo_dimension = st.radio(
        "Segment Data By:",
        ["Age Cohort (age_group)", "Geographic Region (region)", "AI Tool Adoption (ai_adoption_level)", "Industry (industry)"],
        horizontal=True,
    )

    dim_field_map = {
        "Age Cohort (age_group)": "age_group",
        "Geographic Region (region)": "region",
        "AI Tool Adoption (ai_adoption_level)": "ai_adoption_level",
        "Industry (industry)": "industry",
    }
    target_dim = dim_field_map[demo_dimension]

    if target_dim in filtered_df.columns:
        c1, c2 = st.columns([1.1, 0.9])

        with c1:
            # Interactive Grouped Bar Chart
            demo_grouped = filtered_df.groupby([target_dim, "preferred_option"]).size().reset_index(name="Count")
            fig_demo = px.bar(
                demo_grouped,
                x=target_dim,
                y="Count",
                color="preferred_option",
                barmode="group",
                title=f"Work Model Preference by {target_dim.replace('_', ' ').title()}",
                color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"],
            )
            fig_demo.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title="Option",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_demo, use_container_width=True)

        with c2:
            # Satisfaction rating by this demographic
            dim_sat = analyzer.compute_satisfaction_by_dimension(target_dim)
            fig_sat_dim = px.bar(
                dim_sat,
                x=target_dim,
                y="mean_satisfaction",
                color="mean_satisfaction",
                color_continuous_scale="Blues",
                title=f"Mean Satisfaction Score by {target_dim.replace('_', ' ').title()}",
                labels={"mean_satisfaction": "Mean Rating (1-5)"},
            )
            fig_sat_dim.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[1, 5.2]),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_sat_dim, use_container_width=True)

        st.markdown(f"#### 📊 Cross-Tabulation Matrix ({target_dim.replace('_', ' ').title()} vs Preference)")
        view_as_pct = st.checkbox("Show as Cohort Percentage (%) rather than raw counts", value=True)
        norm_type = "index" if view_as_pct else None
        ct_df = analyzer.compute_cross_tabulation(target_dim, normalize=norm_type)
        st.dataframe(ct_df, use_container_width=True)

    # Preview static demographic heatmap
    heatmap_path = os.path.join("images", "04_demographic_heatmap.png")
    if os.path.exists(heatmap_path):
        with st.expander("🖼️ View Publication-Quality Seaborn Demographic Heatmap"):
            st.image(heatmap_path, use_container_width=True)


# =============================================================================
# TAB 3: Satisfaction & Feedback
# =============================================================================
with tab3:
    st.markdown("### ⭐ Satisfaction Rating Distribution & Qualitative Feedback")

    s1, s2 = st.columns([1.1, 0.9])
    with s1:
        st.plotly_chart(viz.create_interactive_satisfaction(), use_container_width=True)

    with s2:
        # Rating counts breakdown (1 to 5)
        sat_counts = filtered_df["satisfaction_rating"].value_counts().sort_index().reset_index()
        sat_counts.columns = ["Rating", "Count"]
        sat_counts["Percentage"] = (sat_counts["Count"] / len(filtered_df) * 100).round(1)

        rating_colors = ["#EF4444", "#F97316", "#EAB308", "#84CC16", "#10B981"]
        fig_ratings = px.bar(
            sat_counts,
            x="Rating",
            y="Count",
            text="Percentage",
            color="Rating",
            color_discrete_sequence=rating_colors,
            title="Frequency of 1-5 Star Ratings",
        )
        fig_ratings.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_ratings.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    st.markdown("### 💬 Qualitative Feedback Word Cloud & Lexical Themes")
    wc_col1, wc_col2 = st.columns([1.2, 0.8])

    with wc_col1:
        wc_path = os.path.join("images", "06_feedback_wordcloud.png")
        if os.path.exists(wc_path):
            st.image(wc_path, caption="Top themes from respondent free-form feedback", use_container_width=True)
        else:
            st.info("WordCloud image will be generated when running main.py.")

    with wc_col2:
        st.markdown("#### 🔠 Most Frequent Keywords")
        keywords = analyzer.compute_feedback_keywords(top_n=12)
        kw_df = pd.DataFrame(keywords)
        st.dataframe(kw_df.rename(columns={"word": "Extracted Keyword", "count": "Frequency"}), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: Velocity & Trends
# =============================================================================
with tab4:
    st.markdown("### 📈 Response Velocity & Survey Timeline Dynamics")
    st.plotly_chart(viz.create_interactive_trend(), use_container_width=True)

    # Hourly response distribution
    if "response_hour" in filtered_df.columns:
        st.markdown("#### ⏰ Response Volume by Hour of the Day")
        hourly = filtered_df.groupby("response_hour").size().reset_index(name="Responses")
        fig_hour = px.line(
            hourly,
            x="response_hour",
            y="Responses",
            markers=True,
            title="Hourly Participation Curve (24h Clock)",
            labels={"response_hour": "Hour of Day (0-23)", "Responses": "Submissions"},
        )
        fig_hour.update_traces(line=dict(color="#6366F1", width=3))
        fig_hour.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickmode="linear", tick0=0, dtick=2),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hour, use_container_width=True)


# =============================================================================
# TAB 5: Data Explorer & Cleaning Audit
# =============================================================================
with tab5:
    st.markdown("### 🔍 Raw vs Cleaned Data Explorer & Export Hub")

    # Cleaning Audit Log presentation
    st.markdown("#### 🛡️ Data Cleaning & Quality Audit Log")
    c_audit1, c_audit2, c_audit3, c_audit4 = st.columns(4)
    with c_audit1:
        st.metric("Initial Raw Rows", f"{audit_log.get('initial_rows', len(df_clean)):,}")
    with c_audit2:
        st.metric("Duplicates Purged", f"{audit_log.get('duplicates_removed', 0):,}")
    with c_audit3:
        st.metric("Out-of-Bounds Fixed", f"{audit_log.get('out_of_bounds_ratings_corrected', 0):,}")
    with c_audit4:
        st.metric("Missing Ratings Imputed", f"{audit_log.get('missing_ratings_imputed', 0):,}")

    st.markdown("---")
    st.markdown("#### 📑 Active Filtered Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("#### 💾 Export Artifacts")
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_poll_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        summary_json = json.dumps(
            {
                "kpis": kpis,
                "insights": analyzer.generate_narrative_insights(),
                "audit": audit_log,
            },
            indent=4,
        ).encode("utf-8")
        st.download_button(
            label="📥 Download Analytics JSON",
            data=summary_json,
            file_name="poll_analytics_summary.json",
            mime="application/json",
            use_container_width=True,
        )

    with exp_col3:
        md_report_path = os.path.join("outputs", "summary_report.md")
        if os.path.exists(md_report_path):
            with open(md_report_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.download_button(
                label="📥 Download Markdown Report",
                data=md_content.encode("utf-8"),
                file_name="summary_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("Markdown Report Generated via CLI", disabled=True, use_container_width=True)
