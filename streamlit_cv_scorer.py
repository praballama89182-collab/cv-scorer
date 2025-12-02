# streamlit_cv_scorer.py
# Polished UI + Plotly visuals for the CV KPI scorer
import streamlit as st
import re
from io import BytesIO
from typing import Dict, List
import PyPDF2
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="CV KPI Scorer — Polished", layout="wide", initial_sidebar_state="expanded")

# --- Small CSS tweaks ---
st.markdown(
    """
    <style>
      .kpi-card { background: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
      .matches-badge { display:inline-block; background:#eef2ff; color:#0b3d91; padding:6px 10px; border-radius:999px; margin:3px; font-size:12px; }
      .small { font-size:12px; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CV KPI Scorer — Marketing + Marketplace (Polished)")
st.write("Upload a CV PDF, get KPI scores (0–10). Use the sidebar to tune scoring behavior and weights.")

# ----------------------------
# KPI dictionary (same as before)
# ----------------------------
KPIS = {
    "Campaign Strategy & Execution": [
        "campaign", "go to market", "gtm", "campaign strategy",
        "launched campaign", "funnel", "activation"
    ],
    "Content & Creative Ability": [
        "content", "copy", "creative", "reels", "social content",
        "a/b test", "storyboard", "video"
    ],
    "Performance Marketing": [
        "meta ads", "facebook ads", "google ads", "adwords",
        "roas", "acos", "cpa", "ctr", "cpc", "paid",
        "campaign optimisation", "paid social"
    ],
    "Social Media Growth & Engagement": [
        "instagram", "facebook", "followers", "engagement",
        "community", "reach", "impressions", "engagement rate"
    ],
    "Branding & Positioning": [
        "brand", "positioning", "brand strategy", "tone of voice",
        "identity", "rebrand", "brand building"
    ],
    "Analytics & Insights": [
        "google analytics", "ga4", "data studio", "looker",
        "tableau", "kpi", "dashboard", "cohort", "segmentation"
    ],
    "SEO / Organic Awareness": [
        "seo", "organic traffic", "search engine", "backlink",
        "on-page", "technical seo", "keyword research"
    ],
    "Copywriting Strength": [
        "copywriting", "ad copy", "product description",
        "content writing", "storytelling"
    ],
    "Project Ownership & Speed": [
        "led", "owned", "end to end", "delivered",
        "launched", "stakeholder", "cross functional"
    ],
    "Tools Knowledge (Core)": [
        "google ads", "meta ads manager", "facebook ads manager",
        "ga4", "mailchimp", "klaviyo", "shopify", "canva", "excel"
    ],
    # Bonus
    "Marketplace Knowledge (Amazon/Flipkart)": [
        "amazon", "flipkart", "marketplace", "seller central",
        "buy box", "fba", "a+ content", "catalog"
    ],
    "Helium10 / DataDive Knowledge": [
        "helium 10", "helium10", "cerebro", "magnet",
        "data dive", "datadive", "keepa", "jungle scout",
        "sellerapp", "merchantwords"
    ],
    "D2C / Website Knowledge (Shopify)": [
        "shopify", "woocommerce", "landing page",
        "conversion rate", "cvr", "hotjar", "webengage"
    ],
    "Influencer / Affiliate Marketing": [
        "influencer", "affiliate", "creator",
        "ugc", "collaboration", "partnership"
    ]
}

# ----------------------------
# Helper functions (same logic)
# ----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_pages = []
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        for p in reader.pages:
            try:
                text_pages.append(p.extract_text() or "")
            except:
                text_pages.append("")
    except:
        return ""
    return "\n".join(text_pages).lower()

def find_numeric_signals(text: str) -> bool:
    if re.search(r"\d+\s?%|\+\d+\s?%|increased by \d+|revenue.*\d+|₹|rs\.|inr|\d+k|\d+,?\d{2,}", text):
        return True
    return False

def score_kpi_for_keywords(text: str, keywords: List[str], numeric_boost: bool, numeric_bonus: float) -> Dict:
    found = set()
    for kw in keywords:
        pattern = re.escape(kw.lower())
        if re.search(rf"\b{pattern}\b", text):
            found.add(kw)
    coverage = len(found) / max(1, len(keywords))
    numeric = find_numeric_signals(text)
    base = coverage * 8.0
    bonus = numeric_bonus if (numeric_boost and numeric) else 0.0
    score = min(10.0, base + bonus)
    return {"score": round(score,1), "matches": sorted(found), "coverage": round(coverage,3), "numeric": numeric}

def score_all_kpis(text: str, numeric_boost: bool, numeric_bonus: float):
    results = {}
    for kpi, kws in KPIS.items():
        results[kpi] = score_kpi_for_keywords(text, kws, numeric_boost, numeric_bonus)
    return results

# ----------------------------
# Sidebar - controls
# ----------------------------
st.sidebar.header("Scoring settings")
numeric_boost = st.sidebar.checkbox("Enable numeric boost (+2 when CV contains %/revenue numbers)", value=True)
numeric_bonus = st.sidebar.slider("Numeric bonus value", min_value=0.0, max_value=3.0, value=2.0, step=0.1)
st.sidebar.markdown("---")
st.sidebar.write("Weights (used for final weighted score)")
core_weight = st.sidebar.slider("Core KPIs weight (%)", 50, 90, 80, 5)
bonus_weight = 100 - core_weight
st.sidebar.write(f"Bonus KPIs weight: {bonus_weight}%")
st.sidebar.markdown("---")
st.sidebar.write("UI options")
show_matches = st.sidebar.checkbox("Show detailed matches panels", value=False)
st.sidebar.caption("Tip: adjust numeric boost and weights for calibration.")

# ----------------------------
# File upload
# ----------------------------
uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"])
if not uploaded:
    st.info("Upload a text-based PDF CV to begin scoring. OCR not included.")
    st.stop()

raw_bytes = uploaded.read()
st.success("PDF uploaded.")

with st.spinner("Extracting text..."):
    text = extract_text_from_pdf(raw_bytes)

if not text.strip():
    st.error("Could not extract text. The PDF may be scanned or require OCR.")
    st.stop()

# Short preview of text
st.subheader("Extracted CV Text (preview)")
st.text_area("Text (lowercased)", value=text[:1600], height=200)

# Score
results = score_all_kpis(text, numeric_boost, numeric_bonus)

# Prepare DataFrame for display and plotting
kpi_names = list(results.keys())
scores = [results[k]["score"] for k in kpi_names]
coverages = [results[k]["coverage"] for k in kpi_names]
numeric_flags = [results[k]["numeric"] for k in kpi_names]
matches = [results[k]["matches"] for k in kpi_names]

df = pd.DataFrame({
    "KPI": kpi_names,
    "Score": scores,
    "Coverage": coverages,
    "Numeric": numeric_flags,
    "Matches": ["; ".join(m) for m in matches]
})

# Split core vs bonus
core_count = 10
core_df = df.iloc[:core_count].copy()
bonus_df = df.iloc[core_count:].copy()

core_avg = core_df["Score"].mean()
bonus_avg = bonus_df["Score"].mean()
weighted_final = round((core_avg * (core_weight / 100.0) + bonus_avg * (bonus_weight / 100.0)), 2)

# Top metrics row
col1, col2, col3 = st.columns([1.5,1,1])
with col1:
    st.metric("Core KPIs Average (0-10)", f"{core_avg:.1f}")
with col2:
    st.metric("Bonus KPIs Average (0-10)", f"{bonus_avg:.1f}")
with col3:
    st.metric("Weighted Final Score (0-10)", f"{weighted_final:.2f}")

st.markdown("----")

# Two column layout: left (cards + details), right (plot + table)
left, right = st.columns([1.2,1])

# Left: KPI cards (compact)
with left:
    st.subheader("KPI Quick View")
    for idx, row in core_df.iterrows():
        kpi = row["KPI"]
        score = row["Score"]
        cov = row["Coverage"]
        numeric_flag = row["Numeric"]
        # Card style
        st.markdown(f"<div class='kpi-card'><b>{kpi}</b></div>", unsafe_allow_html=True)
        # Score + progress
        st.write(f"Score: **{score} / 10** — Coverage: {cov*100:.0f}% {'• Has numbers' if numeric_flag else ''}")
        st.progress(int(score*10))
    if not bonus_df.empty:
        st.subheader("Bonus KPIs")
        for idx, row in bonus_df.iterrows():
            kpi = row["KPI"]
            score = row["Score"]
            cov = row["Coverage"]
            numeric_flag = row["Numeric"]
            st.markdown(f"<div class='kpi-card'><b>{kpi}</b></div>", unsafe_allow_html=True)
            st.write(f"Score: **{score} / 10** — Coverage: {cov*100:.0f}% {'• Has numbers' if numeric_flag else ''}")
            st.progress(int(score*10))

    if show_matches:
        st.subheader("Matches (detailed)")
        for i,k in enumerate(kpi_names):
            st.markdown(f"**{k}** — Matches:")
            if matches[i]:
                # show small badges
                badges = " ".join([f"<span class='matches-badge'>{m}</span>" for m in matches[i][:30]])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.write("_No keyword matches found_")
            st.write("")

# Right: Plotly chart and table
with right:
    st.subheader("Interactive KPI Chart")
    # Create a Plotly horizontal bar chart with hover info
    fig = go.Figure()
    colors = px.colors.sequential.Blues_r
    # Map score to color scale
    fig.add_trace(go.Bar(
        x=df["Score"],
        y=df["KPI"],
        orientation='h',
        marker=dict(
            color=df["Score"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Score")
        ),
        hovertemplate="<b>%{y}</b><br>Score: %{x}<br>Coverage: %{customdata[0]:.0%}<br>Numeric: %{customdata[1]}<extra></extra>",
        customdata=list(zip(df["Coverage"], df["Numeric"])),
    ))
    fig.update_layout(height=650, margin=dict(l=200, r=20, t=30, b=30))
    fig.update_xaxes(range=[0,10], title_text="Score (0-10)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed table")
    st.dataframe(df.style.format({"Score":"{:.1f}", "Coverage":"{:.1%}"}), use_container_width=True)

# CSV download
csv = df.to_csv(index=False)
st.download_button("Download KPI scores CSV", csv, file_name="cv_kpi_scores.csv", mime="text/csv")

st.markdown("---")
st.caption("Scoring is heuristic (keyword + numeric detection). Tune keywords and numeric boost for better calibration.")
