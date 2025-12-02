# streamlit_cv_scorer.py
"""
Marketplace CV Scorer — Pie (no red) + 3D-like pie effect + updated recommendation rules
"""

import re
from io import BytesIO
from typing import List, Tuple
from datetime import datetime

import streamlit as st
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------
# Config & colors
# ----------------------------
st.set_page_config(page_title="Akoi CV Screening", layout="wide")

# Car-like gauge colors (kept)
COLOR_LOW = "#E53935"      # red (used for gauge low zone only)
COLOR_MED = "#FFD600"      # yellow
COLOR_HIGH = "#00C853"     # green

# New pie palette (no red): blues / greens / orange / purple
PIE_PALETTE = ["#1565C0", "#00C853", "#FF9800", "#7B1FA2", "#0288D1", "#4CAF50", "#FFA726"]

def hex_to_rgb(hex_color: str) -> Tuple[int,int,int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return r,g,b

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# ----------------------------
# Keyword dictionaries
# ----------------------------
KPIS = {
    "Marketplace Revenue Growth": ["gmv","gmv growth","revenue growth","increased revenue","scaled revenue","sales growth"],
    "Marketplace PPC (ACoS/ROAS)": ["acos","a cos","roas","sponsored products","sponsored brands","improved acos","reduced acos"],
    "Listing Optimization & SEO (Marketplace)": ["listing","title","backend keywords","search terms","a+ content","enhanced brand content","bullets","images"],
    "Account Health & Operations": ["odr","order defect rate","late shipment","account health","policy violation","suspension"],
    "Inventory & Forecasting": ["replenish","replenishment","forecast","stockout","safety stock","inventory planning"],
    "Pricing & Promotions": ["deal","price off","coupon","flash sale","prime day","big billion","bbd","discount"],
    "Reviews & Rating Management": ["review","rating","feedback","product reviews","improved rating"],
    "Marketplace Tools & Analytics": ["helium 10","helium10","cerebro","magnet","datadive","keepa","jungle scout","sellerapp","merchantwords"],
    "Quickcommerce Experience": ["quick commerce","quickcommerce","blinkit","dunzo","zepto","instamart","swiggy genie"],
}

PLATFORMS = {
    "Amazon": ["amazon","seller central","fba","buy box","a+ content","sponsored products","vendor central"],
    "Flipkart": ["flipkart","flipkart seller","flipkart ads","flipkart catalog"],
    "Meesho": ["meesho","meesho seller"],
    "Myntra": ["myntra","myntra seller"],
    "JioMart": ["jiomart","jiomart seller"],
    "BigBasket": ["bigbasket","big basket"],
    "Nykaa": ["nykaa","nykaa seller"],
}

BRAND_BUILDING = ["brand strategy","brand building","positioning","rebranding","brand awareness","brand campaign"]
PERFORMANCE = ["mom","yoy","qoq","year over year","month over month"]
SCALING = ["scaled","doubled","tripled","3x","2x","grew by","expanded to"]

CORE = [
    "Marketplace Revenue Growth",
    "Marketplace PPC (ACoS/ROAS)",
    "Listing Optimization & SEO (Marketplace)",
    "Account Health & Operations",
    "Inventory & Forecasting",
    "Pricing & Promotions",
    "Reviews & Rating Management",
    "Marketplace Tools & Analytics"
]

# ----------------------------
# PDF text extraction
# ----------------------------
def extract_text_from_pdf(data: bytes) -> str:
    try:
        r = PyPDF2.PdfReader(BytesIO(data))
    except Exception:
        return ""
    text = ""
    for p in r.pages:
        try:
            text += p.extract_text() or ""
        except:
            pass
    return text.lower()

# ----------------------------
# Experience extraction
# ----------------------------
def extract_years_experience(text: str) -> float:
    text = text.lower()
    current = datetime.now().year

    explicit = []
    for m in re.findall(r"(\d{1,2}(?:\.\d)?)\s*(?:years|year|yrs)", text):
        try:
            explicit.append(float(m))
        except:
            pass

    since = []
    for m in re.findall(r"since\s+((?:19|20)\d{2})", text):
        try:
            y = int(m); since.append(max(0, current - y))
        except:
            pass

    ranges = []
    for m in re.findall(r"((?:19|20)\d{2})\s*[-–to]{1,3}\s*((?:19|20)\d{2})", text):
        try:
            s,e = int(m[0]), int(m[1])
            if e >= s: ranges.append(e - s + 1)
        except:
            pass

    candidates = []
    if explicit: candidates.append(max(explicit))
    if since: candidates.append(max(since))
    if ranges: candidates.append(sum(ranges))

    return round(max(candidates) if candidates else 0.0, 1)

# ----------------------------
# Keyword scoring utilities
# ----------------------------
def detect_numeric(text: str) -> bool:
    return bool(re.search(r"\d+%|₹|increased by|\d+k|\d+m|lakh|crore|cr", text))

def score_keywords(text: str, keywords: List[str], numeric_boost: bool = True) -> float:
    found = 0
    for k in keywords:
        if re.search(rf"\b{re.escape(k)}\b", text):
            found += 1
    coverage = found / len(keywords) if keywords else 0
    base = coverage * 8
    if numeric_boost and detect_numeric(text):
        base += 2
    return round(min(10, base), 1)

def score_core(text: str) -> float:
    vals = [score_keywords(text, KPIS[k]) for k in CORE]
    return round(sum(vals)/len(vals), 1) if vals else 0.0

def score_platforms(text: str) -> dict:
    out = {}
    for p,kws in PLATFORMS.items():
        out[p] = score_keywords(text, kws)
    return out

# ----------------------------
# Sidebar weights
# ----------------------------
st.sidebar.header("Weights (adjustable)")
w_exp = st.sidebar.slider("Experience %", 10,30,20)
w_core = st.sidebar.slider("Core KPIs %", 20,40,30)
w_brand = st.sidebar.slider("Brand %", 10,30,20)
w_perf = st.sidebar.slider("Performance over time %", 5,25,15)
w_scale = st.sidebar.slider("Scaling success %", 5,15,10)

total = w_exp + w_core + w_brand + w_perf + w_scale
w_exp /= total; w_core /= total; w_brand /= total; w_perf /= total; w_scale /= total

st.sidebar.markdown("---")
st.sidebar.caption("Pie colors avoid red; gauge keeps car-grade color zones.")

# ----------------------------
# File uploader & extraction
# ----------------------------
st.title("Marketplace CV Scorer — Platform Pie & Car Gauge")

uploaded = st.file_uploader("Upload CV PDF (text PDF preferred)", type=["pdf"])
if not uploaded:
    st.info("Upload a CV PDF to begin screening.")
    st.stop()

text = extract_text_from_pdf(uploaded.read())
if not text.strip():
    st.error("Could not extract text. PDF may be scanned or unreadable.")
    st.stop()

st.subheader("CV preview (first 1100 chars)")
st.text_area("Preview", value=text[:1100], height=200)

# ----------------------------
# Compute scores
# ----------------------------
years = extract_years_experience(text)
exp_score = round(min(10, (years ** 0.85) * 1.25), 1)

core_score = score_core(text)
brand_score = score_keywords(text, BRAND_BUILDING)
perf_score = score_keywords(text, PERFORMANCE)
scale_score = score_keywords(text, SCALING)

combined = (
    exp_score * w_exp +
    core_score * w_core +
    brand_score * w_brand +
    perf_score * w_perf +
    scale_score * w_scale
)

best_comp = max(exp_score, core_score, brand_score, perf_score, scale_score)
final_score = round(min(10, combined + 0.2 * (best_comp - combined)), 2)

qc_val = score_keywords(text, KPIS.get("Quickcommerce Experience", []))
qc_bonus = 1.0 if qc_val > 0 else 0.0
final_with_qc = round(min(10, final_score + qc_bonus), 2)

platform_scores = score_platforms(text)

# ----------------------------
# Top gauge (car-like)
# ----------------------------
low_rgba = hex_to_rgba(COLOR_LOW, 0.35)
med_rgba = hex_to_rgba(COLOR_MED, 0.28)
high_rgba = hex_to_rgba(COLOR_HIGH, 0.28)
bar_color = COLOR_HIGH if final_with_qc > 7 else COLOR_MED if final_with_qc > 4 else COLOR_LOW

g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=final_with_qc,
    number={'suffix': " /10", 'font': {'size': 28}},
    gauge={
        'axis': {'range': [0, 10]},
        'bar': {'color': bar_color},
        'steps': [
            {'range': [0, 3.3], 'color': low_rgba},
            {'range': [3.3, 6.6], 'color': med_rgba},
            {'range': [6.6, 10], 'color': high_rgba}
        ],
        'threshold': {'line': {'color': "#000000", 'width': 3}, 'value': final_with_qc}
    },
    title={'text': "Final Marketplace Score (lenient combined)", 'font': {'size': 14}}
))
g.update_layout(height=260, margin=dict(l=40, r=40, t=30, b=10))
st.subheader("Final Candidate Gauge")
st.plotly_chart(g, use_container_width=True)

# ----------------------------
# Platform Pie (3D-like)
# ----------------------------
st.subheader("Platform Experience Distribution (Pie Chart — 3D-like)")

platform_names = list(platform_scores.keys())
platform_values = [platform_scores[p] for p in platform_names]

# if all zeros, show small uniform values to avoid empty pie
if sum(platform_values) == 0:
    platform_values = [1] * len(platform_values)

# choose color sequence from PIE_PALETTE (avoid red)
color_sequence = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(platform_names))]

# small pulls to create a 3D popped effect
pulls = [0.03] * len(platform_names)

fig_pie = go.Figure(go.Pie(
    labels=platform_names,
    values=platform_values,
    hole=0.25,
    marker=dict(colors=color_sequence, line=dict(color="#111111", width=1)),
    pull=pulls,
    sort=False
))
fig_pie.update_traces(textinfo='label+percent', textposition='inside')
fig_pie.update_layout(height=460, margin=dict(t=10, b=10))
st.plotly_chart(fig_pie, use_container_width=True)

# ----------------------------
# Component table (no matches)
# ----------------------------
st.subheader("Component Scores")
comp_df = pd.DataFrame([
    ["Experience (yrs)", exp_score],
    ["Core Marketplace (avg)", core_score],
    ["Brand Building", brand_score],
    ["Performance Over Time", perf_score],
    ["Scaling Success", scale_score],
    ["Quickcommerce Bonus (if any)", qc_bonus]
], columns=["Component", "Score (0-10)"])
st.table(comp_df.style.format({"Score (0-10)": "{:.1f}"}))

# ----------------------------
# KPI bar chart (no matches column)
# ----------------------------
rows = []
for k in KPIS:
    rows.append([k, score_keywords(text, KPIS[k])])
kpi_df = pd.DataFrame(rows, columns=["KPI", "Score"]).sort_values("Score", ascending=False)

st.subheader("KPI Scores")
fig_bar = px.bar(
    kpi_df, x="Score", y="KPI", orientation='h',
    color="Score", range_x=[0,10],
    color_continuous_scale=[COLOR_LOW, COLOR_MED, COLOR_HIGH]
)
fig_bar.update_layout(height=520, margin=dict(l=260, r=20, t=30, b=30))
st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------------
# Recommendation (new rules)
# ----------------------------
if final_with_qc < 3.5:
    rec = "Not suitable"
elif 3.5 <= final_with_qc < 4.0:
    rec = "Can Proceed"
elif 4.0 <= final_with_qc < 7.0:
    rec = "Good Candidate"
else:
    rec = "Perfect Candidate"

st.markdown(f"### Recommendation: **{rec}**")

# CSV download
out_df = kpi_df.copy()
out_df["DetectedExperienceYears"] = years
out_df["FinalScore"] = final_with_qc
csv = out_df.to_csv(index=False)
st.download_button("Download scores CSV", csv, file_name="cv_scores.csv", mime="text/csv")

st.caption("Screening is keyword + numeric detection based on CV content. Validate claims in interview and request evidence/reports.")
