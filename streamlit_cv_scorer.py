# streamlit_cv_scorer.py
"""
AKOI AI CV SCREEN
- Top 3D glass badge recommendation
- Car-style gauge
- Platform pie, KPI bar chart
- 5-bullet auto-summary (strengths/weaknesses/etc.)
- PDF export of summary (strong points, weak points, overview, reason)
- No red colors used
"""

from io import BytesIO
from typing import List, Dict, Tuple
from datetime import datetime
import re

import streamlit as st
import PyPDF2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# PDF generation
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# --------- App config ----------
st.set_page_config(page_title="AKOI AI CV SCREEN", layout="wide")
APP_TITLE = "AKOI AI CV SCREEN"

# ---------- Colors (no red anywhere) ----------
COLOR_LOW = "#FF9800"    # orange (low)
COLOR_MED = "#FFD54F"    # yellow-ish (mid)
COLOR_HIGH = "#00C853"   # green (high)
COLOR_EXTRA = "#1565C0"  # blue (accent)
PIE_PALETTE = ["#1565C0", "#00C853", "#FF9800", "#7B1FA2", "#0288D1", "#4CAF50", "#FFA726"]

# ---------- Helper color funcs ----------
def hex_to_rgb(hex_color: str) -> Tuple[int,int,int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return r,g,b

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r,g,b = hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# ---------- Keywords & platforms ----------
KPIS = {
    "Marketplace Revenue Growth": ["gmv","gmv growth","revenue growth","increased revenue","scaled revenue","sales growth"],
    "Marketplace PPC (ACoS/ROAS)": ["acos","roas","roas metric","sponsored products","sponsored brands","acos"],
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
PERFORMANCE = ["month over month","month-on-month","mom","yoy","year on year","year-over-year","qoq","sustained","monthly growth"]
SCALING = ["scaled","doubled","tripled","3x","2x","grew by","expanded to","launched in"]

CORE_KPIS = [
    "Marketplace Revenue Growth",
    "Marketplace PPC (ACoS/ROAS)",
    "Listing Optimization & SEO (Marketplace)",
    "Account Health & Operations",
    "Inventory & Forecasting",
    "Pricing & Promotions",
    "Reviews & Rating Management",
    "Marketplace Tools & Analytics"
]

# ---------- Utilities ----------
def extract_text_from_pdf(data: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(BytesIO(data))
    except Exception:
        return ""
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).lower()

def detect_numeric_signals(text: str) -> bool:
    return bool(re.search(r"\d+\s?%|\+\d+\s?%|increased by \d+|₹|rs\.|inr|\d+[kKmM]|lakh|lac|crore|cr", text))

def score_keywords(text: str, keywords: List[str], numeric_boost: bool=True) -> float:
    found = 0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            found += 1
    coverage = found / len(keywords) if keywords else 0
    base = coverage * 8.0
    if numeric_boost and detect_numeric_signals(text):
        base += 2.0
    return round(min(10.0, base), 1)

def score_core_marketplace(text: str) -> float:
    vals = []
    for k in CORE_KPIS:
        vals.append(score_keywords(text, KPIS.get(k, [])))
    return round(sum(vals)/len(vals),1) if vals else 0.0

def score_platforms(text: str) -> Dict[str, float]:
    out = {}
    for p,kws in PLATFORMS.items():
        out[p] = score_keywords(text, kws)
    return out

# ---------- Experience extraction (robust) ----------
def extract_years_experience(text: str) -> float:
    text = text.lower()
    current_year = datetime.now().year
    explicit_years = []
    for m in re.findall(r"(?:over|more than|about|around|approx(?:imately)?\s+)?\s*(\d{1,2}(?:\.\d)?)\s*(?:\+)?\s*(?:years|year|yrs|yr)\b", text):
        try:
            explicit_years.append(float(m))
        except:
            pass
    since_years = []
    for m in re.findall(r"since\s+((?:19|20)\d{2})\b", text):
        try:
            y = int(m)
            since_years.append(max(0, current_year - y))
        except:
            pass
    ranges = []
    for m in re.findall(r"((?:19|20)\d{2})\s*(?:[-–—to]{1,4})\s*((?:19|20)\d{2})", text):
        try:
            s = int(m[0]); e = int(m[1])
            if e >= s:
                ranges.append(e - s + 1)
        except:
            pass
    candidates = []
    if explicit_years:
        candidates.append(max(explicit_years))
    if since_years:
        candidates.append(max(since_years))
    if ranges:
        candidates.append(sum(ranges))
    if not candidates:
        return 0.0
    return round(min(max(candidates), 60.0), 1)

# ---------- Summary generation ----------
def generate_summary(text: str, platform_scores: Dict[str,float], core_score: float) -> Tuple[List[str], str, str]:
    """
    Returns (bullets[5], overview_paragraph, risk_paragraph)
    """
    bullets = []
    # 1. Top strengths: platforms with score >=6
    strengths = [p for p,s in platform_scores.items() if s >= 6.0]
    if strengths:
        bullets.append(f"Strong platform experience: {', '.join(strengths)}.")
    else:
        # pick top 1-2 platforms
        top = sorted(platform_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top_names = [p for p,_ in top if _>0]
        if top_names:
            bullets.append(f"Platform familiarity: {', '.join(top_names)} (basic exposure).")
        else:
            bullets.append("No clear platform dominance detected from the CV.")

    # 2. Core marketplace competency
    if core_score >= 6.0:
        bullets.append(f"Strong marketplace fundamentals (core KPIs average {core_score}/10).")
    elif core_score >= 4.0:
        bullets.append(f"Moderate marketplace skills (core KPIs avg {core_score}/10) — needs targeted experience.")
    else:
        bullets.append(f"Limited marketplace KPI experience (avg {core_score}/10).")

    # 3. Brand / scaling signals
    brand_signal = bool(re.search(r"brand strategy|brand building|brand awareness|rebrand|brand launch", text))
    scale_signal = bool(re.search(r"scaled|doubled|tripled|\d+x|increased by \d+%|grew by", text))
    if brand_signal and scale_signal:
        bullets.append("Evidence of brand building and scaling success.")
    elif brand_signal:
        bullets.append("Experience shows brand-building activities.")
    elif scale_signal:
        bullets.append("Shows scaling/growth achievements (numeric growth mentions).")
    else:
        bullets.append("Little explicit evidence of brand-building or scaling success.")

    # 4. Tools & analytics
    tools = []
    for kw in ["helium 10","keepa","jungle scout","sellerapp","merchantwords","google analytics","google ads","facebook ads"]:
        if kw in text:
            tools.append(kw)
    if tools:
        bullets.append(f"Tools & analytics mentioned: {', '.join(tools)}.")
    else:
        bullets.append("No marketplace tool names explicitly mentioned.")

    # 5. Risk/gap
    risks = []
    if core_score < 4.0:
        risks.append("Weak experience in core marketplace operations.")
    if not brand_signal and not scale_signal:
        risks.append("Lacks clear brand or scaling examples.")
    if sum(platform_scores.values()) == 0:
        risks.append("No platform-specific keywords detected.")
    if risks:
        bullets.append("Risk: " + " ".join(risks))
    else:
        bullets.append("No major risk signals detected in CV.")

    # Ensure bullets length exactly 5 (trim or pad)
    bullets = bullets[:5]
    while len(bullets) < 5:
        bullets.append("Additional details not found in CV.")

    # Overview paragraph
    overview = f"The candidate shows a core marketplace score of {core_score}/10 and platform exposure primarily to " \
               f"{', '.join([p for p,_ in sorted(platform_scores.items(), key=lambda x:-x[1]) if _>0][:3]) or 'none listed'}. " \
               "Use the bullets above as a quick snapshot."

    # Risk / why-why-not paragraph placeholder (will be customized per recommendation)
    risk_para = "Based on CV content; validate during interview and request console reports/screenshots for claims."

    return bullets, overview, risk_para

# ---------- PDF generation ----------
def create_summary_pdf(candidate_name: str, final_score: float, recommendation: str, emoji: str,
                       bullets: List[str], overview: str, reason_paragraph: str) -> bytes:
    if not FPDF_AVAILABLE:
        raise RuntimeError("fpdf package not available. Install with: pip install fpdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"AKOI AI CV SCREEN - Candidate Summary", ln=True, align="C")
    pdf.ln(4)
    # Candidate name & score
    pdf.set_font("Arial", "", 12)
    if candidate_name:
        pdf.cell(0, 8, f"Candidate: {candidate_name}", ln=True)
    pdf.cell(0, 8, f"Final Score: {final_score} / 10   Recommendation: {recommendation} {emoji}", ln=True)
    pdf.ln(6)
    # Bullets
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Strong / Notable points:", ln=True)
    pdf.set_font("Arial", "", 11)
    for b in bullets:
        pdf.multi_cell(0, 7, f"• {b}")
    pdf.ln(4)
    # Weak points / reason
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Overview:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, overview)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Reason / Notes:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, reason_paragraph)
    # footer
    pdf.ln(6)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 6, "Generated by AKOI AI CV SCREEN", ln=True, align="R")
    return pdf.output(dest="S").encode("latin1")

# ---------- Sidebar controls ----------
st.sidebar.header("Scoring Settings")
numeric_boost_enabled = st.sidebar.checkbox("Enable numeric boost (+2 if numeric evidence present)", value=True)
quickcommerce_bonus = st.sidebar.checkbox("Add Quickcommerce bonus (+1)", value=True)

# weights (lenient defaults)
exp_weight = st.sidebar.slider("Experience weight (%)", min_value=10, max_value=30, value=20, step=5)
core_weight = st.sidebar.slider("Core marketplace weight (%)", min_value=25, max_value=45, value=30, step=5)
brand_weight = st.sidebar.slider("Brand building weight (%)", min_value=10, max_value=30, value=20, step=5)
perf_weight = st.sidebar.slider("Performance over time weight (%)", min_value=5, max_value=25, value=15, step=5)
scale_weight = st.sidebar.slider("Scaling success weight (%)", min_value=5, max_value=15, value=10, step=5)
total = exp_weight + core_weight + brand_weight + perf_weight + scale_weight
exp_w = exp_weight / total
core_w = core_weight / total
brand_w = brand_weight / total
perf_w = perf_weight / total
scale_w = scale_weight / total

st.sidebar.markdown("AKOI AI CV SCREEN — Weights determine how components combine into final score.")

# ---------- Main UI ----------
st.title(APP_TITLE)
st.markdown("### AI-assisted marketplace CV screening")

uploaded = st.file_uploader("Upload CV (PDF) — text PDFs extract best", type=["pdf"])
if not uploaded:
    st.info("Upload a CV PDF to score.")
    st.stop()

# read bytes
try:
    file_bytes = uploaded.read()
except Exception as e:
    st.error(f"Unable to read file: {e}")
    st.stop()

# extract text
with st.spinner("Extracting text..."):
    extracted_text = extract_text_from_pdf(file_bytes)

if not extracted_text.strip():
    st.error("Could not extract text from PDF. It might be scanned. Use OCR and upload a text PDF.")
    st.stop()

# ---------- Compute scores ----------
years_exp = extract_years_experience(extracted_text)
# Experience score (lenient mapping)
exp_score = round(min(10.0, (years_exp ** 0.85) * 1.25), 1)

core_score = score_core_marketplace(extracted_text)
brand_score = score_keywords(extracted_text, BRAND_BUILDING) if numeric_boost_enabled else score_keywords(extracted_text, BRAND_BUILDING, numeric_boost=False)
perf_score = score_keywords(extracted_text, PERFORMANCE)
scale_score = score_keywords(extracted_text, SCALING)

combined = (
    exp_score * exp_w
    + core_score * core_w
    + brand_score * brand_w
    + perf_score * perf_w
    + scale_score * scale_w
)

# lenient nudge
best_component = max(exp_score, core_score, brand_score, perf_score, scale_score)
final_score_raw = min(10.0, combined + 0.2 * (best_component - combined))

# quickcommerce bonus if enabled
qc_score = score_keywords(extracted_text, KPIS.get("Quickcommerce Experience", []))
qc_bonus_val = 1.0 if (quickcommerce_bonus and qc_score > 0) else 0.0
final_score = round(min(10.0, final_score_raw + qc_bonus_val), 2)

platform_scores = score_platforms(extracted_text)

# ---------- Determine recommendation and emoji per thresholds ----------
if final_score < 3.7:
    recommendation = "Cannot Proceed"
    emoji = "🙁"
    badge_color = hex_to_rgba(COLOR_LOW, 0.9)
elif 3.7 <= final_score < 5.0:
    recommendation = "Proceed to Next Round"
    emoji = "🙂"
    badge_color = hex_to_rgba(COLOR_MED, 0.95)
else:
    recommendation = "Good Candidate"
    emoji = "😄"
    badge_color = hex_to_rgba(COLOR_HIGH, 0.95)

# ---------- 3D glass-style recommendation card (Option A) ----------
card_html = f"""
<div style="
  border-radius:12px;
  padding:18px;
  margin-bottom:10px;
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  box-shadow: 0 6px 18px rgba(22, 28, 45, 0.20);
  border: 1px solid rgba(255,255,255,0.06);
  display:flex;
  align-items:center;
  justify-content:space-between;
">
  <div style="display:flex; align-items:center;">
    <div style="width:64px; height:64px; border-radius:10px; background:{badge_color}; display:flex; align-items:center; justify-content:center; font-size:28px; margin-right:12px;">
      {emoji}
    </div>
    <div>
      <div style="font-weight:700; font-size:18px; color:#0b2545;">{recommendation}</div>
      <div style="font-size:13px; color:#243b55;">Final score: <strong>{final_score} / 10</strong></div>
    </div>
  </div>
  <div style="text-align:right; color:#243b55;">
    <div style="font-size:12px">AKOI AI CV SCREEN</div>
    <div style="font-size:11px; color:#4b6584">Automated screening — validate in interview</div>
  </div>
</div>
"""
st.markdown(card_html, unsafe_allow_html=True)

# ---------- Gauge (car-style) ----------
low_rgba = hex_to_rgba(COLOR_LOW, 0.35)
med_rgba = hex_to_rgba(COLOR_MED, 0.35)
high_rgba = hex_to_rgba(COLOR_HIGH, 0.35)
bar_color = COLOR_HIGH if final_score >= 5.0 else COLOR_MED if final_score >= 3.7 else COLOR_LOW

g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=final_score,
    number={'suffix': " /10", 'font': {'size': 28}},
    gauge={
        'axis': {'range': [0, 10]},
        'bar': {'color': bar_color},
        'steps': [
            {'range': [0, 3.7], 'color': low_rgba},
            {'range': [3.7, 5.0], 'color': med_rgba},
            {'range': [5.0, 10], 'color': high_rgba}
        ],
        'threshold': {'line': {'color': "#000000", 'width': 3}, 'value': final_score}
    },
    title={'text': "Final Marketplace Score", 'font': {'size': 14}}
))
g.update_layout(height=260, margin=dict(l=40, r=40, t=10, b=10))
st.plotly_chart(g, use_container_width=True)

# ---------- Replace preview with 5-bullet auto summary ----------
bullets, overview_para, risk_para = generate_summary(extracted_text, platform_scores, core_score)
st.markdown("### Candidate Snapshot — Auto summary (5 quick bullets)")
for b in bullets:
    st.markdown(f"- {b}")

st.markdown("---")
st.markdown("**Overview:**")
st.write(overview_para)
st.markdown("**Risk / Notes:**")
st.write(risk_para)

# ---------- Charts and tables ----------
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Platform Experience Distribution (Pie)")
    plat_names = list(platform_scores.keys())
    plat_vals = [platform_scores[p] for p in plat_names]
    if sum(plat_vals) == 0:
        plat_vals = [1]*len(plat_vals)
    color_sequence = [PIE_PALETTE[i % len(PIE_PALETTE)] for i in range(len(plat_names))]
    fig_pie = go.Figure(go.Pie(labels=plat_names, values=plat_vals, hole=0.3,
                               marker=dict(colors=color_sequence, line=dict(color="#111827", width=1))))
    fig_pie.update_traces(textinfo='label+percent', textposition='inside', sort=False)
    fig_pie.update_layout(height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("KPI Scores (detected from CV)")
    rows = []
    for k in KPIS:
        rows.append([k, score_keywords(extracted_text, KPIS[k])])
    kpi_df = pd.DataFrame(rows, columns=["KPI", "Score"]).sort_values("Score", ascending=False)
    fig_bar = px.bar(kpi_df, x="Score", y="KPI", orientation='h', color="Score",
                     color_continuous_scale=[COLOR_LOW, COLOR_MED, COLOR_HIGH], range_x=[0,10])
    fig_bar.update_layout(height=520, margin=dict(l=260, r=20, t=30, b=30))
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Component Scores")
    comp_df = pd.DataFrame([
        ["Experience (yrs)", years_exp],
        ["Experience Score", exp_score],
        ["Core Marketplace (avg)", core_score],
        ["Brand Building", brand_score],
        ["Performance Over Time", perf_score],
        ["Scaling Success", scale_score],
        ["Quickcommerce Bonus", qc_bonus_val]
    ], columns=["Component", "Value"])
    # show a clean table with relevant scores
    st.table(comp_df.style.format({"Value": "{:.1f}"}))

    st.markdown("---")
    st.subheader("Final Recommendation")
    st.markdown(f"**{recommendation}** {emoji}")
    st.markdown(f"Final score: **{final_score} / 10**")

