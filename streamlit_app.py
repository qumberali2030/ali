# Inject custom CSS
with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
import os, joblib, numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# ───────────────────────────────────────────────
# 1 ▸ DEFINE CUSTOM TRANSFORMERS FIRST
# ───────────────────────────────────────────────

class CBCDomainCleaner(BaseEstimator, TransformerMixin):
    """Removes biologically implausible CBC values"""
    def __init__(self, domain_ranges=None):
        self.domain_ranges = domain_ranges or {
            "HB": (3, 25), "RBC": (1.5, 8), "HCT": (10, 70),
            "MCV": (40, 130), "MCH": (8, 45), "RDW": (5, 30), "MCHC": (20, 40)
        }
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_clean = X.copy()
        for col, (low, high) in self.domain_ranges.items():
            if col in X_clean.columns:
                X_clean = X_clean[(X_clean[col] >= low) & (X_clean[col] <= high)]
        return X_clean.reset_index(drop=True)

class CBCColumnMapper(BaseEstimator, TransformerMixin):
    """Ensures consistent column ordering for the model"""
    def __init__(self):
        self.columns = ['HB', 'RBC', 'HCT', 'MCV', 'MCH', 'RDW', 'MCHC']
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.columns]

class DiagnosticIndexCalculator(BaseEstimator, TransformerMixin):
    """Adds diagnostic indices for thalassemia screening"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X["Mentzer"]    = X["MCV"] / X["RBC"]
        X["ShineLal"]   = (X["MCV"]**2 * X["MCH"]) / 100
        X["Ehsani"]     = X["MCV"] - (10 * X["RBC"])
        X["GreenKing"]  = (X["MCV"]**2 * X["RDW"]) / (X["HB"] * 100)
        X["Srivastava"] = X["MCH"] / X["RBC"]
        return X
class TargetProcessor(BaseEstimator, TransformerMixin):
    """Dummy target transformer (kept for backward compatibility)"""
    def fit(self, y, *_):
        return self
    def transform(self, y):
        return y

# ───────────────────────────────────────────────
# 2 ▸ LOAD THE MODEL
# ───────────────────────────────────────────────
# ---------- 4 ▸ MODEL LOADING -------------------------------------------------#
MODEL_PATH = "cbc_xgb_model_complete.pkl"

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

# ✅ Load and patch the artifacts
artifacts = load_artifacts()
if artifacts is None:
    st.stop()

# ✅ Patch missing column_mapper attributes (backward compatibility)
if "column_mapper" in artifacts and not hasattr(artifacts["column_mapper"], "columns"):
    artifacts["column_mapper"].columns = ['HB', 'RBC', 'HCT', 'MCV', 'MCH', 'RDW', 'MCHC']

# ✅ Also patch old TargetProcessor if needed (prevent earlier AttributeError)
if "pipeline" in artifacts:
    for step_name, step_obj in artifacts["pipeline"].steps:
        if step_name == "preprocess" and hasattr(step_obj, "steps"):
            for sub_step_name, sub_step_obj in step_obj.steps:
                if type(sub_step_obj).__name__ == "TargetProcessor":
                    # Replace with a dummy working version
                    class TargetProcessor(BaseEstimator, TransformerMixin):
                        def fit(self, y, *_): return self
                        def transform(self, y): return y
                    sub_step_obj.__class__ = TargetProcessor

# ───────────────────────────────────────────────
# 3 ▸ BUSINESS LOGIC
# ───────────────────────────────────────────────

def calc_indices(cbc):
    """Compute diagnostic indices for UI display"""
    return {
        "Mentzer":    cbc["MCV"] / cbc["RBC"],
        "ShineLal":   (cbc["MCV"]**2 * cbc["MCH"]) / 100,
        "Ehsani":     cbc["MCV"] - (10 * cbc["RBC"]),
        "GreenKing":  (cbc["MCV"]**2 * cbc["RDW"]) / (cbc["HB"] * 100),
        "Srivastava": cbc["MCH"] / cbc["RBC"]
    }

def predict(cbc):
    """Make prediction for a single CBC dictionary"""
    raw    = pd.DataFrame([cbc])
    mapped = artifacts["column_mapper"].transform(raw)
    idx_df = pd.DataFrame([calc_indices(cbc)])
    X      = pd.concat([mapped.reset_index(drop=True), idx_df], axis=1)
    model  = artifacts["pipeline"]

    pred      = model.predict(X)[0]
    proba_vec = model.predict_proba(X)[0]
    label     = "Positive" if pred == 1 else "Negative"
    conf      = max(proba_vec)
    return label, conf, proba_vec

def proba_bar(probas):
    """Plot prediction probabilities as a bar chart"""
    fig = go.Figure(
        go.Bar(
            x=["Negative", "Positive"],
            y=probas,
            marker_color=["#2ecc71", "#e74c3c"],
            text=[f"{p:.2%}" for p in probas],
            textposition="auto",
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, b=10, t=10))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────
# 4 ▸ UI FUNCTIONS
# ───────────────────────────────────────────────

def local_css(file_path: str):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def header():
    st.markdown(
        """
        <div class="main-header">
            <h1>🧬 SMART THALASSEMIA DETECTOR 🧬</h1>
            <p>AI-Powered CBC Analysis for Early Screening</p>
            <p>Confidence in Every Call · Your Digital Hematology Assistant</p>
        </div>
        """, unsafe_allow_html=True
    )

def single_patient():
    st.subheader("Enter CBC Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        hb  = st.number_input("HB (g/dL)",       3.0, 25.0, 12.5)
        rbc = st.number_input("RBC (×10⁶/µL)",   1.5, 8.0, 4.5)
        hct = st.number_input("HCT (%)",        10.0, 70.0, 38.0)
    with col2:
        mcv = st.number_input("MCV (fL)",       40.0,130.0,85.0)
        mch = st.number_input("MCH (pg)",        8.0, 45.0,28.0)
        rdw = st.number_input("RDW (%)",         5.0, 30.0,13.5)
    with col3:
        mchc= st.number_input("MCHC (g/dL)",    20.0, 40.0,33.0)

    if st.button("🔍 Predict", type="primary"):
        cbc = dict(HB=hb, RBC=rbc, HCT=hct, MCV=mcv, MCH=mch, RDW=rdw, MCHC=mchc)
        label, conf, probas = predict(cbc)
        color = "#28a745" if label=="Negative" else "#dc3545"
        icon  = "✅" if label=="Negative" else "⚠️"
        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <div style="font-size:3rem;">{icon}</div>
                <h2 style="color:{color};">Prediction: {label}</h2>
                <p style="color:#6c757d;font-size:1.1rem;">Confidence: {conf:.1%}</p>
            </div>
            """, unsafe_allow_html=True
        )
        proba_bar(probas)
        st.subheader("📌 Diagnostic Indices")
        st.dataframe(pd.DataFrame([calc_indices(cbc)]).T.rename(columns={0: "Value"}))

def batch_mode():
    st.subheader("📦 Upload CBC Data File (CSV or Excel)")
    file = st.file_uploader("Upload CBC file", type=["csv", "xlsx"])
    
    if file and st.button("🚀 Run Predictions", type="primary"):
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        df.columns = df.columns.str.strip().str.upper()

        required_cols = ["HB", "RBC", "HCT", "MCV", "MCH", "RDW", "MCHC"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            st.stop()

        results = []
        skipped = 0

        for _, row in df.iterrows():
            cbc = {k: row[k] for k in required_cols}
            try:
                label, conf, _ = predict(cbc)
                results.append({**cbc, **calc_indices(cbc),
                                "Prediction": label, "Confidence": f"{conf:.2%}"})
            except ValueError:
                # ✅ Skip invalid rows (cleaner removed them)
                skipped += 1

        if not results:
            st.error("❌ No valid rows found (all removed by biological plausibility filter).")
            return

        out = pd.DataFrame(results)
        st.dataframe(out, height=450)
        st.download_button("⬇️ Download Results", out.to_csv(index=False),
                           file_name="thalassemia_predictions.csv")
        
        if skipped > 0:
            st.warning(f"⚠️ {skipped} row(s) were skipped due to biologically implausible values.")


# ───────────────────────────────────────────────
# 5 ▸ MAIN
# ───────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SMART THALASSEMIA DETECTOR", layout="wide")
    local_css("assets/custom.css")
    header()
    mode = st.radio("Select Mode", ["Single Patient", "Batch Upload"], horizontal=True)
    single_patient() if mode == "Single Patient" else batch_mode()
    st.markdown('<div class="footer">© 2025 • For research & screening use only</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
