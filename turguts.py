import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from io import BytesIO
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IC50 Calculator",
    page_icon="🧬",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}
h1 {
    color: #1e3a8a;
    font-weight: 700;
}
.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin: 1rem 0;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
}
.metric-label {
    text-transform: uppercase;
    font-size: 0.9rem;
    opacity: 0.9;
}
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LANGUAGE DICTIONARY
# =========================
LANG = {
    "tr": {
        "title": "🧬 IC₅₀ Hesaplayıcı",
        "subtitle": "4-Parametreli Lojistik Regresyon (4PL)",
        "compound": "🔬 Madde adı",
        "settings": "⚙️ Deney Ayarları",
        "rep": "Tekrar sayısı",
        "conc": "Konsantrasyon sayısı",
        "unit": "Birim",
        "control": "🎯 Kontrol absorbansları",
        "table": "📊 Absorbans Tablosu",
        "calc": "🧮 IC₅₀ HESAPLA",
        "success": "Hesaplama başarılı",
        "error": "Hesaplama yapılamadı. Verileri kontrol edin.",
        "xlab": "Konsantrasyon ({unit})",
        "ylab": "Normalize yanıt (%)",
        "data": "Veri",
        "fit": "4PL uyum",
        "ic50": "IC₅₀ Değeri",
        "download_png": "PNG indir",
        "download_pdf": "PDF indir",
        "author": "Turgut Şekerler"
    },
    "en": {
        "title": "🧬 IC₅₀ Calculator",
        "subtitle": "4-Parameter Logistic Regression (4PL)",
        "compound": "🔬 Compound name",
        "settings": "⚙️ Experiment Settings",
        "rep": "Replicates",
        "conc": "Concentrations",
        "unit": "Unit",
        "control": "🎯 Control absorbance",
        "table": "📊 Absorbance Table",
        "calc": "🧮 CALCULATE IC₅₀",
        "success": "Calculation successful",
        "error": "Calculation failed. Check your data.",
        "xlab": "Concentration ({unit})",
        "ylab": "Normalized response (%)",
        "data": "Data",
        "fit": "4PL fit",
        "ic50": "IC₅₀ Value",
        "download_png": "Download PNG",
        "download_pdf": "Download PDF",
        "author": "Turgut Sekerler"
    }
}

language = st.selectbox("🌍 Language / Dil", ["Türkçe", "English"])
lang = "tr" if language == "Türkçe" else "en"
T = LANG[lang]

# =========================
# HEADER
# =========================
st.markdown(f"# {T['title']}")
st.caption(T["subtitle"])
st.caption(T["author"])

compound_name = st.text_input(T["compound"])

# =========================
# 4PL MODEL
# =========================
def four_pl(x, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill)

# =========================
# SETTINGS
# =========================
st.markdown(f"### {T['settings']}")
c1, c2, c3 = st.columns(3)
with c1:
    reps = st.number_input(T["rep"], 2, 8, 3)
with c2:
    nconc = st.number_input(T["conc"], 2, 12, 3)
with c3:
    unit = st.selectbox(T["unit"], ["nM", "µM", "mg/mL", "µg/mL"])

# =========================
# CONTROL
# =========================
st.markdown(f"### {T['control']}")
control_vals = []
for i, col in enumerate(st.columns(reps)):
    control_vals.append(col.number_input(f"{i+1}", value=0.0, format="%.4f"))

# =========================
# TABLE
# =========================
st.markdown(f"### {T['table']}")
table = pd.DataFrame(
    np.zeros((nconc, reps + 1)),
    columns=["Concentration"] + [f"Rep {i+1}" for i in range(reps)]
)
edited = st.data_editor(table, use_container_width=True, hide_index=True)

# =========================
# CALCULATION
# =========================
if st.button(T["calc"], use_container_width=True):
    try:
        df = edited.apply(pd.to_numeric, errors="coerce").dropna()
        concs = df.iloc[:, 0].values
        abs_vals = df.iloc[:, 1:].values

        control_mean = np.mean(control_vals)
        response = (abs_vals.mean(axis=1) / control_mean) * 100

        p0 = [0, 100, np.median(concs), 1.0]
        bounds = ([0, 80, 0, 0.1], [20, 120, max(concs) * 10, 5])

        popt, _ = curve_fit(
            four_pl, concs, response,
            p0=p0, bounds=bounds, maxfev=30000
        )

        ic50 = popt[2]
        st.success(T["success"])

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{T['ic50']}</div>
            <div class="metric-value">{ic50:.4g} {unit}</div>
            <div class="metric-label">{compound_name or "—"}</div>
        </div>
        """, unsafe_allow_html=True)

        # ===== SHARED AXIS =====
        xmin = min(concs) / 2
        xmax = max(concs) * 2
        xfit = np.logspace(np.log10(xmin), np.log10(xmax), 400)
        yfit = four_pl(xfit, *popt)

        # ===== PLOTLY =====
        fig = go.Figure()
        fig.add_scatter(x=concs, y=response, mode="markers", name=T["data"])
        fig.add_scatter(x=xfit, y=yfit, mode="lines", name=T["fit"])
        fig.add_vline(x=ic50, line_dash="dash", annotation_text="IC₅₀")

        fig.update_xaxes(
            type="log",
            range=[np.log10(xmin), np.log10(xmax)],
            title=T["xlab"].format(unit=unit)
        )
        fig.update_yaxes(title=T["ylab"], range=[0, 110])
        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        # ===== MATPLOTLIB DOWNLOAD =====
        fig_mpl, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(concs, response)
        ax.plot(xfit, yfit)
        ax.axvline(ic50, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 110)
        ax.set_xlabel(T["xlab"].format(unit=unit))
        ax.set_ylabel(T["ylab"])
        ax.set_title(compound_name or "IC50 curve")

        col1, col2 = st.columns(2)
        with col1:
            buf = BytesIO()
            fig_mpl.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(T["download_png"], buf.getvalue(), "IC50.png", "image/png")
        with col2:
            buf = BytesIO()
            fig_mpl.savefig(buf, format="pdf", bbox_inches="tight")
            st.download_button(T["download_pdf"], buf.getvalue(), "IC50.pdf", "application/pdf")

        plt.close(fig_mpl)

    except Exception as e:
        st.error(T["error"])
        st.write(str(e))
