import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from io import BytesIO

# =========================
# LANGUAGE DICTIONARY
# =========================
LANG = {
    "tr": {
        "title": "Turgut Şekerler – IC₅₀ Hesaplayıcı",
        "version": "v1.2 · 4-parametreli lojistik regresyon (4PL), %95 güven aralığı",
        "lang": "Dil",
        "compound": "IC₅₀ hesaplanacak madde adı",
        "settings": "🧪 Deney Ayarları",
        "rep": "Tekrar sayısı",
        "conc": "Konsantrasyon sayısı",
        "unit": "Konsantrasyon birimi",
        "control": "Kontrol kuyucukları (Absorbans)",
        "table": "📋 Absorbans Tablosu",
        "table_hint": "Excel’den kopyalayıp tabloya direkt yapıştırabilirsiniz.",
        "calc": "IC₅₀ HESAPLA",
        "error": "Hesaplama yapılamadı. Lütfen tabloyu ve kontrol değerlerini kontrol edin.",
        "result": "Sonuç",
        "xlab": "Konsantrasyon ({unit})",
        "ylab": "Normalize yanıt (%)",
        "curve": "IC₅₀ Eğrisi – {compound}",
        "data": "Veri",
        "fit": "4PL uyum",
        "download_png": "PNG olarak indir",
        "download_pdf": "PDF olarak indir",
        "cite": "How to cite",
    },
    "en": {
        "title": "Turgut Sekerler – IC₅₀ Calculator",
        "version": "v1.2 · 4-parameter logistic regression (4PL), 95% confidence interval",
        "lang": "Language",
        "compound": "Compound name for IC₅₀ calculation",
        "settings": "🧪 Experiment Settings",
        "rep": "Number of replicates",
        "conc": "Number of concentrations",
        "unit": "Concentration unit",
        "control": "Control wells (Absorbance)",
        "table": "📋 Absorbance Table",
        "table_hint": "Copy from Excel and paste directly into the table.",
        "calc": "CALCULATE IC₅₀",
        "error": "Calculation failed. Please check the table and control values.",
        "result": "Result",
        "xlab": "Concentration ({unit})",
        "ylab": "Normalized response (%)",
        "curve": "IC₅₀ Curve – {compound}",
        "data": "Data",
        "fit": "4PL fit",
        "download_png": "Download PNG",
        "download_pdf": "Download PDF",
        "cite": "How to cite",
    }
}

# =========================
# LANGUAGE SELECT
# =========================
language = st.selectbox("Language / Dil", ["Türkçe", "English"])
lang = "tr" if language == "Türkçe" else "en"
T = LANG[lang]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title=T["title"], layout="centered")

# =========================
# HEADER
# =========================
st.title(T["title"])
st.caption(T["version"])

# =========================
# COMPOUND NAME
# =========================
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
col1, col2 = st.columns(2)
with col1:
    replicates = st.number_input(T["rep"], min_value=2, max_value=8, value=3)
with col2:
    num_conc = st.number_input(T["conc"], min_value=2, max_value=12, value=3)

unit = st.selectbox(T["unit"], ["nM", "µM", "mg/mL", "µg/mL"])

# =========================
# CONTROL
# =========================
st.markdown(f"### {T['control']}")
control_vals = [
    col.number_input(f"{i+1}", value=0.0, format="%.4f")
    for i, col in enumerate(st.columns(replicates))
]

# =========================
# TABLE
# =========================
st.markdown(f"### {T['table']}")
st.caption(T["table_hint"])

table_data = pd.DataFrame(
    np.zeros((num_conc, replicates + 1)),
    columns=["Concentration"] + [f"Replicate {i+1}" for i in range(replicates)]
)

edited = st.data_editor(table_data, use_container_width=True, num_rows="fixed")

# =========================
# CALCULATION
# =========================
st.markdown("---")
if st.button(T["calc"]):
    try:
        df = edited.apply(pd.to_numeric, errors="coerce").dropna()
        concs = df.iloc[:, 0].values
        abs_vals = df.iloc[:, 1:].values

        response = (abs_vals.mean(axis=1) / np.mean(control_vals)) * 100

        p0 = [min(response), max(response), np.exp(np.mean(np.log(concs))), 1.0]
        bounds = ([0, 50, 0, 0.1], [100, 120, max(concs) * 10, 5])

        popt, pcov = curve_fit(four_pl, concs, response, p0=p0, bounds=bounds, maxfev=30000)
        ic50 = popt[2]

        # Plot
        xfit = np.logspace(np.log10(min(concs)), np.log10(max(concs)), 300)
        yfit = four_pl(xfit, *popt)

        fig, ax = plt.subplots()
        ax.scatter(concs, response, label=T["data"])
        ax.plot(xfit, yfit, label=T["fit"])
        ax.axvline(ic50, linestyle="--", label="IC₅₀")
        ax.set_xscale("log")
        ax.set_xlabel(T["xlab"].format(unit=unit))
        ax.set_ylabel(T["ylab"])
        ax.set_title(T["curve"].format(compound=compound_name or "—"))
        ax.legend()

        st.pyplot(fig)

        # DOWNLOADS
        buf_png = BytesIO()
        fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            T["download_png"],
            buf_png.getvalue(),
            file_name=f"IC50_{compound_name or 'plot'}.png",
            mime="image/png"
        )

        buf_pdf = BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        st.download_button(
            T["download_pdf"],
            buf_pdf.getvalue(),
            file_name=f"IC50_{compound_name or 'plot'}.pdf",
            mime="application/pdf"
        )

    except Exception:
        st.error(T["error"])

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    f"**{T['cite']}:**  \n"
    f"Şekerler, T. *IC₅₀ Calculator* (v1.2).  \n"
    f"https://turgut-sekerler-ic50.streamlit.app"
)
