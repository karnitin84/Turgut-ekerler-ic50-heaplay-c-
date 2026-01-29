import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Turgut Şekerler – IC₅₀ Hesaplayıcı",
    layout="centered"
)

# =========================
# HEADER
# =========================
st.title("Turgut Şekerler – IC₅₀ Hesaplayıcı")
st.caption("v1.1.1 · 4-parametreli lojistik regresyon (4PL), %95 güven aralığı")

# =========================
# COMPOUND NAME
# =========================
compound_name = st.text_input(
    "IC₅₀ hesaplanacak madde adı",
    placeholder="Örn: Jaceidin, Compound X, Extract A"
)

# =========================
# 4PL MODEL
# =========================
def four_pl(x, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill)

# =========================
# EXPERIMENT SETTINGS
# =========================
st.markdown("### 🧪 Deney Ayarları")

col1, col2 = st.columns(2)
with col1:
    replicates = st.number_input("Tekrar sayısı", min_value=2, max_value=8, value=3)
with col2:
    num_conc = st.number_input("Konsantrasyon sayısı", min_value=2, max_value=12, value=3)

unit = st.selectbox("Konsantrasyon birimi", ["nM", "µM", "mg/mL", "µg/mL"])

# =========================
# CONTROL INPUT
# =========================
st.markdown("### Kontrol kuyucukları (Absorbans)")

control_cols = st.columns(replicates)
control_vals = [
    col.number_input(f"Kontrol {i+1}", value=0.0, format="%.4f")
    for i, col in enumerate(control_cols)
]

# =========================
# DATA TABLE
# =========================
st.markdown("### 📋 Absorbans Tablosu")
st.caption("Excel’den kopyalayıp tabloya direkt yapıştırabilirsiniz.")

table_data = pd.DataFrame(
    np.zeros((num_conc, replicates + 1)),
    columns=["Konsantrasyon"] + [f"Tekrar {i+1}" for i in range(replicates)]
)

edited_table = st.data_editor(
    table_data,
    use_container_width=True,
    num_rows="fixed"
)

# =========================
# CALCULATION
# =========================
st.markdown("---")

if st.button("IC₅₀ HESAPLA"):
    try:
        control_mean = np.mean(control_vals)

        df = edited_table.apply(pd.to_numeric, errors="coerce").dropna()

        concentrations = df["Konsantrasyon"].values
        if np.any(concentrations <= 0):
            st.error("Konsantrasyon değerleri 0 veya negatif olamaz.")
            st.stop()

        absorbance_vals = df.iloc[:, 1:].values
        absorbance_means = absorbance_vals.mean(axis=1)

        response = (absorbance_means / control_mean) * 100

        p0 = [
            min(response),
            max(response),
            np.exp(np.mean(np.log(concentrations))),
            1.0
        ]

        bounds = (
            [0, 50, 0, 0.1],
            [100, 120, np.max(concentrations) * 10, 5]
        )

        popt, pcov = curve_fit(
            four_pl,
            concentrations,
            response,
            p0=p0,
            bounds=bounds,
            maxfev=30000
        )

        ic50 = popt[2]
        ic50_se = np.sqrt(pcov[2, 2])
        ci_low = ic50 - 1.96 * ic50_se
        ci_high = ic50 + 1.96 * ic50_se

        max_conc = np.max(concentrations)

        if ic50 > max_conc:
            st.warning(f"**Madde:** {compound_name or '—'}  \nIC₅₀ > {max_conc:.4g} {unit}")
        else:
            st.success(
                f"**Madde:** {compound_name or '—'}  \n"
                f"**IC₅₀ = {ic50:.4g} {unit}**  \n"
                f"95% CI: {ci_low:.4g} – {ci_high:.4g}"
            )

        x_fit = np.logspace(
            np.log10(min(concentrations)),
            np.log10(max(concentrations)),
            300
        )
        y_fit = four_pl(x_fit, *popt)

        fig, ax = plt.subplots()
        ax.scatter(concentrations, response, label="Veri")
        ax.plot(x_fit, y_fit, label="4PL uyum")
        ax.axvline(ic50, linestyle="--", label="IC₅₀")
        ax.set_xscale("log")
        ax.set_xlabel(f"Konsantrasyon ({unit})")
        ax.set_ylabel("Normalize yanıt (%)")
        ax.set_title(f"IC₅₀ Eğrisi – {compound_name}" if compound_name else "IC₅₀ Eğrisi")
        ax.legend()
        st.pyplot(fig)

    except Exception:
        st.error("Hesaplama yapılamadı. Lütfen tabloyu ve kontrol değerlerini kontrol edin.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "**How to cite:**  \n"
    "Şekerler, T. *IC₅₀ Calculator* (v1.1.1).  \n"
    "https://turgut-sekerler-ic50.streamlit.app"
)
