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
st.caption("v1.1 · 4-parametreli lojistik regresyon (4PL), %95 güven aralığı")

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
with st.container():
    st.markdown("### 🧪 Deney Ayarları")

    col1, col2 = st.columns(2)

    with col1:
        replicates = st.number_input(
            "Tekrar sayısı",
            min_value=2,
            max_value=8,
            value=2,
            step=1
        )

    with col2:
        num_conc = st.number_input(
            "Konsantrasyon sayısı",
            min_value=2,
            max_value=12,
            value=3,
            step=1
        )

    unit = st.selectbox(
        "Konsantrasyon birimi",
        ["nM", "µM", "mg/mL", "µg/mL"]
    )

# =========================
# CONTROL INPUT
# =========================
st.markdown("### Kontrol kuyucukları (Absorbans)")

control_cols = st.columns(replicates)
control_vals = []

for i, col in enumerate(control_cols):
    control_vals.append(
        col.number_input(
            f"Kontrol {i+1}",
            value=0.0,
            format="%.4f"
        )
    )

# =========================
# DATA TABLE (COPY–PASTE)
# =========================
st.markdown("### 📋 Absorbans Tablosu")
st.caption("Excel’den hücreleri kopyalayıp tabloya direkt yapıştırabilirsiniz.")

# Create empty table
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

        concentrations = edited_table["Konsantrasyon"].values.astype(float)
        absorbance_vals = edited_table.iloc[:, 1:].values.astype(float)

        absorbance_means = absorbance_vals.mean(axis=1)

        # Normalize
        response = (absorbance_means / control_mean) * 100

        # Initial guesses + bounds
        p0 = [
            np.min(response),
            np.max(response),
            np.median(concentrations),
            1.0
        ]

        bounds = (
            [0, 50, 0, 0.1],
            [100, 120, np.inf, 5]
        )

        popt, pcov = curve_fit(
            four_pl,
            concentrations,
            response,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )

        bottom, top, ic50, hill = popt
        ic50_se = np.sqrt(pcov[2, 2])
        ci_low = ic50 - 1.96 * ic50_se
        ci_high = ic50 + 1.96 * ic50_se

        st.success(
            f"**Madde:** {compound_name if compound_name else '—'}  \n"
            f"**IC₅₀ = {ic50:.4g} {unit}**  \n"
            f"95% CI: {ci_low:.4g} – {ci_high:.4g}"
        )

        # Plot
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

        title = f"IC₅₀ Eğrisi – {compound_name}" if compound_name else "IC₅₀ Eğrisi"
        ax.set_title(title)

        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error("Hesaplama yapılamadı. Lütfen tabloyu ve kontrol değerlerini kontrol edin.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "**How to cite:**  \n"
    "Şekerler, T. *IC₅₀ Calculator* (v1.1).  \n"
    "https://turgut-sekerler-ic50.streamlit.app"
)
