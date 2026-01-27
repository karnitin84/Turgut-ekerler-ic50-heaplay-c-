import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

st.set_page_config(page_title="Turgut Şekerler – IC₅₀ Hesaplayıcı", layout="centered")

# =========================
# TITLE
# =========================
st.title("Turgut Şekerler – IC₅₀ Hesaplayıcı")
st.caption("4-parametreli lojistik regresyon (4PL), %95 güven aralığı ile")

# =========================
# 4PL MODEL
# =========================
def four_pl(x, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill)

# =========================
# EXPERIMENT SETUP
# =========================
st.header("Deney Ayarları")

replicates = st.number_input(
    "Tekrar sayısı",
    min_value=2,
    max_value=8,
    value=2,
    step=1
)

num_conc = st.number_input(
    "Konsantrasyon sayısı",
    min_value=2,
    max_value=10,
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
st.header("Kontrol kuyucukları (Absorbans)")

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
# CONCENTRATION DATA
# =========================
st.header("Konsantrasyon verileri")

concentrations = []
abs_means = []

for i in range(int(num_conc)):
    st.subheader(f"Konsantrasyon {i+1}")

    conc = st.number_input(
        "Konsantrasyon değeri",
        key=f"conc_{i}",
        format="%.6f"
    )
    concentrations.append(conc)

    cols = st.columns(replicates)
    abs_vals = []

    for j, col in enumerate(cols):
        abs_vals.append(
            col.number_input(
                f"Abs {j+1}",
                key=f"abs_{i}_{j}",
                format="%.4f"
            )
        )

    abs_means.append(np.mean(abs_vals))

# =========================
# CALCULATION
# =========================
if st.button("IC₅₀ Hesapla"):

    try:
        control_mean = np.mean(control_vals)

        concentrations = np.array(concentrations, dtype=float)
        absorbance_means = np.array(abs_means, dtype=float)

        # Normalize (% of control)
        response = (absorbance_means / control_mean) * 100

        # -------------------------
        # INITIAL GUESSES + BOUNDS
        # -------------------------
        p0 = [
            np.min(response),              # bottom
            np.max(response),              # top
            np.median(concentrations),     # IC50
            1.0                             # Hill
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
        max_conc = np.max(concentrations)

        # -------------------------
        # 95% CONFIDENCE INTERVAL
        # -------------------------
        ic50_se = np.sqrt(pcov[2, 2])
        ci_low = ic50 - 1.96 * ic50_se
        ci_high = ic50 + 1.96 * ic50_se

        # -------------------------
        # REPORTING
        # -------------------------
        if ic50 > max_conc:
            st.warning(f"IC₅₀ > {max_conc} {unit}")
        else:
            st.success(
                f"IC₅₀ = {ic50:.4g} {unit}  "
                f"(95% CI: {ci_low:.4g} – {ci_high:.4g})"
            )

        # -------------------------
        # PLOT
        # -------------------------
        x_fit = np.logspace(
            np.log10(min(concentrations)),
            np.log10(max(concentrations)),
            300
        )
        y_fit = four_pl(x_fit, *popt)

        fig, ax = plt.subplots()
        ax.scatter(concentrations, response, label="Veri")
        ax.plot(x_fit, y_fit, label="4PL uyum")

        ax.axvline(ic50, linestyle="--", linewidth=1, label="IC₅₀")

        ax.set_xscale("log")
        ax.set_xlabel(f"Konsantrasyon ({unit})")
        ax.set_ylabel("Normalize yanıt (%)")
        ax.legend()

        st.pyplot(fig)

    except Exception:
        st.error("Girilen verilerle IC₅₀ ve güven aralığı hesaplanamadı.")
