import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from io import BytesIO
import matplotlib.pyplot as plt

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h3 {
        color: #334155;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info box */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Section container */
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Button customization */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background-color: #10b981;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.25rem;
    }
    
    .stDownloadButton > button:hover {
        background-color: #059669;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    /* Caption styling */
    .caption {
        color: #64748b;
        font-size: 0.9rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LANGUAGE DICTIONARY
# =========================
LANG = {
    "tr": {
        "title": "🧬 IC₅₀ Hesaplayıcı",
        "subtitle": "4-Parametreli Lojistik Regresyon (4PL) ile Profesyonel İnhibisyon Analizi",
        "version": "v1.2 · %95 güven aralığı",
        "author": "Turgut Şekerler",
        "lang": "🌍 Dil Seçimi",
        "compound": "🔬 Madde Adı",
        "compound_placeholder": "Örn: Cisplatin, Doxorubicin...",
        "settings": "⚙️ Deney Parametreleri",
        "rep": "Tekrar Sayısı",
        "conc": "Konsantrasyon Sayısı",
        "unit": "Birim",
        "control": "🎯 Kontrol Kuyucukları (Absorbans)",
        "table": "📊 Absorbans Verileri",
        "table_hint": "💡 İpucu: Excel'den kopyalayıp doğrudan tabloya yapıştırabilirsiniz.",
        "calc": "🧮 IC₅₀ HESAPLA",
        "error": "❌ Hesaplama yapılamadı. Lütfen verileri kontrol edin.",
        "result": "📈 Analiz Sonuçları",
        "xlab": "Konsantrasyon ({unit})",
        "ylab": "Normalize Yanıt (%)",
        "curve": "IC₅₀ Doz-Yanıt Eğrisi: {compound}",
        "data": "Deneysel Veri",
        "fit": "4PL Model",
        "download_png": "📥 PNG İndir",
        "download_pdf": "📥 PDF İndir",
        "cite": "📚 Alıntı",
        "ic50_label": "IC₅₀ Değeri",
        "success": "✅ Hesaplama başarıyla tamamlandı!",
        "instructions": "📖 Kullanım Kılavuzu",
        "step1": "1️⃣ Madde adını girin",
        "step2": "2️⃣ Deney parametrelerini ayarlayın",
        "step3": "3️⃣ Kontrol değerlerini girin",
        "step4": "4️⃣ Absorbans tablosunu doldurun",
        "step5": "5️⃣ 'IC₅₀ HESAPLA' butonuna tıklayın",
    },
    "en": {
        "title": "🧬 IC₅₀ Calculator",
        "subtitle": "Professional Inhibition Analysis with 4-Parameter Logistic Regression (4PL)",
        "version": "v1.2 · 95% confidence interval",
        "author": "Turgut Sekerler",
        "lang": "🌍 Language",
        "compound": "🔬 Compound Name",
        "compound_placeholder": "e.g., Cisplatin, Doxorubicin...",
        "settings": "⚙️ Experiment Parameters",
        "rep": "Number of Replicates",
        "conc": "Number of Concentrations",
        "unit": "Unit",
        "control": "🎯 Control Wells (Absorbance)",
        "table": "📊 Absorbance Data",
        "table_hint": "💡 Tip: Copy from Excel and paste directly into the table.",
        "calc": "🧮 CALCULATE IC₅₀",
        "error": "❌ Calculation failed. Please check your data.",
        "result": "📈 Analysis Results",
        "xlab": "Concentration ({unit})",
        "ylab": "Normalized Response (%)",
        "curve": "IC₅₀ Dose-Response Curve: {compound}",
        "data": "Experimental Data",
        "fit": "4PL Model",
        "download_png": "📥 Download PNG",
        "download_pdf": "📥 Download PDF",
        "cite": "📚 Citation",
        "ic50_label": "IC₅₀ Value",
        "success": "✅ Calculation completed successfully!",
        "instructions": "📖 User Guide",
        "step1": "1️⃣ Enter compound name",
        "step2": "2️⃣ Set experiment parameters",
        "step3": "3️⃣ Enter control values",
        "step4": "4️⃣ Fill in absorbance table",
        "step5": "5️⃣ Click 'CALCULATE IC₅₀' button",
    }
}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="IC₅₀ Calculator",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# LANGUAGE SELECT
# =========================
col_lang, col_space = st.columns([1, 3])
with col_lang:
    language = st.selectbox(
        "🌍",
        ["Türkçe", "English"],
        label_visibility="collapsed"
    )

lang = "tr" if language == "Türkçe" else "en"
T = LANG[lang]

# =========================
# HEADER
# =========================
st.markdown(f"# {T['title']}")
st.markdown(f"**{T['subtitle']}**")
st.caption(f"{T['version']} · {T['author']}")

# =========================
# INSTRUCTIONS (COLLAPSIBLE)
# =========================
with st.expander(T["instructions"], expanded=False):
    st.markdown(f"""
    {T['step1']}  
    {T['step2']}  
    {T['step3']}  
    {T['step4']}  
    {T['step5']}
    """)

st.markdown("---")

# =========================
# COMPOUND NAME
# =========================
compound_name = st.text_input(
    T["compound"],
    placeholder=T["compound_placeholder"]
)

# =========================
# 4PL MODEL
# =========================
def four_pl(x, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill)

# =========================
# SETTINGS
# =========================
st.markdown(f"### {T['settings']}")

col1, col2, col3 = st.columns(3)
with col1:
    replicates = st.number_input(
        T["rep"],
        min_value=2,
        max_value=8,
        value=3
    )
with col2:
    num_conc = st.number_input(
        T["conc"],
        min_value=2,
        max_value=12,
        value=3
    )
with col3:
    unit = st.selectbox(
        T["unit"],
        ["nM", "µM", "mg/mL", "µg/mL"]
    )

# =========================
# CONTROL
# =========================
st.markdown(f"### {T['control']}")
control_cols = st.columns(replicates)
control_vals = []
for i, col in enumerate(control_cols):
    with col:
        val = st.number_input(
            f"#{i+1}",
            value=0.0,
            format="%.4f",
            key=f"control_{i}"
        )
        control_vals.append(val)

# =========================
# TABLE
# =========================
st.markdown(f"### {T['table']}")
st.caption(T["table_hint"])

table_data = pd.DataFrame(
    np.zeros((num_conc, replicates + 1)),
    columns=["Concentration"] + [f"Rep {i+1}" for i in range(replicates)]
)

edited = st.data_editor(
    table_data,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True
)

# =========================
# CALCULATION
# =========================
st.markdown("---")

if st.button(T["calc"], use_container_width=True):
    with st.spinner('🔬 Hesaplama yapılıyor...' if lang == "tr" else '🔬 Calculating...'):
        try:
            df = edited.apply(pd.to_numeric, errors="coerce").dropna()
            concs = df.iloc[:, 0].values
            abs_vals = df.iloc[:, 1:].values

            response = (abs_vals.mean(axis=1) / np.mean(control_vals)) * 100

            p0 = [min(response), max(response), np.exp(np.mean(np.log(concs))), 1.0]
            bounds = ([0, 50, 0, 0.1], [100, 120, max(concs) * 10, 5])

            popt, pcov = curve_fit(
                four_pl,
                concs,
                response,
                p0=p0,
                bounds=bounds,
                maxfev=30000
            )
            ic50 = popt[2]

            # Success message
            st.success(T["success"])

            # =========================
            # RESULT METRIC
            # =========================
            st.markdown(f"### {T['result']}")
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{T['ic50_label']}</div>
                <div class="metric-value">{ic50:.4f} {unit}</div>
                <div class="metric-label">{compound_name or "—"}</div>
            </div>
            """, unsafe_allow_html=True)

            # =========================
            # PLOTLY GRAPH
            # =========================
            xfit = np.logspace(np.log10(min(concs)), np.log10(max(concs)), 300)
            yfit = four_pl(xfit, *popt)

            fig = go.Figure()

            # Experimental data points
            fig.add_trace(go.Scatter(
                x=concs,
                y=response,
                mode='markers',
                name=T["data"],
                marker=dict(
                    size=12,
                    color='#ef4444',
                    symbol='circle',
                    line=dict(width=2, color='white')
                )
            ))

            # 4PL fit curve
            fig.add_trace(go.Scatter(
                x=xfit,
                y=yfit,
                mode='lines',
                name=T["fit"],
                line=dict(color='#3b82f6', width=3)
            ))

            # IC50 vertical line
            fig.add_vline(
                x=ic50,
                line_dash="dash",
                line_color="#8b5cf6",
                line_width=2,
                annotation_text=f"IC₅₀ = {ic50:.4f} {unit}",
                annotation_position="top"
            )

            # Horizontal line at 50%
            fig.add_hline(
                y=50,
                line_dash="dot",
                line_color="#6b7280",
                line_width=1,
                opacity=0.5
            )

            fig.update_xaxes(
                type="log",
                title_text=T["xlab"].format(unit=unit),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200,200,200,0.3)'
            )

            fig.update_yaxes(
                title_text=T["ylab"],
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200,200,200,0.3)',
                range=[0, 110]
            )

            fig.update_layout(
                title=T["curve"].format(compound=compound_name or "—"),
                title_font_size=18,
                title_font_color='#1e3a8a',
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # DOWNLOAD BUTTONS
            # =========================
            st.markdown("### 💾 Downloads")
            
            col_down1, col_down2 = st.columns(2)

            # Create matplotlib figure for downloads
            fig_mpl, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(concs, response, label=T["data"], s=100, color='#ef4444', zorder=3, edgecolors='white', linewidth=2)
            ax.plot(xfit, yfit, label=T["fit"], linewidth=3, color='#3b82f6')
            ax.axvline(ic50, linestyle="--", label=f"IC₅₀ = {ic50:.4f} {unit}", color='#8b5cf6', linewidth=2)
            ax.axhline(50, linestyle=":", color='#6b7280', linewidth=1, alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlabel(T["xlab"].format(unit=unit), fontsize=12, fontweight='bold')
            ax.set_ylabel(T["ylab"], fontsize=12, fontweight='bold')
            ax.set_title(T["curve"].format(compound=compound_name or "—"), fontsize=14, fontweight='bold', color='#1e3a8a')
            ax.legend(frameon=True, shadow=True, fancybox=True)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_ylim(0, 110)
            plt.tight_layout()

            with col_down1:
                buf_png = BytesIO()
                fig_mpl.savefig(buf_png, format="png", dpi=300, bbox_inches="tight", facecolor='white')
                st.download_button(
                    T["download_png"],
                    buf_png.getvalue(),
                    file_name=f"IC50_{compound_name or 'plot'}.png",
                    mime="image/png",
                    use_container_width=True
                )

            with col_down2:
                buf_pdf = BytesIO()
                fig_mpl.savefig(buf_pdf, format="pdf", bbox_inches="tight", facecolor='white')
                st.download_button(
                    T["download_pdf"],
                    buf_pdf.getvalue(),
                    file_name=f"IC50_{compound_name or 'plot'}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            plt.close(fig_mpl)

        except Exception as e:
            st.error(T["error"])
            with st.expander("Debug Info"):
                st.write(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(f"""
<div class="info-box">
    <strong>{T['cite']}</strong><br>
    Şekerler, T. <em>IC₅₀ Calculator</em> (v1.2).<br>
    <a href="https://turgut-sekerler-ic50.streamlit.app" target="_blank">https://turgut-sekerler-ic50.streamlit.app</a>
</div>
""", unsafe_allow_html=True)
