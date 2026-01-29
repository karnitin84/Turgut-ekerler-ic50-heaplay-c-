if st.button("IC₅₀ HESAPLA"):
    try:
        control_mean = np.mean(control_vals)

        df = edited_table.copy()

        # Temizlik: NaN ve stringleri temizle
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        concentrations = df["Konsantrasyon"].values
        absorbance_vals = df.iloc[:, 1:].values

        # Konsantrasyon 0 olamaz (log-scale)
        if np.any(concentrations <= 0):
            st.error("Konsantrasyon değerleri 0 veya negatif olamaz.")
            st.stop()

        absorbance_means = absorbance_vals.mean(axis=1)

        response = (absorbance_means / control_mean) * 100

        # Başlangıç tahminleri (daha stabil)
        p0 = [
            min(response),
            max(response),
            np.exp(np.mean(np.log(concentrations))),  # geometrik ortalama
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

        bottom, top, ic50, hill = popt

        ic50_se = np.sqrt(pcov[2, 2])
        ci_low = ic50 - 1.96 * ic50_se
        ci_high = ic50 + 1.96 * ic50_se

        max_conc = np.max(concentrations)

        if ic50 > max_conc:
            st.warning(f"**Madde:** {compound_name if compound_name else '—'}  \n"
                       f"IC₅₀ > {max_conc:.4g} {unit}")
        else:
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
