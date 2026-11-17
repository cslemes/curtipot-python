"""Streamlit UI tab for non-linear regression titration curve fitting (CurTiPot Análise II)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curtipot.analysis import fit_titration_curve
from curtipot.database import get_system_by_name, list_substance_names
from curtipot.models import ChemicalSystem, TitrationConfig, TitrationResult
from curtipot.ui.data_loader import extract_matched_volume_ph, load_experimental_dataframe


def _generate_example_fit_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample noisy experimental points for regression fitting (Acetic acid ~0.085 M)."""
    vols = np.array([
        0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 19.0, 20.0, 20.5,
        21.0, 21.2, 21.5, 22.0, 23.0, 25.0, 30.0, 35.0,
    ], dtype=float)
    # True C = 0.0850 M (Veq ~ 21.25 mL with 25 mL sample + 25 mL H2O and 0.1 M NaOH)
    phs = np.array([
        2.92, 3.73, 4.05, 4.28, 4.47, 4.65, 4.83, 5.03, 5.28, 5.62, 5.86, 6.28, 6.64,
        7.45, 8.68, 9.75, 10.42, 10.89, 11.33, 11.75, 11.98,
    ], dtype=float)
    return vols, phs


def render() -> None:
    """Render the Non-linear Regression Fitting tab (Análise II)."""
    st.markdown("### 🎯 Análise II: Regressão Não-Linear Multimétrica (RNLMP)")
    st.caption(
        "Ajusta parâmetros desconhecidos do analito (concentração analítica e/ou constantes pKₐ) "
        "minimizando os resíduos quadráticos de pH em relação aos dados experimentais com o algoritmo de Levenberg-Marquardt / Trust-Region."
    )

    all_names = list_substance_names()

    data_source = st.radio(
        "Origem dos Dados Experimentais",
        options=["Importar da Última Simulação", "Upload de Arquivo (CSV / TXT)", "Carregar Dados de Exemplo"],
        horizontal=True,
        key="fit_data_source",
    )

    v_exp: np.ndarray | None = None
    ph_exp: np.ndarray | None = None

    if data_source == "Importar da Última Simulação":
        last_titr: TitrationResult | None = st.session_state.get("last_titration")
        if last_titr is None:
            st.warning("Nenhuma simulação anterior disponível. Simule uma titulação primeiro ou carregue dados.")
        else:
            v_exp = last_titr.volumes
            ph_exp = last_titr.phs
            st.success(f"Carregados {len(v_exp)} pontos da última simulação!")

    elif data_source == "Upload de Arquivo (CSV / TXT)":
        uploaded_file = st.file_uploader(
            "Selecione arquivo CSV/TXT para ajuste",
            type=["csv", "txt", "tsv"],
            key="fit_uploader",
        )
        if uploaded_file is not None:
            try:
                df = load_experimental_dataframe(uploaded_file.getvalue())
                col_v, col_ph = st.columns(2)
                with col_v:
                    v_col_name = st.selectbox("Coluna de Volume", options=list(df.columns), index=0, key="fit_vcol")
                with col_ph:
                    default_ph_idx = 1 if len(df.columns) > 1 else 0
                    ph_col_name = st.selectbox("Coluna de pH", options=list(df.columns), index=default_ph_idx, key="fit_phcol")

                v_exp, ph_exp = extract_matched_volume_ph(df, v_col_name, ph_col_name)
                st.success(f"Carregados {len(v_exp)} pontos do arquivo!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

    else:
        v_exp, ph_exp = _generate_example_fit_data()
        st.info(f"Dados de exemplo carregados: Titulação experimental de Ácido Acético ({len(v_exp)} pontos).")

    if v_exp is None or ph_exp is None or len(v_exp) < 3:
        st.info("Aguardando carregamento de ao menos 3 pontos experimentais.")
        return

    st.markdown("---")
    st.markdown("#### 🔬 Configuração do Modelo Químico")

    col_model1, col_model2 = st.columns(2)

    with col_model1:
        st.markdown("**Analito na Amostra (Alvo do Ajuste)**")
        default_analyte_idx = 0
        if "Ácido acético" in all_names:
            default_analyte_idx = all_names.index("Ácido acético")
        analyte_name = st.selectbox(
            "Substância do Analito",
            options=all_names,
            index=default_analyte_idx,
            key="fit_analyte_select",
        )
        c_guess = st.number_input(
            "Estimativa Inicial de Concentração [mol/L]",
            min_value=1e-5,
            max_value=10.0,
            value=0.100,
            step=0.01,
            format="%.4f",
            key="fit_c_guess",
        )

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            sample_vol = st.number_input("Volume Amostra [mL]", value=25.0, step=1.0, key="fit_svol")
        with col_v2:
            water_vol = st.number_input("Água Adicionada [mL]", value=25.0, step=5.0, key="fit_wvol")

        try:
            base_analyte = get_system_by_name(analyte_name, concentration=c_guess)
        except Exception as e:
            st.error(f"Erro ao obter analito: {e}")
            return

    with col_model2:
        st.markdown("**Titulante Conhecido**")
        default_titr_idx = 0
        if "Hidróxido de sódio" in all_names:
            default_titr_idx = all_names.index("Hidróxido de sódio")
        titr_name = st.selectbox(
            "Substância Titulante",
            options=all_names,
            index=default_titr_idx,
            key="fit_titr_select",
        )
        titr_conc = st.number_input(
            "Concentração Conhecida do Titulante [mol/L]",
            min_value=1e-5,
            max_value=10.0,
            value=0.100,
            step=0.01,
            format="%.4f",
            key="fit_titr_conc",
        )
        try:
            titrant_sys = get_system_by_name(titr_name, concentration=titr_conc)
        except Exception as e:
            st.error(f"Erro ao obter titulante: {e}")
            return

    # Fitting parameter selection
    st.markdown("#### ⚙️ Parâmetros a Otimizar")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fit_c = st.checkbox("Ajustar Concentração do Analito", value=True, key="fit_opt_c")

    with col_p2:
        pka_options = [f"pKₐ{i+1} = {k:.3f}" for i, k in enumerate(base_analyte.pKas)]
        selected_pkas_labels = st.multiselect(
            "Ajustar Constantes de Dissociação (pKₐ):",
            options=pka_options,
            default=[],
            key="fit_opt_pka",
        )
        selected_pka_indices = [pka_options.index(label) for label in selected_pkas_labels]

    if not fit_c and not selected_pka_indices:
        st.warning("Selecione ao menos um parâmetro (concentração ou pKₐ) para ajustar.")
        return

    config = TitrationConfig(
        sample_systems=[base_analyte],
        sample_volume=sample_vol,
        titrant_systems=[titrant_sys],
        water_volume=water_vol,
        max_volume=float(np.max(v_exp)),
        step_size=0.1,
    )

    run_fit = st.button("🚀 Executar Regressão Não-Linear", type="primary", width="stretch")

    if run_fit:
        with st.spinner("Executando otimização por mínimos quadrados não-lineares..."):
            try:
                fit_result = fit_titration_curve(
                    volumes=v_exp,
                    ph_measured=ph_exp,
                    config=config,
                    fit_concentration=fit_c,
                    fit_pka_indices=selected_pka_indices,
                )
                st.session_state["last_fit_result"] = fit_result
            except Exception as e:
                st.error(f"Falha na regressão não-linear: {e}")
                return

    res = st.session_state.get("last_fit_result")
    if res is None:
        st.info("Configure os parâmetros acima e clique em 'Executar Regressão Não-Linear'.")
        return

    st.markdown("---")
    st.subheader("📊 Resultados da Regressão")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Status da Convergência", "Sucesso" if res.success else "Não convergiu")
    m2.metric("Qualidade do Ajuste (R²)", f"{res.r_squared:.5f}")
    m3.metric("RMSE dos Resíduos", f"{res.rmse:.4e}")
    m4.metric("Pontos Ajustados", len(v_exp))

    # Parameter Estimates Table
    st.markdown("##### 📋 Estimativas Finais dos Parâmetros Ajustados")
    param_rows = []
    if "concentration" in res.fitted_parameters:
        val_fitted = res.fitted_parameters["concentration"]
        param_rows.append({
            "Parâmetro": "Concentração Analítica",
            "Estimativa Inicial": f"{c_guess:.5f} M",
            "Valor Otimizado": f"{val_fitted:.5f} M",
            "Variação (Δ)": f"{val_fitted - c_guess:+.5f} M",
        })

    for idx in selected_pka_indices:
        pka_key = f"pKa_{idx + 1}"
        init_val = base_analyte.pKas[idx]
        val_fitted = res.fitted_parameters.get(pka_key, init_val)
        param_rows.append({
            "Parâmetro": f"pKₐ{idx + 1}",
            "Estimativa Inicial": f"{init_val:.4f}",
            "Valor Otimizado": f"{val_fitted:.4f}",
            "Variação (Δ)": f"{val_fitted - init_val:+.4f}",
        })

    st.dataframe(pd.DataFrame(param_rows), width="stretch")

    # Interactive plots: Fitted Curve & Residuals
    tab_fit_curve, tab_residuals = st.tabs(["Curva de Ajuste Teórico vs Experimental", "Gráfico de Resíduos"])

    with tab_fit_curve:
        fig_fit = go.Figure()
        fig_fit.add_trace(
            go.Scatter(
                x=v_exp,
                y=ph_exp,
                mode="markers",
                name="pH Medido (Experimental)",
                marker=dict(size=8, color="#1f77b4", symbol="circle"),
                hovertemplate="V: %{x:.2f} mL<br>pH Exp: %{y:.2f}<extra></extra>",
            )
        )
        fig_fit.add_trace(
            go.Scatter(
                x=v_exp,
                y=res.fitted_ph,
                mode="lines",
                name=f"Curva Teórica Ajustada (R² = {res.r_squared:.5f})",
                line=dict(color="#d62728", width=2.5),
                hovertemplate="V: %{x:.2f} mL<br>pH Ajustado: %{y:.3f}<extra></extra>",
            )
        )
        fig_fit.update_layout(
            title=f"Ajuste Não-Linear: {analyte_name} Titulado com {titr_name}",
            xaxis_title="Volume de Titulante (mL)",
            yaxis_title="pH",
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fit, width="stretch")

    with tab_residuals:
        fig_res = go.Figure()
        fig_res.add_trace(
            go.Scatter(
                x=v_exp,
                y=res.residuals,
                mode="lines+markers",
                name="Resíduo (pH_calc - pH_exp)",
                marker=dict(size=7, color="#ff7f0e"),
                line=dict(width=1.5, dash="dot"),
                hovertemplate="V: %{x:.2f} mL<br>Resíduo: %{y:+.4f}<extra></extra>",
            )
        )
        fig_res.add_hline(y=0.0, line_dash="solid", line_color="black")
        fig_res.update_layout(
            title="Distribuição dos Resíduos ao Longo da Titulação",
            xaxis_title="Volume de Titulante (mL)",
            yaxis_title="Resíduo de pH (calc - exp)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_res, width="stretch")


render_tab_analysis_fit = render
