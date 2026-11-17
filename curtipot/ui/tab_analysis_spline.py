"""Streamlit UI tab for experimental curve analysis using Cubic Splines (CurTiPot Análise I)."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curtipot.analysis import analyze_spline
from curtipot.models import TitrationResult
from curtipot.ui.data_loader import extract_matched_volume_ph, load_experimental_dataframe


def _generate_example_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate representative experimental titration points with slight realistic noise."""
    vols = np.array([
        0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 23.0, 24.0, 24.5,
        24.8, 25.0, 25.2, 25.5, 26.0, 27.0, 28.0, 30.0, 35.0, 40.0, 50.0,
    ], dtype=float)
    # Simulated 25 mL of 0.1 M Acetic acid (pKa 4.76) + 25 mL H2O with 0.1 M NaOH
    # Approximate realistic pH values
    phs = np.array([
        2.88, 3.65, 4.10, 4.41, 4.58, 4.74, 4.96, 5.22, 5.43, 5.76, 6.01, 6.42, 6.81,
        7.32, 8.72, 9.98, 10.51, 10.92, 11.28, 11.49, 11.75, 12.08, 12.26, 12.48,
    ], dtype=float)
    return vols, phs


def render() -> None:
    """Render the Cubic Spline experimental analysis tab."""
    st.markdown("### 📈 Análise I: Interpolação por Spline Cúbica e Derivadas")
    st.caption(
        "Ajusta uma Spline Cúbica Natural aos dados experimentais de titulação potentiométrica, "
        "calculando as derivadas contínuas (dpH/dV e d²pH/dV²) e localizando pontos de equivalência "
        "pelo refinamento da passagem por zero da 2ª derivada."
    )

    data_source = st.radio(
        "Origem dos Dados",
        options=["Importar da Última Simulação", "Upload de Arquivo (CSV / TXT)", "Carregar Dados de Exemplo"],
        horizontal=True,
    )

    v_exp: np.ndarray | None = None
    ph_exp: np.ndarray | None = None

    if data_source == "Importar da Última Simulação":
        last_titr: TitrationResult | None = st.session_state.get("last_titration")
        if last_titr is None:
            st.warning("Nenhuma simulação encontrada no simulador. Realize uma simulação ou selecione outra origem.")
        else:
            v_exp = last_titr.volumes
            ph_exp = last_titr.phs
            st.success(f"Carregados {len(v_exp)} pontos da última simulação!")

    elif data_source == "Upload de Arquivo (CSV / TXT)":
        uploaded_file = st.file_uploader(
            "Selecione um arquivo CSV ou TXT contendo colunas de Volume e pH",
            type=["csv", "txt", "tsv"],
        )
        if uploaded_file is not None:
            try:
                df = load_experimental_dataframe(uploaded_file.getvalue())
                col_v, col_ph = st.columns(2)
                with col_v:
                    v_col_name = st.selectbox("Coluna de Volume (mL)", options=list(df.columns), index=0)
                with col_ph:
                    default_ph_idx = 1 if len(df.columns) > 1 else 0
                    ph_col_name = st.selectbox("Coluna de pH", options=list(df.columns), index=default_ph_idx)

                v_exp, ph_exp = extract_matched_volume_ph(df, v_col_name, ph_col_name)
                st.success(f"Carregados {len(v_exp)} pontos do arquivo!")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")

    else:  # Carregar Dados de Exemplo
        v_exp, ph_exp = _generate_example_data()
        st.info(f"Dados de exemplo carregados: Titulação de Ácido Acético ({len(v_exp)} pontos).")

    if v_exp is None or ph_exp is None or len(v_exp) < 4:
        st.info("Aguardando carregamento de ao menos 4 pontos de dados experimentais.")
        return

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        num_grid_points = st.slider(
            "Resolução da Spline (Número de pontos da malha contínua)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
        )

    # Perform Spline Analysis
    try:
        spline_res = analyze_spline(v_exp, ph_exp, num_points=num_grid_points)
    except Exception as e:
        st.error(f"Erro na análise de spline: {e}")
        return

    st.markdown("---")
    st.subheader("📊 Resultados da Análise de Spline")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pontos Experimentais", len(v_exp))
    m2.metric("Qualidade do Ajuste (R²)", f"{spline_res.r_squared:.5f}")
    m3.metric("RMSE do Ajuste", f"{spline_res.rmse:.4e}")
    m4.metric("Pontos de Equivalência", len(spline_res.equivalence_points))

    # Equivalence Points Table
    if spline_res.equivalence_points:
        st.markdown("##### 🎯 Pontos de Equivalência Encontrados (Refinamento por d²pH/dV² = 0)")
        eq_data = [ep.to_dict() for ep in spline_res.equivalence_points]
        df_eq = pd.DataFrame(eq_data)
        st.dataframe(
            df_eq.style.format({
                "volume": "{:.3f} mL",
                "ph": "{:.3f}",
                "dpH_dV": "{:.2f}",
                "d2pH_dV2": "{:.2e}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("Nenhum ponto de inflexão/equivalência foi detectado na faixa de dados.")

    # Interactive Plots
    t_spline, t_d1, t_d2, t_tab = st.tabs(
        [
            "Ajuste da Curva (pH vs Volume)",
            "1ª Derivada Contínua (dpH/dV)",
            "2ª Derivada Contínua (d²pH/dV²)",
            "Tabela de Dados da Spline",
        ]
    )

    with t_spline:
        fig_s = go.Figure()
        # Experimental points
        fig_s.add_trace(
            go.Scatter(
                x=v_exp,
                y=ph_exp,
                mode="markers",
                name="Pontos Experimentais",
                marker=dict(size=8, color="#1f77b4", symbol="circle"),
                hovertemplate="Exp V: %{x:.2f} mL<br>Exp pH: %{y:.2f}<extra></extra>",
            )
        )
        # Continuous Spline
        fig_s.add_trace(
            go.Scatter(
                x=spline_res.interpolated_volumes,
                y=spline_res.interpolated_phs,
                mode="lines",
                name=f"Spline Cúbica (R² = {spline_res.r_squared:.4f})",
                line=dict(color="#2ca02c", width=2.5),
                hovertemplate="V: %{x:.2f} mL<br>Spline pH: %{y:.3f}<extra></extra>",
            )
        )
        # Equivalence points markers
        for ep in spline_res.equivalence_points:
            fig_s.add_trace(
                go.Scatter(
                    x=[ep.volume],
                    y=[ep.ph],
                    mode="markers",
                    name=f"V_eq = {ep.volume:.2f} mL",
                    marker=dict(size=12, color="red", symbol="star"),
                    hovertemplate=f"V_eq: {ep.volume:.3f} mL<br>pH: {ep.ph:.3f}<extra></extra>",
                )
            )

        fig_s.update_layout(
            title="Ajuste Spline Cúbica aos Dados Experimentais",
            xaxis_title="Volume de Titulante (mL)",
            yaxis_title="pH",
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with t_d1:
        fig_d1 = go.Figure()
        fig_d1.add_trace(
            go.Scatter(
                x=spline_res.interpolated_volumes,
                y=spline_res.dpH_dV,
                mode="lines",
                name="dpH/dV Contínuo",
                line=dict(color="#ff7f0e", width=2.2),
                hovertemplate="V: %{x:.2f} mL<br>dpH/dV: %{y:.2f}<extra></extra>",
            )
        )
        for ep in spline_res.equivalence_points:
            fig_d1.add_vline(
                x=ep.volume,
                line_dash="dot",
                line_color="red",
                annotation_text=f"V_eq={ep.volume:.2f} mL",
            )
        fig_d1.update_layout(
            title="Primeira Derivada Analítica da Spline (dpH/dV)",
            xaxis_title="Volume de Titulante (mL)",
            yaxis_title="dpH / dV (mL⁻¹)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_d1, use_container_width=True)

    with t_d2:
        fig_d2 = go.Figure()
        fig_d2.add_trace(
            go.Scatter(
                x=spline_res.interpolated_volumes,
                y=spline_res.d2pH_dV2,
                mode="lines",
                name="d²pH/dV² Contínuo",
                line=dict(color="#9467bd", width=2.0),
                hovertemplate="V: %{x:.2f} mL<br>d²pH/dV²: %{y:.2e}<extra></extra>",
            )
        )
        fig_d2.add_hline(y=0.0, line_dash="solid", line_color="gray")
        for ep in spline_res.equivalence_points:
            fig_d2.add_trace(
                go.Scatter(
                    x=[ep.volume],
                    y=[0.0],
                    mode="markers",
                    name=f"Passagem Zero V={ep.volume:.2f} mL",
                    marker=dict(size=9, color="red", symbol="x"),
                )
            )
        fig_d2.update_layout(
            title="Segunda Derivada Analítica da Spline (d²pH/dV²)",
            xaxis_title="Volume de Titulante (mL)",
            yaxis_title="d²pH / dV² (mL⁻²)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_d2, use_container_width=True)

    with t_tab:
        df_spline = pd.DataFrame({
            "Volume_mL": spline_res.interpolated_volumes,
            "pH_Spline": spline_res.interpolated_phs,
            "dpH_dV": spline_res.dpH_dV,
            "d2pH_dV2": spline_res.d2pH_dV2,
        })
        st.dataframe(df_spline, use_container_width=True)
        csv_bytes = df_spline.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar Curva Spline Interpolada (CSV)",
            data=csv_bytes,
            file_name="spline_analise_curtipot.csv",
            mime="text/csv",
        )


render_tab_analysis_spline = render
