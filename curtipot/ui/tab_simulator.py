"""Streamlit UI tab for potentiometric titration simulation."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curtipot.database import get_system_by_name, list_substance_names
from curtipot.models import ChemicalSystem, TitrationConfig, TitrationResult
from curtipot.titration import simulate_titration


def render() -> None:
    """Render the Potentiometric Titration Simulator tab."""
    st.markdown("### ⚖️ Simulador de Titulação Potenciométrica")
    st.caption(
        "Simula curvas de titulação considerando a diluição contínua da amostra e do titulante, "
        "com cálculo simultâneo de pH, força iônica e derivadas numéricas (dpH/dV e d²pH/dV²)."
    )

    all_names = list_substance_names()

    # Initialize session state for retained curves
    if "retained_curves" not in st.session_state:
        st.session_state["retained_curves"] = []
    if "last_titration" not in st.session_state:
        st.session_state["last_titration"] = None

    col_sample, col_titrant = st.columns(2)

    with col_sample:
        st.markdown("#### 🧪 Amostra (Titulado)")
        custom_sample = st.checkbox("Amostra personalizada", key="sim_cust_sample")
        if not custom_sample:
            default_sample_idx = 0
            if "Ácido acético" in all_names:
                default_sample_idx = all_names.index("Ácido acético")
            s_name = st.selectbox(
                "Substância da Amostra",
                options=all_names,
                index=default_sample_idx,
                key="sim_sample_select",
            )
            s_conc = st.number_input(
                "Concentração da Amostra [mol/L]",
                min_value=1e-5,
                max_value=10.0,
                value=0.100,
                step=0.01,
                format="%.4f",
                key="sim_s_conc",
            )
            try:
                sample_sys = get_system_by_name(s_name, concentration=s_conc)
            except Exception as e:
                st.error(f"Erro ao carregar amostra: {e}")
                return
        else:
            s_name = st.text_input("Nome da Amostra", value="Ácido Amostra", key="sim_cust_sname")
            s_charge = st.number_input("Carga Base", value=-1, step=1, key="sim_cust_sz")
            s_pka_str = st.text_input("pKas", value="4.757", key="sim_cust_spka")
            s_conc = st.number_input("Concentração [mol/L]", value=0.100, step=0.01, format="%.4f", key="sim_cust_sconc")
            try:
                pkas = [float(k.strip()) for k in s_pka_str.split(",") if k.strip()]
                sample_sys = ChemicalSystem(
                    name=s_name,
                    charge_base=int(s_charge),
                    pKas=pkas,
                    concentration=s_conc,
                )
            except Exception as e:
                st.error(f"Amostra inválida: {e}")
                return

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            sample_vol = st.number_input(
                "Volume da Amostra [mL]",
                min_value=0.1,
                max_value=1000.0,
                value=25.0,
                step=1.0,
                key="sim_v_sample",
            )
        with c_v2:
            water_vol = st.number_input(
                "Água Adicionada [mL]",
                min_value=0.0,
                max_value=1000.0,
                value=25.0,
                step=5.0,
                key="sim_v_water",
            )

        sample_inert_conc = st.number_input(
            "Sal inerte na Amostra (ex: NaCl) [mol/L]",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
            key="sim_s_inert",
        )

    with col_titrant:
        st.markdown("#### 💧 Titulante (Bureta)")
        custom_titrant = st.checkbox("Titulante personalizado", key="sim_cust_titrant")
        if not custom_titrant:
            default_t_idx = 0
            if "Hidróxido de sódio" in all_names:
                default_t_idx = all_names.index("Hidróxido de sódio")
            t_name = st.selectbox(
                "Substância Titulante",
                options=all_names,
                index=default_t_idx,
                key="sim_titrant_select",
            )
            t_conc = st.number_input(
                "Concentração do Titulante [mol/L]",
                min_value=1e-5,
                max_value=10.0,
                value=0.100,
                step=0.01,
                format="%.4f",
                key="sim_t_conc",
            )
            try:
                titrant_sys = get_system_by_name(t_name, concentration=t_conc)
            except Exception as e:
                st.error(f"Erro ao carregar titulante: {e}")
                return
        else:
            t_name = st.text_input("Nome do Titulante", value="Titulante Base", key="sim_cust_tname")
            t_charge = st.number_input("Carga Base Titulante", value=-1, step=1, key="sim_cust_tz")
            t_pka_str = st.text_input("pKas Titulante (vazio se forte)", value="", key="sim_cust_tpka")
            t_conc = st.number_input("Concentração Titulante [mol/L]", value=0.100, step=0.01, format="%.4f", key="sim_cust_tconc")
            try:
                pkas_t = [float(k.strip()) for k in t_pka_str.split(",") if k.strip()]
                titrant_sys = ChemicalSystem(
                    name=t_name,
                    charge_base=int(t_charge),
                    pKas=pkas_t,
                    concentration=t_conc,
                )
            except Exception as e:
                st.error(f"Titulante inválido: {e}")
                return

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            max_vol = st.number_input(
                "Volume Máximo [mL]",
                min_value=1.0,
                max_value=500.0,
                value=50.0,
                step=5.0,
                key="sim_max_vol",
            )
        with c_t2:
            step_size = st.number_input(
                "Passo de Volume [mL]",
                min_value=0.01,
                max_value=5.0,
                value=0.20,
                step=0.05,
                format="%.3f",
                key="sim_step_size",
            )

        titrant_inert_conc = st.number_input(
            "Sal inerte no Titulante [mol/L]",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
            key="sim_t_inert",
        )

    # General configuration
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        sim_mode = st.selectbox(
            "Modo de Simulação",
            options=["Passo de Volume (ΔV constante)", "Passo de pH (ΔpH constante)"],
            key="sim_mode_select",
        )
    with col_c2:
        calc_is = st.checkbox("Correção de Davies", value=True, key="sim_calc_is")
    with col_c3:
        temp_c = st.number_input("Temperatura (°C)", value=25.0, step=1.0, key="sim_temp")

    mode_code = "volume" if "Volume" in sim_mode else "ph"

    sample_inert = [(sample_inert_conc, 1), (sample_inert_conc, -1)] if sample_inert_conc > 0 else []
    titrant_inert = [(titrant_inert_conc, 1), (titrant_inert_conc, -1)] if titrant_inert_conc > 0 else []

    config = TitrationConfig(
        sample_systems=[sample_sys],
        sample_volume=sample_vol,
        titrant_systems=[titrant_sys],
        water_volume=water_vol,
        max_volume=max_vol,
        step_size=step_size,
        calc_ionic_strength=calc_is,
        temp_c=temp_c,
        mode=mode_code,
        sample_inert_salts=sample_inert,
        titrant_inert_salts=titrant_inert,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
    with col_btn1:
        run_sim = st.button("🚀 Simular Titulação", type="primary", width="stretch")
    with col_btn2:
        retain_btn = st.button("📌 Retor / Fixar Curva Atual", width="stretch")
    with col_btn3:
        clear_retained = st.button("🗑️ Limpar Curvas Retidas", width="stretch")

    if clear_retained:
        st.session_state["retained_curves"] = []
        st.success("Curvas retidas foram limpas.")

    if run_sim:
        with st.spinner("Simulando titulação e computando derivadas..."):
            try:
                res = simulate_titration(config)
                st.session_state["last_titration"] = res
            except Exception as e:
                st.error(f"Erro durante a simulação: {e}")
                return

    current_res: TitrationResult | None = st.session_state.get("last_titration")

    if retain_btn and current_res is not None:
        label = f"{sample_sys.name} ({s_conc} M) vs {titrant_sys.name} ({t_conc} M)"
        st.session_state["retained_curves"].append({"label": label, "result": current_res})
        st.success(f"Curva '{label}' adicionada à sobreposição!")

    if current_res is None:
        st.info("Clique em 'Simular Titulação' para executar o cálculo e exibir os gráficos.")
        return

    st.markdown("---")
    st.subheader("📊 Resultados da Titulação")

    # Metrics row
    eq_pts = current_res.equivalence_points
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Pontos Calculados", len(current_res.points))
    col_m1_ph_start = current_res.points[0].ph if current_res.points else 0.0
    col_m2.metric("pH Inicial", f"{col_m1_ph_start:.2f}")
    col_m3.metric("Pontos de Equivalência", len(eq_pts))
    if eq_pts:
        v_eq_main = eq_pts[0].get("volume", 0.0)
        ph_eq_main = eq_pts[0].get("ph", 0.0)
        col_m4.metric("1º V_eq / pH_eq", f"{v_eq_main:.2f} mL / {ph_eq_main:.2f}")
    else:
        col_m4.metric("1º V_eq / pH_eq", "Não detectado")

    # Equivalence Points Table
    if eq_pts:
        st.markdown("##### 🎯 Pontos de Equivalência Detectados")
        df_eq = pd.DataFrame(eq_pts)
        st.dataframe(
            df_eq.style.format({
                "volume": "{:.3f} mL",
                "ph": "{:.3f}",
                "dpH_dV": "{:.2f}",
                "d2pH_dV2": "{:.2e}",
            }),
            width="stretch",
        )

    # Plot Tabs
    tab_curve, tab_d1, tab_d2, tab_buffer_is, tab_export = st.tabs(
        [
            "Curva de pH vs Volume",
            "1ª Derivada (dpH/dV)",
            "2ª Derivada (d²pH/dV²)",
            "Força Iônica & Tampão",
            "Tabela de Dados & Exportação",
        ]
    )

    with tab_curve:
        fig_curve = go.Figure()
        # Add retained curves first
        for item in st.session_state.get("retained_curves", []):
            r_res: TitrationResult = item["result"]
            fig_curve.add_trace(
                go.Scatter(
                    x=r_res.volumes,
                    y=r_res.phs,
                    mode="lines",
                    name=item["label"],
                    line=dict(dash="dot", width=1.8),
                    opacity=0.7,
                )
            )

        # Add current active curve
        current_label = f"{sample_sys.name} vs {titrant_sys.name} (Atual)"
        fig_curve.add_trace(
            go.Scatter(
                x=current_res.volumes,
                y=current_res.phs,
                mode="lines+markers",
                name=current_label,
                line=dict(color="#1f77b4", width=2.8),
                marker=dict(size=4),
                hovertemplate="Volume: %{x:.2f} mL<br>pH: %{y:.3f}<extra></extra>",
            )
        )

        # Mark equivalence points
        for eq in eq_pts:
            v_val = eq.get("volume", 0.0)
            ph_val = eq.get("ph", 0.0)
            fig_curve.add_trace(
                go.Scatter(
                    x=[v_val],
                    y=[ph_val],
                    mode="markers",
                    name=f"V_eq = {v_val:.2f} mL",
                    marker=dict(size=11, color="red", symbol="star"),
                    hovertemplate=f"V_eq: {v_val:.3f} mL<br>pH_eq: {ph_val:.3f}<extra></extra>",
                )
            )

        fig_curve.update_layout(
            title="Curva de Titulação Potenciométrica",
            xaxis_title="Volume de Titulante Adicionado (mL)",
            yaxis_title="pH",
            yaxis=dict(range=[-0.5, 14.5]),
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_curve, width="stretch")

    with tab_d1:
        fig_d1 = go.Figure()
        fig_d1.add_trace(
            go.Scatter(
                x=current_res.volumes,
                y=current_res.dpH_dV,
                mode="lines+markers",
                name="dpH/dV",
                line=dict(color="#ff7f0e", width=2.2),
                marker=dict(size=4),
                hovertemplate="Volume: %{x:.2f} mL<br>dpH/dV: %{y:.2f}<extra></extra>",
            )
        )
        for eq in eq_pts:
            fig_d1.add_vline(
                x=eq["volume"],
                line_dash="dot",
                line_color="red",
                annotation_text=f"V_eq={eq['volume']:.2f} mL",
            )
        fig_d1.update_layout(
            title="Primeira Derivada (dpH/dV)",
            xaxis_title="Volume de Titulante Adicionado (mL)",
            yaxis_title="dpH / dV (mL⁻¹)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_d1, width="stretch")

    with tab_d2:
        fig_d2 = go.Figure()
        fig_d2.add_trace(
            go.Scatter(
                x=current_res.volumes,
                y=current_res.d2pH_dV2,
                mode="lines+markers",
                name="d²pH/dV²",
                line=dict(color="#9467bd", width=2.0),
                marker=dict(size=4),
                hovertemplate="Volume: %{x:.2f} mL<br>d²pH/dV²: %{y:.2e}<extra></extra>",
            )
        )
        fig_d2.add_hline(y=0.0, line_dash="solid", line_color="gray")
        fig_d2.update_layout(
            title="Segunda Derivada (d²pH/dV²)",
            xaxis_title="Volume de Titulante Adicionado (mL)",
            yaxis_title="d²pH / dV² (mL⁻²)",
            template="plotly_white",
            height=460,
        )
        st.plotly_chart(fig_d2, width="stretch")

    with tab_buffer_is:
        fig_bis = go.Figure()
        fig_bis.add_trace(
            go.Scatter(
                x=current_res.volumes,
                y=current_res.ionic_strengths,
                mode="lines",
                name="Força Iônica (mol/L)",
                line=dict(color="#2ca02c", width=2.2),
            )
        )
        fig_bis.add_trace(
            go.Scatter(
                x=current_res.volumes,
                y=current_res.buffer_capacities,
                mode="lines",
                name="Capacidade Tampão β",
                line=dict(color="#d62728", width=2.2, dash="dash"),
            )
        )
        fig_bis.update_layout(
            title="Evolução da Força Iônica e Capacidade de Tamponamento",
            xaxis_title="Volume de Titulante Adicionado (mL)",
            yaxis_title="Valor",
            template="plotly_white",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bis, width="stretch")

    with tab_export:
        df_out = pd.DataFrame({
            "Volume_mL": current_res.volumes,
            "pH": current_res.phs,
            "dpH_dV": current_res.dpH_dV,
            "d2pH_dV2": current_res.d2pH_dV2,
            "Buffer_Capacity": current_res.buffer_capacities,
            "Ionic_Strength": current_res.ionic_strengths,
        })
        st.dataframe(df_out, width="stretch")
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar Dados da Titulação (CSV)",
            data=csv_bytes,
            file_name=f"titulacao_{sample_sys.name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


render_tab_simulator = render
